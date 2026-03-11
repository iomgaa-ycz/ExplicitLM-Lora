"""
pipeline.py — ExplicitLMPipeline 端到端推理管线

职责：串联 MemoryRouter 检索 → KnowledgeEncoder 编码 → ModifiedQwen 生成的完整推理流程，
提供统一对外接口：
  - answer(): query text → 预测答案（演示用）
  - evaluate_loglikelihood(): 多选题 loglikelihood 评测（评测核心）
  - from_checkpoints(): 从 checkpoint 目录快速加载完整管线

推理数据流：
    query str
      ↓ tokenize + KnowledgeEncoder.encode_mean → q_emb [1, D]
      ↓ MemoryRouter.retrieve(q_emb, store) → knowledge_ids [1, K_f]
      ↓ tokenize query → input_ids [1, L]
      ↓ ModifiedQwen(input_ids, knowledge_ids, attention_mask) → logits [1, L, V]
      ↓ greedy decode / loglikelihood → answer str / predicted choice idx

Checkpoint 目录约定（.pt 格式，与 DualKnowledgeStore.save_state 一致）：
    router_ckpt/
        router.pt              # MemoryRouter.state_dict()（pkm + adapter + selector）
    fusion_ckpt/
        injection_modules.pt   # injection_modules.state_dict()
        encoder_layers.pt      # KnowledgeEncoder.layers.state_dict()（Phase 2 解冻后，可选）
        encoder_norm.pt        # KnowledgeEncoder.norm.state_dict()（可选）

若文件不存在，则 log WARNING 后跳过（便于训练管线未完成时测试推理逻辑）。

依赖：config.py, models/, router/, transformers
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from models import AttentionInjection, KnowledgeEncoder, ModifiedQwen, load_base_model
from router.memory_bank import DualKnowledgeStore
from router.model import MemoryRouter

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 输出数据类
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineOutput:
    """
    ExplicitLMPipeline.answer() 的返回值。

    参数：
        answer:       模型预测的答案文本（greedy decode）
        retrieved_id: 路由检索到的知识条目全局 ID（Oracle 模式下为 oracle_map 查到的 ID）
        latency_ms:   本次推理总耗时（毫秒），含检索 + 编码 + 生成全流程
    """

    answer: str
    retrieved_id: int
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# 管线主类
# ─────────────────────────────────────────────────────────────────────────────


class ExplicitLMPipeline:
    """
    端到端推理管线（无梯度），将 Router 检索与 ModifiedQwen 生成整合为统一接口。

    使用方式（两种构建路径）：
        1. 直接传入已构建组件（测试/训练循环中使用）：
               pipeline = ExplicitLMPipeline(config, modified_qwen, router, store, tokenizer)

        2. 从 checkpoint 目录加载（推理 / 评测时使用）：
               pipeline = ExplicitLMPipeline.from_checkpoints(
                   config, router_ckpt="checkpoints/phase1_best",
                   fusion_ckpt="checkpoints/phase2_best", store_path="data/store.pt"
               )

    Oracle 模式：
        传入 oracle_map: Dict[str, int]（question text → knowledge_id），
        调用 answer(question, use_real_router=False) 时直接查表取知识，
        用于上限实验（E4 G2/G3）。

    参数：
        config:        全局 Config 实例
        modified_qwen: 已构建的 ModifiedQwen 实例（eval 模式）
        router:        已构建的 MemoryRouter 实例（eval 模式）
        store:         已构建的 DualKnowledgeStore 实例
        tokenizer:     Qwen3 tokenizer
        oracle_map:    可选，question text → knowledge_id 映射（Oracle 模式专用）
    """

    def __init__(
        self,
        config: "Config",
        modified_qwen: ModifiedQwen,
        router: MemoryRouter,
        store: DualKnowledgeStore,
        tokenizer: AutoTokenizer,
        oracle_map: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        初始化 ExplicitLMPipeline，存储所有子组件引用。

        参数：
            config:        全局配置（ModelConfig.hidden_dim, fusion_length 等）
            modified_qwen: Hook 注入式融合模型，应已处于 eval 模式
            router:        知识路由器，应已处于 eval 模式
            store:         双存储知识库，需已完成 compact_and_recluster
            tokenizer:     与 base_model 配套的 tokenizer
            oracle_map:    question text → knowledge_id 字典（可选）

        返回：
            None
        """
        self._config = config
        self._modified_qwen = modified_qwen
        self._router = router
        self._store = store
        self._tokenizer = tokenizer
        self._oracle_map: Optional[Dict[str, int]] = oracle_map

        # 从 modified_qwen 内部获取 knowledge_encoder 引用（供 _embed_query 使用）
        self._encoder: KnowledgeEncoder = modified_qwen.knowledge_encoder

        logger.info(
            "ExplicitLMPipeline 初始化完毕 (oracle_map=%s)",
            "已提供" if oracle_map is not None else "未提供",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 类方法：从 checkpoint 加载
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        config: "Config",
        router_ckpt: str,
        fusion_ckpt: str,
        store_path: str,
        device: str = "cpu",
        oracle_map: Optional[Dict[str, int]] = None,
    ) -> "ExplicitLMPipeline":
        """
        从 checkpoint 目录加载完整管线。

        Checkpoint 目录约定：
            router_ckpt/
                router.pt              # MemoryRouter.state_dict()
            fusion_ckpt/
                injection_modules.pt   # injection_modules.state_dict()
                encoder_layers.pt      # KnowledgeEncoder.layers（可选，Phase 2 解冻后）
                encoder_norm.pt        # KnowledgeEncoder.norm（可选）

        若上述可选文件不存在，打印 WARNING 后跳过（使用随机初始化权重）。

        参数：
            config:      全局 Config 实例
            router_ckpt: Phase 1 最优权重目录路径
            fusion_ckpt: Phase 2/3 最优权重目录路径
            store_path:  DualKnowledgeStore 状态文件路径（.pt）
            device:      推理设备，如 "cpu" 或 "cuda:0"
            oracle_map:  可选 Oracle 映射字典

        返回：
            ExplicitLMPipeline 实例（所有组件已加载至 device，处于 eval 模式）

        异常：
            FileNotFoundError: store_path 不存在时
        """
        mc = config.model
        rc = config.router
        tc = config.train

        logger.info("开始从 checkpoint 加载管线 (device=%s)", device)

        # Phase 1: 加载基础模型 + 构建 KnowledgeEncoder
        base_model = load_base_model(mc.base_model, bf16=tc.bf16)
        base_model = base_model.to(device)

        encoder = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=mc.encoder_depth,
            hidden_dim=mc.hidden_dim,
        )

        # Phase 2: 加载可选的 encoder checkpoint
        _load_state_dict_if_exists(
            module=encoder.layers,
            path=Path(fusion_ckpt) / "encoder_layers.pt",
            label="KnowledgeEncoder.layers",
        )
        _load_state_dict_if_exists(
            module=encoder.norm,
            path=Path(fusion_ckpt) / "encoder_norm.pt",
            label="KnowledgeEncoder.norm",
        )
        encoder = encoder.to(device).eval()

        # Phase 3: 构建注入模块 + 加载 fusion checkpoint
        num_injection = len(mc.injection_layers)
        injection_modules = nn.ModuleList(
            [AttentionInjection(hidden_dim=mc.hidden_dim) for _ in range(num_injection)]
        )
        _load_state_dict_if_exists(
            module=injection_modules,
            path=Path(fusion_ckpt) / "injection_modules.pt",
            label="injection_modules",
        )
        injection_modules = injection_modules.to(device)

        # Phase 4: 构建 ModifiedQwen
        modified_qwen = ModifiedQwen(
            base_model=base_model,
            knowledge_encoder=encoder,
            injection_modules=injection_modules,
            injection_layers=mc.injection_layers,
        )
        modified_qwen = modified_qwen.to(device).eval()

        # Phase 5: 构建 MemoryRouter + 加载 router checkpoint
        router = MemoryRouter(config=rc, encoder=encoder)
        _load_state_dict_if_exists(
            module=router,
            path=Path(router_ckpt) / "router.pt",
            label="MemoryRouter",
        )
        router = router.to(device).eval()

        # Phase 6: 构建 DualKnowledgeStore + 从文件加载状态
        store_file = Path(store_path)
        if not store_file.exists():
            raise FileNotFoundError(f"DualKnowledgeStore 状态文件不存在: {store_path}")

        store = DualKnowledgeStore(
            config=rc,
            fusion_length=mc.fusion_length,
            anchor_length=mc.anchor_length,
            device=device,
        )
        store.load_state(str(store_file))
        logger.info("DualKnowledgeStore 加载完毕 (next_free=%d)", store.next_free)

        # Phase 7: 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(mc.base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info("管线加载完毕，device=%s", device)
        return cls(
            config=config,
            modified_qwen=modified_qwen,
            router=router,
            store=store,
            tokenizer=tokenizer,
            oracle_map=oracle_map,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 私有辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_query(self, question: str) -> torch.Tensor:
        """
        将问题文本编码为稠密向量（query embedding）。

        实现：tokenize → KnowledgeEncoder.encode_mean（双向注意力 + mean pool）→ [1, D]

        说明：复用 KnowledgeEncoder 而非基础模型的 causal LM 隐藏状态，
        因为 encode_mean 使用双向注意力，语义质量更高，且与路由器训练时一致。

        参数：
            question: 问题文本（原始字符串）

        返回：
            [1, D] FloatTensor，问题的稠密 embedding
        """
        enc = self._tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self._config.model.anchor_length,
        )
        input_ids = enc["input_ids"].to(self._encoder.device)       # [1, L]
        attention_mask = enc["attention_mask"].to(self._encoder.device)  # [1, L]

        with torch.no_grad():
            q_emb = self._encoder.encode_mean(input_ids, attention_mask)  # [1, D]

        return q_emb

    def _get_device(self) -> torch.device:
        """返回管线当前所在设备（取 encoder 的 device）。"""
        return self._encoder.device

    # ─────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def answer(
        self,
        question: str,
        use_real_router: bool = True,
    ) -> PipelineOutput:
        """
        端到端推理：query text → 预测答案（演示 / 定性分析用）。

        注意：此方法使用 greedy argmax 解码最后一个 token 位置，仅用于演示。
        多选题评测请使用 evaluate_loglikelihood()，其 loglikelihood 方法更准确。

        流程：
            1. _embed_query(question) → q_emb [1, D]
            2. router.retrieve(q_emb, store) 或 oracle_map 查表 → knowledge_ids [1, K_f]
            3. tokenize question → input_ids [1, L], attention_mask [1, L]
            4. modified_qwen(input_ids, knowledge_ids, attention_mask) → logits [1, L, V]
            5. logits[0, -1, :].argmax() → 解码为 answer str

        参数：
            question:       用户问题（原始文本）
            use_real_router: True=真实路由检索，False=Oracle 模式（需提供 oracle_map）

        返回：
            PipelineOutput(answer, retrieved_id, latency_ms)

        异常：
            ValueError: use_real_router=False 且 oracle_map 未提供时
            KeyError:   use_real_router=False 但 question 不在 oracle_map 中时
        """
        t0 = time.perf_counter()
        device = self._get_device()

        # Phase 1: 检索 knowledge_ids
        retrieved_id: int
        if use_real_router:
            q_emb = self._embed_query(question)                          # [1, D]
            knowledge_ids = self._router.retrieve(q_emb, self._store)   # [1, K_f]
            retrieved_id = self._router.forward(q_emb, self._store).best_id[0].item()
        else:
            if self._oracle_map is None:
                raise ValueError(
                    "use_real_router=False 时必须提供 oracle_map，当前 oracle_map=None"
                )
            if question not in self._oracle_map:
                raise KeyError(
                    f"question 不在 oracle_map 中，请检查映射表（question 前 50 字符: "
                    f"'{question[:50]}'）"
                )
            retrieved_id = self._oracle_map[question]
            id_tensor = torch.tensor([retrieved_id], dtype=torch.long, device=device)
            knowledge_ids = self._store.fusion_bank[id_tensor]           # [1, K_f]

        # Phase 2: tokenize question → input_ids
        enc = self._tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)           # [1, L]
        attention_mask = enc["attention_mask"].to(device)  # [1, L]

        # Phase 3: forward → logits
        output = self._modified_qwen(
            input_ids=input_ids,
            knowledge_ids=knowledge_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        # [1, L, V] → 取最后位置的 argmax → 解码为 token
        next_token_id = output.logits[0, -1, :].argmax(dim=-1, keepdim=True)  # [1]
        answer_str = self._tokenizer.decode(next_token_id, skip_special_tokens=True)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "answer() 完成: retrieved_id=%d, answer='%s', latency=%.1f ms",
            retrieved_id,
            answer_str,
            latency_ms,
        )
        return PipelineOutput(
            answer=answer_str,
            retrieved_id=retrieved_id,
            latency_ms=latency_ms,
        )

    @torch.no_grad()
    def evaluate_loglikelihood(
        self,
        question: str,
        choices: List[str],
        knowledge_ids: torch.Tensor,
    ) -> int:
        """
        多选题 loglikelihood 评测（核心评测接口）。

        对每个 choice 计算 continuation 的 log-probability，返回最高者的索引。

        实现：
            对每个 choice_i:
                1. q_ids   = tokenize(question, add_special_tokens=True)
                2. c_ids   = tokenize(" " + choice_i, add_special_tokens=False)  # 前缀空格
                3. full_ids = q_ids + c_ids  → [1, L_q + L_c]
                4. labels   = [-100 × L_q, c_ids]（只计算 continuation 部分的 loss）
                5. output   = modified_qwen(full_ids, knowledge_ids, attention_mask, labels)
                6. log_prob = -output.loss.item() × len(c_ids)  （还原为 sum，非 mean）
            return argmax(log_probs)

        参数：
            question:     问题文本
            choices:      候选答案列表，如 ["A", "B", "C", "D"] 或完整选项文本
            knowledge_ids: [1, K_f] 已检索的知识 token IDs（来自 router.retrieve 或 oracle）

        返回：
            int，预测的选项索引 (0-based)，值域 [0, len(choices))

        异常：
            AssertionError: knowledge_ids 形状不合法
        """
        assert knowledge_ids.ndim == 2, (
            f"knowledge_ids 必须为 2D [B, K_f]，实际 ndim={knowledge_ids.ndim}"
        )

        device = self._get_device()
        knowledge_ids = knowledge_ids.to(device)

        # Phase 1: tokenize question（带 BOS）
        q_ids_list: List[int] = self._tokenizer.encode(
            question, add_special_tokens=True
        )

        log_probs: List[float] = []

        # Phase 2: 逐选项计算 log-prob
        for choice in choices:
            # continuation token IDs（无 BOS，前缀空格与 LM 训练对齐）
            c_ids_list: List[int] = self._tokenizer.encode(
                " " + choice, add_special_tokens=False
            )
            num_cont = len(c_ids_list)

            if num_cont == 0:
                # 空 continuation 给 -inf，防止异常选项入选
                log_probs.append(float("-inf"))
                continue

            # 拼接全文 token IDs
            full_ids_list = q_ids_list + c_ids_list
            full_ids = torch.tensor([full_ids_list], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(full_ids)

            # labels：question 部分置 -100，只对 continuation 计算 loss
            labels = full_ids.clone()
            labels[:, : len(q_ids_list)] = -100

            output = self._modified_qwen(
                input_ids=full_ids,
                knowledge_ids=knowledge_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # output.loss = mean CE over num_cont tokens，还原为 sum（log-prob）
            log_prob = -output.loss.item() * num_cont
            log_probs.append(log_prob)

        best_idx = int(torch.tensor(log_probs, dtype=torch.float).argmax().item())
        logger.debug(
            "evaluate_loglikelihood: log_probs=%s → predicted=%d",
            [f"{p:.2f}" for p in log_probs],
            best_idx,
        )
        return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# 模块内部辅助函数
# ─────────────────────────────────────────────────────────────────────────────


def _load_state_dict_if_exists(
    module: nn.Module,
    path: Path,
    label: str,
) -> None:
    """
    若 path 存在则加载 state_dict，否则打印 WARNING 并跳过。

    参数：
        module: 目标 nn.Module
        path:   .pt 文件路径
        label:  日志标识符（如 "MemoryRouter"）

    返回：
        None
    """
    if path.exists():
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=False)
        logger.info("%s checkpoint 已加载：%s", label, path)
    else:
        logger.warning("%s checkpoint 不存在（跳过，使用随机初始化）：%s", label, path)
