"""
tests/unit/test_pipeline.py — ExplicitLMPipeline 单元测试

使用 MagicMock 模拟所有子组件（不加载真实模型），快速验证：
  - answer() 返回 PipelineOutput，字段类型正确
  - Oracle 模式（use_real_router=False）正确使用 oracle_map
  - Oracle 模式缺少 oracle_map 时 raise ValueError
  - latency_ms > 0
  - evaluate_loglikelihood() 返回合法 idx，选最低 loss 的 choice
  - _embed_query() 返回 [1, D] Tensor

测试策略：
  - 所有模型组件通过 MagicMock 构造，无 GPU / 真实权重依赖
  - 通过控制 mock 返回值构造可预期的场景

常量：HIDDEN_DIM=1024, FUSION_LENGTH=64, VOCAB_SIZE=32000, D=1024
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import ExplicitLMPipeline, PipelineOutput, _load_state_dict_if_exists  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 1024
FUSION_LENGTH = 64
VOCAB_SIZE = 32000
B = 1
L = 8


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures：构造 Mock 组件
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    """构造最小化 Config mock。"""
    cfg = MagicMock()
    cfg.model.hidden_dim = HIDDEN_DIM
    cfg.model.fusion_length = FUSION_LENGTH
    cfg.model.anchor_length = 128
    cfg.model.injection_layers = [6, 12, 18, 24]
    return cfg


@pytest.fixture
def mock_encoder():
    """
    构造 KnowledgeEncoder mock。

    encode_mean 返回随机 [1, HIDDEN_DIM] Tensor（模拟真实 embedding）。
    device 属性返回 cpu。
    """
    enc = MagicMock()
    enc.device = torch.device("cpu")
    enc.encode_mean.return_value = torch.randn(1, HIDDEN_DIM)
    return enc


@pytest.fixture
def mock_modified_qwen(mock_encoder):
    """
    构造 ModifiedQwen mock。

    forward 返回 MagicMock，其 .logits 为随机 [1, L, VOCAB_SIZE] Tensor，
    .loss 为随机标量（供 evaluate_loglikelihood 测试使用）。
    knowledge_encoder 引用 mock_encoder。
    """
    model = MagicMock()
    model.knowledge_encoder = mock_encoder

    # 每次调用 forward 时动态生成随机 logits（通过 side_effect）
    def _fake_forward(input_ids, knowledge_ids, attention_mask, labels=None):
        seq_len = input_ids.shape[1]
        out = MagicMock()
        out.logits = torch.randn(1, seq_len, VOCAB_SIZE)
        # 若提供 labels，模拟一个随机正数 loss
        out.loss = torch.tensor(torch.rand(1).item() + 0.5)  # (0.5, 1.5) 范围
        return out

    # MagicMock 必须通过 side_effect 设置可调用行为，不能用实例级 __call__ 赋值
    model.side_effect = _fake_forward
    return model


@pytest.fixture
def mock_router():
    """
    构造 MemoryRouter mock。

    retrieve 返回随机 [1, FUSION_LENGTH] knowledge_ids。
    forward 返回 RouterOutput-like mock，其 best_id = [42]。
    """
    router = MagicMock()
    router.retrieve.return_value = torch.randint(1, 1000, (1, FUSION_LENGTH))

    router_output = MagicMock()
    router_output.best_id = torch.tensor([42], dtype=torch.long)
    router.forward.return_value = router_output
    return router


@pytest.fixture
def mock_store():
    """
    构造 DualKnowledgeStore mock。

    fusion_bank[id_tensor] 返回随机 [1, FUSION_LENGTH] Tensor。
    """
    store = MagicMock()
    store.fusion_bank.__getitem__ = MagicMock(
        return_value=torch.randint(1, 1000, (1, FUSION_LENGTH))
    )
    return store


@pytest.fixture
def mock_tokenizer():
    """
    构造 AutoTokenizer mock。

    __call__ 返回 {input_ids: [1, L], attention_mask: [1, L]}。
    encode 返回固定长度 token ID 列表。
    decode 返回简单字符串。
    pad_token_id = 0。
    """
    tok = MagicMock()
    tok.pad_token_id = 0

    # tokenizer(text, return_tensors="pt", ...) 返回 dict
    def _fake_call(**kwargs):
        return {
            "input_ids": torch.randint(1, 1000, (1, L)),
            "attention_mask": torch.ones(1, L, dtype=torch.long),
        }

    tok.side_effect = None
    tok.__call__ = MagicMock(side_effect=lambda *a, **kw: _fake_call(**kw))

    # tokenizer.encode(text, add_special_tokens=...) 返回固定列表
    tok.encode.side_effect = lambda text, **kw: list(range(1, L + 1))

    # tokenizer.decode → 固定字符串
    tok.decode.return_value = "A"

    return tok


@pytest.fixture
def pipeline(mock_config, mock_modified_qwen, mock_router, mock_store, mock_tokenizer):
    """构造不含 oracle_map 的标准 ExplicitLMPipeline。"""
    return ExplicitLMPipeline(
        config=mock_config,
        modified_qwen=mock_modified_qwen,
        router=mock_router,
        store=mock_store,
        tokenizer=mock_tokenizer,
        oracle_map=None,
    )


@pytest.fixture
def oracle_pipeline(
    mock_config, mock_modified_qwen, mock_router, mock_store, mock_tokenizer
):
    """构造含 oracle_map 的 ExplicitLMPipeline。"""
    oracle_map = {"What is the capital of France?": 7}
    return ExplicitLMPipeline(
        config=mock_config,
        modified_qwen=mock_modified_qwen,
        router=mock_router,
        store=mock_store,
        tokenizer=mock_tokenizer,
        oracle_map=oracle_map,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineAnswer:
    """验证 answer() 接口行为。"""

    def test_answer_real_router_returns_output(self, pipeline: ExplicitLMPipeline) -> None:
        """
        测试：answer(use_real_router=True) 应返回 PipelineOutput，各字段类型正确。

        验证点：
            - 返回类型为 PipelineOutput
            - answer 为 str
            - retrieved_id 为 int
            - latency_ms 为 float
        """
        result = pipeline.answer("What is the capital of France?", use_real_router=True)

        assert isinstance(result, PipelineOutput)
        assert isinstance(result.answer, str)
        assert isinstance(result.retrieved_id, int)
        assert isinstance(result.latency_ms, float)

    def test_answer_oracle_mode_uses_oracle_map(
        self, oracle_pipeline: ExplicitLMPipeline, mock_router: MagicMock
    ) -> None:
        """
        测试：use_real_router=False 时应查 oracle_map，retrieved_id = oracle_map 中的值，
        且 router.retrieve() 不应被调用。

        验证点：
            - retrieved_id == oracle_map 中对应的 knowledge_id（7）
            - mock_router.retrieve 未被调用
        """
        result = oracle_pipeline.answer(
            "What is the capital of France?", use_real_router=False
        )

        assert result.retrieved_id == 7
        mock_router.retrieve.assert_not_called()

    def test_answer_oracle_mode_no_map_raises(self, pipeline: ExplicitLMPipeline) -> None:
        """
        测试：oracle_map=None 时调用 answer(use_real_router=False) 应 raise ValueError。

        验证点：
            - 抛出 ValueError
            - 错误信息包含 "oracle_map"
        """
        with pytest.raises(ValueError, match="oracle_map"):
            pipeline.answer("Any question", use_real_router=False)

    def test_answer_latency_is_positive(self, pipeline: ExplicitLMPipeline) -> None:
        """
        测试：answer() 返回的 latency_ms 应为正数。

        验证点：
            - latency_ms > 0
        """
        result = pipeline.answer("Test question", use_real_router=True)
        assert result.latency_ms > 0.0, f"latency_ms 应 > 0，实际: {result.latency_ms}"


class TestLoglikelihood:
    """验证 evaluate_loglikelihood() 接口行为。"""

    def test_evaluate_returns_valid_idx(self, pipeline: ExplicitLMPipeline) -> None:
        """
        测试：evaluate_loglikelihood 返回值应为合法 int，值域 [0, len(choices))。

        验证点：
            - 返回类型为 int
            - 值在 [0, 4) 内（4选题）
        """
        choices = ["Paris", "Berlin", "London", "Tokyo"]
        knowledge_ids = torch.randint(1, 1000, (1, FUSION_LENGTH))

        idx = pipeline.evaluate_loglikelihood(
            question="What is the capital of France?",
            choices=choices,
            knowledge_ids=knowledge_ids,
        )

        assert isinstance(idx, int), f"返回类型应为 int，实际: {type(idx)}"
        assert 0 <= idx < len(choices), f"idx={idx} 超出范围 [0, {len(choices)})"

    def test_evaluate_selects_highest_logprob(
        self,
        mock_config,
        mock_encoder,
        mock_router,
        mock_store,
        mock_tokenizer,
    ) -> None:
        """
        测试：evaluate_loglikelihood 应返回 log-prob 最高（loss 最小）的 choice 索引。

        设计：构造 ModifiedQwen mock，使其按 choice 顺序依次返回递减的 loss 值，
        验证返回的索引确实是 loss 最小的那个（第 0 个）。

        验证点：
            - 返回 idx = 0（第一个 choice loss 最小，log-prob 最大）
        """
        # loss 值：choice 0 → 0.1（最小 loss = 最高 log-prob），choice 1/2/3 递增
        loss_values = [0.1, 0.5, 0.8, 1.2]
        call_count = [0]

        def _controlled_forward(input_ids, knowledge_ids, attention_mask, labels=None):
            out = MagicMock()
            seq_len = input_ids.shape[1]
            out.logits = torch.randn(1, seq_len, VOCAB_SIZE)
            loss_idx = min(call_count[0], len(loss_values) - 1)
            out.loss = torch.tensor(loss_values[loss_idx])
            call_count[0] += 1
            return out

        controlled_model = MagicMock()
        controlled_model.knowledge_encoder = mock_encoder
        # MagicMock 必须通过 side_effect 设置可调用行为，不能用实例级 __call__ 赋值
        controlled_model.side_effect = _controlled_forward

        p = ExplicitLMPipeline(
            config=mock_config,
            modified_qwen=controlled_model,
            router=mock_router,
            store=mock_store,
            tokenizer=mock_tokenizer,
        )

        knowledge_ids = torch.randint(1, 1000, (1, FUSION_LENGTH))
        choices = ["Paris", "Berlin", "London", "Tokyo"]

        idx = p.evaluate_loglikelihood(
            question="What is the capital of France?",
            choices=choices,
            knowledge_ids=knowledge_ids,
        )

        assert idx == 0, (
            f"最低 loss 为 choices[0]={choices[0]}（loss={loss_values[0]}），"
            f"但预测 idx={idx}"
        )


class TestEmbedQuery:
    """验证 _embed_query() 内部辅助方法。"""

    def test_embed_query_shape(
        self,
        mock_config,
        mock_encoder,
        mock_modified_qwen,
        mock_router,
        mock_store,
        mock_tokenizer,
    ) -> None:
        """
        测试：_embed_query() 应返回 [1, HIDDEN_DIM] Tensor。

        验证点：
            - 返回 shape == (1, HIDDEN_DIM)
            - 类型为 torch.Tensor
        """
        # mock_encoder.encode_mean 已设置返回 [1, HIDDEN_DIM]
        p = ExplicitLMPipeline(
            config=mock_config,
            modified_qwen=mock_modified_qwen,
            router=mock_router,
            store=mock_store,
            tokenizer=mock_tokenizer,
        )

        emb = p._embed_query("What is the capital of France?")

        assert isinstance(emb, torch.Tensor), "返回类型应为 torch.Tensor"
        assert emb.shape == (1, HIDDEN_DIM), (
            f"embedding shape {emb.shape} != 预期 (1, {HIDDEN_DIM})"
        )


class TestEdgeCases:
    """覆盖边界分支：KeyError、空 continuation、_load_state_dict_if_exists。"""

    def test_answer_oracle_mode_key_not_found(
        self, mock_config, mock_modified_qwen, mock_router, mock_store, mock_tokenizer
    ) -> None:
        """
        测试：use_real_router=False 且 oracle_map 中找不到 question 时，raise KeyError。

        验证点：
            - 抛出 KeyError
        """
        oracle_map = {"Other question": 3}
        p = ExplicitLMPipeline(
            config=mock_config,
            modified_qwen=mock_modified_qwen,
            router=mock_router,
            store=mock_store,
            tokenizer=mock_tokenizer,
            oracle_map=oracle_map,
        )

        with pytest.raises(KeyError):
            p.answer("What is the capital of France?", use_real_router=False)

    def test_evaluate_empty_continuation_returns_not_empty_choice(
        self,
        mock_config,
        mock_modified_qwen,
        mock_router,
        mock_store,
    ) -> None:
        """
        测试：当某个 choice 的 continuation 编码为空时，该 choice 获得 -inf，
        不影响其他非空 choice 的评测，最终返回非空 choice 的索引。

        验证点：
            - empty choice（索引 0）获得 -inf
            - 非空 choice（索引 1）获得正常 log-prob
            - 返回 idx = 1（非空选项）
        """
        # 构造 tokenizer：第一次 encode（question）返回普通 ids，
        # 第二次（empty choice）返回空列表，第三次（normal choice）返回 [1,2,3]
        tok = MagicMock()
        tok.pad_token_id = 0
        encode_calls = [0]

        def _encode_side_effect(text, **kw):
            n = encode_calls[0]
            encode_calls[0] += 1
            if n == 0:
                return [1, 2, 3, 4]   # question
            elif n == 1:
                return []              # empty choice → triggers -inf
            else:
                return [5, 6]          # normal choice

        tok.encode.side_effect = _encode_side_effect
        tok.decode.return_value = "A"
        tok.__call__ = MagicMock(return_value={
            "input_ids": torch.randint(1, 100, (1, L)),
            "attention_mask": torch.ones(1, L, dtype=torch.long),
        })

        p = ExplicitLMPipeline(
            config=mock_config,
            modified_qwen=mock_modified_qwen,
            router=mock_router,
            store=mock_store,
            tokenizer=tok,
        )

        knowledge_ids = torch.randint(1, 1000, (1, FUSION_LENGTH))
        # choices[0] 编码为空，choices[1] 编码为 [5, 6]
        idx = p.evaluate_loglikelihood(
            question="Test?",
            choices=["", "Paris"],
            knowledge_ids=knowledge_ids,
        )

        # 空 choice 获得 -inf，所以 idx 必须是 1
        assert idx == 1, f"空 continuation 应获得 -inf，预测 idx 应为 1，实际 {idx}"

    def test_load_state_dict_if_exists_loads_when_file_present(self) -> None:
        """
        测试：_load_state_dict_if_exists 在文件存在时正确加载权重。

        验证点：
            - 加载后 target 模块的权重与保存的 state_dict 一致
        """
        import tempfile

        linear = torch.nn.Linear(4, 4)
        state = linear.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state, f.name)
            temp_path = Path(f.name)

        try:
            target_module = torch.nn.Linear(4, 4)
            _load_state_dict_if_exists(target_module, temp_path, label="TestLinear")

            for key in state:
                assert torch.allclose(
                    target_module.state_dict()[key], state[key]
                ), f"权重 {key} 加载后不一致"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_state_dict_if_exists_warns_when_file_missing(self) -> None:
        """
        测试：_load_state_dict_if_exists 在文件不存在时不 raise，只 log WARNING。

        验证点：
            - 不抛出任何异常
            - 模块权重未改变（随机初始化值）
        """
        module = torch.nn.Linear(4, 4)
        original_weight = module.weight.data.clone()
        missing_path = Path("/nonexistent_dir_xyz/missing_file.pt")

        # 不应 raise
        _load_state_dict_if_exists(module, missing_path, label="TestMissing")

        # 权重未被修改
        assert torch.allclose(module.weight.data, original_weight)

    def test_from_checkpoints_raises_when_store_missing(
        self, mock_config
    ) -> None:
        """
        测试：from_checkpoints 在 store_path 不存在时 raise FileNotFoundError。

        通过 patch 重型模型加载操作，仅测试 store_path 不存在时的异常路径。
        这同时覆盖 from_checkpoints 中模型构建的前半段代码。

        验证点：
            - 抛出 FileNotFoundError，错误信息包含 store_path
        """
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_base.eval.return_value = mock_base

        mock_enc = MagicMock()
        mock_enc.to.return_value = mock_enc
        mock_enc.eval.return_value = mock_enc
        mock_enc.layers = MagicMock()
        mock_enc.norm = MagicMock()

        mock_inj = MagicMock()
        mock_inj.to.return_value = mock_inj

        mock_qwen = MagicMock()
        mock_qwen.to.return_value = mock_qwen
        mock_qwen.eval.return_value = mock_qwen

        mock_rtr = MagicMock()
        mock_rtr.to.return_value = mock_rtr
        mock_rtr.eval.return_value = mock_rtr

        mock_store_inst = MagicMock()

        with (
            patch("pipeline.load_base_model", return_value=mock_base),
            patch("pipeline.KnowledgeEncoder", return_value=mock_enc),
            patch("pipeline.AttentionInjection", return_value=MagicMock()),
            patch("pipeline.nn.ModuleList", return_value=mock_inj),
            patch("pipeline.ModifiedQwen", return_value=mock_qwen),
            patch("pipeline.MemoryRouter", return_value=mock_rtr),
            patch("pipeline.DualKnowledgeStore", return_value=mock_store_inst),
            patch("pipeline.AutoTokenizer"),
        ):
            with pytest.raises(FileNotFoundError, match="不存在"):
                ExplicitLMPipeline.from_checkpoints(
                    config=mock_config,
                    router_ckpt="/nonexistent/router",
                    fusion_ckpt="/nonexistent/fusion",
                    store_path="/nonexistent/store.pt",
                    device="cpu",
                )
