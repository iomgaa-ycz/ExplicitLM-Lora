# 技术方案（TD）— ExplicitLM-LoRA

## 技术决策

- 基础模型（Qwen3）**完全冻结**，所有可训练参数集中在外挂模块（Router ~15M + Fusion ~20M），保护原始语言能力。
- 知识**双路存储**：Fusion Bank 存 LLMLingua 压缩后的高密度 facts（64 tokens/条），Anchor Bank 存原文截断（128 tokens/条），两者服务于不同目的（注入 vs 路由索引）。
- 路由与融合**解耦**：MemoryRouter 负责"找什么"（检索），AttentionInjection 负责"怎么用"（注入），两套参数独立训练。
- 注入模块采用**零初始化**策略（out_proj 权重初始为 0），训练初期退化为原始模型，避免初始化震荡。
- **两阶段训练**：Phase 2 在通用语料上学习融合能力，Phase 3 SFT 激活下游任务性能，Phase 3 无跨域 trade-off 需要验证（参考旧项目结论）。
- Product Key Memory 采用**两维独立路由**（√N × √N 网格），时间复杂度 O(√N) 替代暴力 O(N) 扫描，支持百万级知识条目。
- 配置管理：dataclass（无默认值）+ YAML（全量非敏感配置）+ `.env`（敏感信息），优先级 CLI args > `.env` > YAML。
- 知识库支持**动态增删**：近似 cluster 分配（不 recluster）支持热更新，`change_counter / N > 0.1` 时触发全量 recluster。

---

## 目录

- [技术决策](#技术决策)
- [模块设计](#1-模块设计)
  - [config.py](#11-configpy--配置管理)
  - [router/memory_bank.py](#12-routermemory_bankpy--双存储架构)
  - [models/qwen_wrapper.py](#13-modelsqwen_wrapperpy--知识编码器)
  - [router/memory_gate.py](#14-routermemory_gatepy--product-key-memory-粗排)
  - [router/clustering.py](#15-routerclusteringpy--独立子空间聚类)
  - [router/feature_adapter.py + router/refined_selector.py](#16-routerfeature_adapterpy--routerrefined_selectorpy--精排系统)
  - [router/model.py](#17-routermodelpy--memoryrouter-整合)
  - [models/injection_modules.py](#18-modelsinjection_modulespy--注入模块)
  - [models/modified_model.py](#19-modelsmodified_modelpy--modifiedqwen)
  - [pipeline.py](#110-pipelinepy--端到端管线)
- [训练管线](#2-训练管线)
- [实验计划](#3-实验计划)
- [文件结构与依赖](#4-文件结构与依赖)

---

## 1. 模块设计

### 1.1 config.py — 配置管理

**状态**: ✅ 已完成 · L0 基础 · 前置: 无 · 解锁: 全部 · 测试: `test_config.py` (8 tests)

**文件**: `config.py`（dataclass 定义 + 加载逻辑）+ `config/default.yaml`（全量配置值）+ `.env.example`（敏感信息模板）
**职责**: 所有超参数的 dataclass 类型定义（无默认值，YAML 漏写即报错）+ 三层优先级加载。

#### 设计原则

- dataclass **无默认值**：纯类型定义 + 结构化访问，强制 YAML 写全所有字段，缺字段在 `load_config()` 时立即报错
- 三层优先级：`CLI args > .env > YAML`，高优先级覆盖低优先级
- 敏感信息（模型绝对路径、API keys）只写 `.env`，不进 YAML 和代码

#### Dataclass 定义

```python
@dataclass
class ModelConfig:
    base_model: str              # "Qwen/Qwen3-0.6B"
    hidden_dim: int              # 1024
    num_layers: int              # 28
    injection_method: str        # "attention" | "concat" | "gated"
    injection_layers: List[int]  # [6, 12, 18, 24]
    encoder_depth: int           # 6（知识编码器使用前 N 层）
    fusion_length: int           # 64（K_f，Fusion Bank token 数）
    anchor_length: int           # 128（K_a，Anchor Bank token 数）

@dataclass
class RouterConfig:
    knowledge_num: int             # 1024 * 1024（1M 条目）
    dim: int                       # 1024（backbone hidden_dim）
    query_dim: int                 # 1024（query_proj 输出维度）
    key_proj_dim: int              # 512（行/列键投影维度，= dim // 2）
    adapter_dim: int               # 512（FeatureAdapter 输出维度）
    num_candidates: int            # 32（粗排固定输出候选数）
    temperature: float             # 0.1（PKM 路由温度）
    recluster_threshold: float     # 0.1（触发 recluster 的变更比例）
    max_candidates_per_cell: int   # -1=全量倒排索引；>0=每格上限（=1时退化为1:1映射）

@dataclass
class TrainConfig:
    # Phase 1: Router 训练（PKM + FeatureAdapter + RefinedSelector）
    phase1_lr: float             # 1e-3
    phase1_batch_size: int       # 64（每卡）
    phase1_max_epochs: int       # 3
    phase1_warmup_steps: int     # 200
    # Phase 2: Fusion 预训练（FineWeb-Edu，Injection + Encoder）
    phase2_lr: float             # 3e-4
    phase2_batch_size: int       # 32（每卡）
    phase2_max_epochs: int       # 5
    phase2_warmup_steps: int     # 100
    # Phase 3: 下游 SFT（MedQA，早停）
    phase3_lr: float             # 1e-4
    phase3_batch_size: int       # 16（每卡）
    phase3_max_epochs: int       # 10
    phase3_warmup_steps: int     # 50
    # 公共参数
    patience: int                # 3（Phase 3 early stopping）
    grad_clip: float             # 1.0
    bf16: bool                   # True

@dataclass
class DataConfig:
    fusion_length: int           # 64（与 model.fusion_length 保持一致）
    anchor_length: int           # 128（与 model.anchor_length 保持一致）
    num_workers: int             # 4（DataLoader 并行 worker 数）
    train_max_samples: int       # -1（-1 = 使用全量数据）

@dataclass
class PathsConfig:
    model_dir: str               # 基础模型目录（可被 .env MODEL_PATH 覆盖）
    llmlingua_model_dir: str     # LLMLingua 模型目录（可被 .env LLMLINGUA_PATH 覆盖）
    data_dir: str                # 数据根目录
    checkpoint_dir: str          # 检查点目录
    log_dir: str                 # 日志目录
    results_dir: str             # 评测结果目录

@dataclass
class EvalConfig:
    medqa_knowledge_map: str     # MedQA 知识映射文件路径（.jsonl）
    arc_knowledge_map: str       # ARC 知识映射文件路径
    mmlu_knowledge_map: str      # MMLU 知识映射文件路径
    lm_eval_tasks: List[str]     # lm-eval 任务名列表
    num_fewshot: int             # few-shot 数量（0 = zero-shot）

@dataclass
class Config:
    model: ModelConfig
    router: RouterConfig
    train: TrainConfig
    data: DataConfig
    paths: PathsConfig
    eval: EvalConfig
```

#### 加载函数

```python
def load_config(yaml_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    三层合并加载（对外唯一入口）:
      1. _load_yaml()          → 读取 YAML → base dict
      2. _override_from_env()  → 读取 .env → 覆盖 paths.model_dir / paths.llmlingua_model_dir
      3. _override_from_cli()  → cli_overrides（点路径格式）→ 最终覆盖
      4. _dict_to_config()     → dict → Config（缺字段报 TypeError）
    """

# CLI 覆盖使用点路径格式，例如：
# cli_overrides = {"model.injection_layers": [4, 8, 12], "train.phase2_lr": 1e-4}
```

#### .env 支持的覆盖字段

| 环境变量 | 覆盖目标 |
|---------|---------|
| `MODEL_PATH` | `paths.model_dir` |
| `LLMLINGUA_PATH` | `paths.llmlingua_model_dir` |

#### 文件分工

| 文件 | 内容 | 提交到 Git |
|------|------|-----------|
| `config/default.yaml` | 全量非敏感配置（必须写全所有字段） | 是 |
| `.env` | 敏感信息（模型绝对路径、API Token） | 否 |
| `.env.example` | `.env` 模板（值留空，供参考） | 是 |

**依赖**: `dataclasses`（标准库）, `pyyaml`, `python-dotenv`

---

### 1.2 router/memory_bank.py — 双存储架构

**状态**: ✅ 已完成 · L1 存储 · 前置: §1.1 · 解锁: §1.4,§1.5,§1.7,§1.10 · 测试: `test_memory_bank.py` (19 tests)

**文件**: `router/memory_bank.py`
**职责**: 双路知识存储（Fusion Bank + Anchor Bank），支持动态增删、近似 cluster 分配和物理压缩。

#### 核心数据结构

```python
# Fusion Bank: 存 LLMLingua 压缩后的高密度知识（注入用）
fusion_bank:  Tensor[N, K_f]    # K_f=64, 压缩 facts token IDs
# Anchor Bank: 存原文截断（路由索引用）
anchor_bank:  Tensor[N, K_a]    # K_a=128, 原文截断 token IDs
# 共享索引
valid_mask:         Tensor[N]       # 有效条目标记（bool）
inverted_index:     Tensor[N]       # 按 cluster 排序的数据 ID，初始全 -1
cluster_offsets:    Tensor[C+1]     # 每个 cluster 在 inverted_index 中的起始偏移
cluster_counts:     Tensor[C]       # 每个 cluster 的有效条目数
# 近似分配状态（全量 recluster 后保存）
pca_matrix:         Optional[Tensor[D, D]]      # PCA 旋转矩阵，初始 None
pca_mean:           Optional[Tensor[D]]         # 均值向量，初始 None
row_centroids:      Optional[Tensor[num_keys, D//2]]
col_centroids:      Optional[Tensor[num_keys, D//2]]
next_free:          int             # 下一个可写入槽位
change_counter:     int             # 自上次 recluster 以来的变更计数
```

C = num_keys²，num_keys = √N，knowledge_num 必须为完全平方数。

#### 类设计

```python
class FusionBank:
    """存 LLMLingua 压缩 facts token IDs [N, K_f]，供知识编码器读取注入。"""

    def __init__(self, knowledge_num: int, fusion_length: int, device: str) -> None: ...

    def update_all(self, token_ids: Tensor) -> None:
        """全量替换（Phase 0/训练期）: token_ids [N, K_f]，断言 shape 和 dtype"""

    def __getitem__(self, ids: Tensor) -> Tensor:
        """批量读取: ids [B] → [B, K_f]，断言索引不越界"""


class AnchorBank:
    """存原文截断 token IDs [N, K_a]，供聚类计算 embedding。"""

    def __init__(self, knowledge_num: int, anchor_length: int, device: str) -> None: ...

    def update_all(self, token_ids: Tensor) -> None:
        """全量替换，断言 shape 和 dtype"""

    def get_embeddings(
        self, encoder: "KnowledgeEncoder", valid_mask: Tensor, chunk_size: int
    ) -> Tensor:
        """
        仅对 valid_mask=True 的条目流式编码（chunk_size=64 避免 OOM）:
          → [N_valid, D]（encoder.encode_mean mean-pool 后）
        valid_mask 全 False 时返回空张量
        """


class DualKnowledgeStore:
    """统一管理 FusionBank + AnchorBank + 倒排索引，提供动态更新接口。"""

    def __init__(
        self, config: RouterConfig, fusion_length: int, anchor_length: int, device: str
    ) -> None:
        """
        fusion_length / anchor_length 显式传入（来自 ModelConfig），不从 RouterConfig 读取。
        内部含 threading.Lock 保护所有写操作。
        """

    def add_entries(
        self, fusion_token_ids: Tensor, anchor_token_ids: Tensor
    ) -> None:
        """
        热更新添加条目（调用方负责压缩/截断，只接受已预处理的 token IDs）:
          1. pca_matrix 为 None → RuntimeError（必须先调 compact_and_recluster）
          2. next_free + B > knowledge_num → RuntimeError（不自动扩容）
          3. 写入双 Bank；valid_mask[start:end] = True
          4. 近似分配（_approximate_assign）→ _append_to_inverted_index
          5. next_free += B; change_counter += B
        """

    def delete_entries(self, ids: List[int]) -> None:
        """逻辑删除: valid_mask[ids] = False; change_counter += len(ids)"""

    def should_recluster(self) -> bool:
        """change_counter / N_valid > recluster_threshold（N_valid=0 时返回 False）"""

    def compact_and_recluster(self, encoder: "KnowledgeEncoder") -> None:
        """
        物理压缩 + 全量重聚类（受 _lock 保护）:
          1. 收集 valid 条目 → 紧凑排列到 bank[0..N_valid-1]
          2. anchor_bank.get_embeddings(encoder, valid_mask, chunk_size=64) → [N_valid, D]
          3. SubspaceClustering.fit(embeddings.numpy(), num_keys) → ClusteringResult
          4. 更新 pca_matrix, pca_mean, row_centroids, col_centroids
          5. _rebuild_inverted_index → 重建 inverted_index, cluster_offsets, cluster_counts
          6. next_free = N_valid; change_counter = 0
        """

    def save_state(self, path: str) -> None:
        """torch.save 序列化所有 buffer + next_free + change_counter"""

    def load_state(self, path: str) -> None:
        """torch.load 恢复所有状态，自动处理 pca_matrix 等 Optional 字段"""
```

**私有方法**：
- `_approximate_assign(anchor_token_ids)` — 热更新近似分配：用 token ID 浮点均值作为极简 embedding，经 PCA 旋转 + 最近邻 centroid 得 grid_idx（精度较低，下次 recluster 时纠正）
- `_rebuild_inverted_index(data_indices, grid_indices)` — 全量重建：按 grid_indices 排序后构造 inverted_index、cluster_offsets、cluster_counts
- `_append_to_inverted_index(data_indices, grid_indices)` — 热更新追加：逐条插入到对应 cluster 末尾（不重排已有条目）

**为什么双 Bank？**

| 维度 | Fusion Bank | Anchor Bank |
|------|-------------|-------------|
| 内容 | LLMLingua 压缩后的高密度知识 | 原始文本直接截断 |
| 用途 | 知识编码 → Cross-Attention 注入 | 计算 embedding → 聚类 → 更新 Keys |
| 语义空间 | 压缩语义（信息密度高） | 原文语义（与 query 同一空间） |

**依赖**: torch（无其他运行时依赖；`KnowledgeEncoder`/`SubspaceClustering` 通过 TYPE_CHECKING 前向引用，避免循环导入）

---

### 1.3 models/qwen_wrapper.py — 知识编码器

**状态**: ✅ 已完成 · L2 编码 · 前置: §1.1 · 解锁: §1.7,§1.9,§1.10 · 测试: `test_qwen_wrapper.py` (12 tests)

**文件**: `models/qwen_wrapper.py`
**职责**: 封装 Qwen3 前 N 层作为知识编码器，将知识 token IDs 上下文化为稠密向量供注入。

#### `load_base_model(model_path: str, bf16: bool) -> AutoModelForCausalLM`

加载 Qwen3 基础模型并冻结所有参数。`model_path` 支持本地路径（如 `"Qwen3-0.6B"`）或 HuggingFace Hub 名称。

#### `class KnowledgeEncoder(nn.Module)`

```python
def __init__(self, base_model: AutoModelForCausalLM, encoder_depth: int, hidden_dim: int):
    # 共享权重（均冻结）
    self.embed_tokens = base_model.model.embed_tokens
    self.layers = nn.ModuleList(list(base_model.model.layers[:encoder_depth]))
    self.rotary_emb = base_model.model.rotary_emb   # ⚠️ 必须保留：transformers 4.51.0
                                                     # Qwen3Attention.forward 要求外部传入
                                                     # position_embeddings，不接受 None
    # 独立权重（可训练）
    self.norm = copy.deepcopy(base_model.model.norm)
    self._freeze_all()  # 冻结所有共享组件，norm 保持可训练
```

```python
def forward(self, knowledge_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Args:
        knowledge_ids:   [B, K] LongTensor — 来自 FusionBank / AnchorBank 的 token IDs
        attention_mask:  [B, K] LongTensor — 1=有效，0=padding
    Returns:
        [B, K, D] FloatTensor — 上下文化的知识表示
    实现（双向注意力，无 causal mask）:
        Phase 1: h = embed_tokens(knowledge_ids)              # [B, K, D]
        Phase 2: position_ids = arange(K).expand(B)          # [B, K]
        Phase 3: cos, sin = self.rotary_emb(h, position_ids) # 外部计算后传入各层
        Phase 4: attn_bias = _build_attention_mask(...)       # [B, 1, 1, K]，pad→-inf
        Phase 5: for layer in self.layers:
                     h = layer(h, attn_bias, position_ids,
                               position_embeddings=(cos, sin))  # 直接返回 Tensor
        Phase 6: h = self.norm(h)
    """
```

> **Qwen3 4.51.0 接口注意事项**:
> - `Qwen3DecoderLayer.forward` 直接返回 `torch.Tensor`（非元组），不要用 `[0]` 取
> - `Qwen3Attention.forward` 的 `position_embeddings` 参数直接解包，不能为 `None`
> - `Qwen3Attention` 内部的 rotary 相关属性名为 `rotary_fn`（apply 函数），`rotary_emb` 在 `base_model.model` 上

```python
def encode_mean(self, knowledge_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """masked mean pooling → [B, D]，供 AnchorBank.get_embeddings 调用"""
    h = self.forward(knowledge_ids, attention_mask)     # [B, K, D]
    mask_float = attention_mask.float().unsqueeze(-1)   # [B, K, 1]
    return (h * mask_float).sum(1) / mask_float.sum(1).clamp(min=1.0)
```

#### 其他接口

| 方法 | 说明 |
|------|------|
| `unfreeze_layers()` | Phase 2 开始时调用，解冻前 encoder_depth 层 |
| `device` (property) | 返回当前设备，供 `AnchorBank.get_embeddings` 的 `.to(encoder.device)` |
| `_freeze_all()` | 内部：冻结 embed_tokens / layers / rotary_emb |
| `_build_attention_mask(mask, dtype)` | 内部：padding mask → additive bias `[B, 1, 1, K]` |

**关键设计**:
- encoder_depth=6（Qwen3-0.6B 共 28 层，用前 6 层）
- 这 6 层在 Phase 2 开始时**解冻**，与 Injection 模块联合训练
- `self.norm` 深拷贝（独立权重），不与 base_model 共享，全程可训练
- 双向注意力（无 causal mask），适合知识文本编码

**依赖**: transformers（AutoModelForCausalLM）, torch, copy（标准库）

---

### 1.4 router/memory_gate.py — Product Key Memory（粗排）

**状态**: ✅ 已完成 · L2 粗排 · 前置: §1.1,§1.2 · 解锁: §1.7 · 测试: `test_memory_gate.py` (19 tests)

**文件**: `router/memory_gate.py`
**职责**: 两维独立路由，将 N 条知识映射到 √N × √N 网格，O(√N) 时间检索 num_candidates 个候选。

```python
class ProductKeyMemory(nn.Module):
    """两维独立 Product Key Memory，粗排检索固定 num_candidates 个候选条目。"""

    def __init__(self, config: RouterConfig):
        # 可训练投影层（无 bias）
        self.query_proj   = Linear(config.dim, config.query_dim, bias=False)
        self.row_key_proj = Linear(config.key_proj_dim, config.key_proj_dim, bias=False)
        self.col_key_proj = Linear(config.key_proj_dim, config.key_proj_dim, bias=False)
        # 非训练 Keys：register_buffer [√N, key_proj_dim]，初始全零
        # 通过 update_keys() 更新，不从 store 动态读取
        register_buffer("row_keys", zeros(num_keys, config.key_proj_dim))
        register_buffer("col_keys", zeros(num_keys, config.key_proj_dim))
        self.num_keys    = int(config.knowledge_num ** 0.5)  # √N
        self.K_COARSE    = 4     # 每维 top-k，共 4×4=16 个候选 cluster
        self.temperature = config.temperature
        self.num_candidates          = config.num_candidates
        self.max_candidates_per_cell = config.max_candidates_per_cell

    def update_keys(self, row_keys: Tensor, col_keys: Tensor) -> None:
        """每 epoch 在 compact_and_recluster 后调用，更新 register_buffer 中的 keys。
        典型调用：pkm.update_keys(store.row_centroids, store.col_centroids)
        入参形状：[√N, key_proj_dim]，dtype=torch.float
        """

    def forward(
        self,
        embedding: Tensor,        # [B, D]
        store: DualKnowledgeStore,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        粗排检索流程:
          # Phase 1: 查询侧投影 + 分割 + L2 归一化
          q = query_proj(embedding)              → [B, query_dim=1024]
          q1, q2 = q.chunk(2, dim=-1)           → [B, 512] × 2
          q1, q2 = L2_normalize(q1), L2_normalize(q2)

          # Phase 2: Key 侧投影 + L2 归一化（register_buffer，无梯度）
          k1 = L2_normalize(row_key_proj(row_keys))  → [√N, 512]
          k2 = L2_normalize(col_key_proj(col_keys))  → [√N, 512]

          # Phase 3: 温度缩放分数
          scores_1 = q1 @ k1.T / τ              → [B, √N]
          scores_2 = q2 @ k2.T / τ              → [B, √N]

          # Phase 4: Top-K + 笛卡儿积
          top_rows = topk(scores_1, K_COARSE=4).indices  → [B, 4]
          top_cols = topk(scores_2, K_COARSE=4).indices  → [B, 4]
          grid_indices = top_rows[:,:,None]*√N + top_cols[:,None,:]
                        .reshape(B, 16)                  → [B, 16]

          # Phase 5: 倒排索引查询（支持 max_candidates_per_cell 限制）
          candidates = _lookup_candidates(grid_indices, store)  → [B, num_candidates]

        Returns:
            candidates: Tensor[B, num_candidates]  — 候选知识条目 ID（固定大小，不足则重复填充）
            scores_1:   Tensor[B, √N]              — 行匹配分数（训练损失用）
            scores_2:   Tensor[B, √N]              — 列匹配分数（训练损失用）
            q_adapted:  Tensor[B, key_proj_dim]    — q1（L2 归一化，传给 RefinedSelector）
        """

    def _lookup_candidates(
        self, grid_indices: Tensor, store: DualKnowledgeStore
    ) -> Tensor:
        """
        倒排索引查询：遍历 16 个 grid cell，从 store.inverted_index 汇聚候选 ID。
        max_candidates_per_cell:
            -1  → 每格取全部条目（热更新多条/格场景）
            >0  → 每格最多取该数目（=1 时退化为 1:1 简单映射）
        不足 num_candidates 时循环重复填充；空 store 时全零填充。
        """
```

**关键设计**:
- **register_buffer Keys**：`row_keys / col_keys` 存于模块内，随 save/load 持久化；通过 `update_keys()` 更新而非每次 forward 从 store 读取（对齐参考项目设计）
- **双侧 L2 归一化**：查询侧（q1/q2）和 Key 侧（k1/k2）均做 L2 normalize，保证余弦相似度语义
- **`q_adapted = q1`**：直接返回归一化行子查询 `[B, key_proj_dim]`；FeatureAdapter 独立于 PKM，不在此处调用
- **倒排索引 + 可配置 cap**：`max_candidates_per_cell`（default=-1）控制每格取条数，=1 时退化为参考项目的 1:1 映射风格

**依赖**: torch, router/memory_bank.py（DualKnowledgeStore）, config.py（RouterConfig）

---

### 1.5 router/clustering.py — 独立子空间聚类

**状态**: ⬜ 待实现 · L3 聚类 · 前置: §1.2 · 解锁: §1.2(recluster),§1.7 · 测试: `test_clustering.py`

**文件**: `router/clustering.py`
**职责**: 每 epoch 将 Anchor Bank 的所有 embeddings 聚类，更新 row_keys / col_keys 和倒排索引。

```python
class SubspaceClustering:
    """
    独立子空间聚类（Axis-Aligned Recursive Bisection）。
    保证两个子空间正交，使 Product Key 的行/列键相互正交。
    """

    @staticmethod
    def fit(
        embeddings: np.ndarray,  # [N, D]
        num_keys: int,           # √N
    ) -> ClusteringResult:
        """
        聚类流程:
          Step 1: PCA 旋转去相关
            X_centered = X - mean(X)
            U, S, Vt = SVD(cov(X_centered))
            X_rotated = X_centered @ U          → [N, D]

          Step 2: 交错分割（保证正交）
            subspace_1 = X_rotated[:, 0::2]    → [N, D//2] (Row)
            subspace_2 = X_rotated[:, 1::2]    → [N, D//2] (Col)

          Step 3: Axis-Aligned Recursive Bisection（num_keys 须为 2^k）
            - 每步选一个维度，按中位数二分
            - 保证每 cluster 条目数均匀

          Step 4: 构造正交 Keys
            row_keys = [center_1, 0]          → [num_keys, D//2]
            col_keys = [0, center_2]          → [num_keys, D//2]

          Step 5: 构建倒排索引
            grid_idx = row_label × num_keys + col_label
            → inverted_index 按 grid_idx 排序，记录 offsets + counts

        Returns:
            ClusteringResult(
                row_keys,        # [√N, D//2]
                col_keys,        # [√N, D//2]
                pca_matrix,      # [D, D]（U 旋转矩阵）
                pca_mean,        # [D]
                row_centroids,   # [√N, D//2]
                col_centroids,   # [√N, D//2]
                inverted_index,  # [N]（按 grid_idx 排序的数据 ID）
                cluster_offsets, # [C+1]
                cluster_counts,  # [C]
            )
        """

    @staticmethod
    def assign_approximate(
        embedding: np.ndarray,   # [D]
        pca_matrix, pca_mean,
        row_centroids, col_centroids,
        num_keys: int,
    ) -> int:
        """
        近似 cluster 分配（新增条目热更新，不触发全量 recluster）:
          x_rotated = (embedding - pca_mean) @ pca_matrix
          sub1 = x_rotated[0::2]
          sub2 = x_rotated[1::2]
          row_label = argmin(||sub1 - row_centroids||)
          col_label = argmin(||sub2 - col_centroids||)
          → grid_idx = row_label × num_keys + col_label
        """
```

**关键设计**:
- `num_keys` 必须为 2 的幂次（保证平衡二分）
- 近似分配**不修改 Keys**，只更新倒排索引，cluster 渐进失衡由周期性 recluster 恢复
- 默认阈值：`change_counter / N_valid > 0.1` 触发 recluster

**依赖**: numpy, scipy（SVD）

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_clustering_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/clustering/`
- [ ] cluster 均衡 max/mean < 3；PCA 子空间正交

---

### 1.6 router/feature_adapter.py + router/refined_selector.py — 精排系统

**状态**: ⬜ 待实现 · L3 精排 · 前置: 无硬依赖 · 解锁: §1.7 · 测试: `test_feature_adapter.py`, `test_refined_selector.py`

**文件**: `router/feature_adapter.py`, `router/refined_selector.py`
**职责**: 防止特征坍缩的查询投影（FeatureAdapter）+ 从 ~64 候选中精选 Top-1 的交叉编码器（RefinedSelector）。

#### FeatureAdapter

```python
class FeatureAdapter(nn.Module):
    """
    将冻结 Qwen3 embedding 投影到适配空间，防止 Feature Collapse。
    输入: 冻结 embedding 的 mean pool [B, D]
    输出: 适配向量 [B, adapter_dim=512]
    """

    def __init__(self, in_dim: int, adapter_dim: int):
        self.centering = ...               # 可学习均值（去模式偏差）
        self.layer_norm = LayerNorm(in_dim)
        self.proj = Linear(in_dim, adapter_dim)
        self.activation = Tanh()
        self.output_norm = LayerNorm(adapter_dim)  # 防止 collapse

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, S, D] 或 [B, D]（已 pool）
        1. Mean pooling with mask: [B, S, D] → [B, D]
        2. Centering → LayerNorm → Linear → Tanh
        3. Output LayerNorm
        → [B, adapter_dim]
        """
```

#### RefinedSelector

```python
class RefinedSelector(nn.Module):
    """
    2 层 Transformer 交叉编码器，从 ~64 候选中精选 Top-1 知识条目。
    训练时：CE loss + 与 teacher 的 KL 对比（见 §2.2）。
    """

    def __init__(self, adapter_dim: int, num_heads: int = 8, num_layers: int = 2):
        self.transformer = TransformerEncoder(
            d_model=adapter_dim, nhead=num_heads, num_layers=num_layers
        )
        self.score_head = Linear(adapter_dim, 1)

    def forward(
        self,
        query_vec: Tensor,     # [B, adapter_dim]  ← FeatureAdapter 输出
        cand_vecs: Tensor,     # [B, C, adapter_dim] ← 候选知识编码
    ) -> Tuple[Tensor, Tensor]:
        """
        拼接: [query; cand_1; ...; cand_C]  → [B, 1+C, adapter_dim]
        TransformerEncoder(2 层, 8 heads)   → [B, 1+C, adapter_dim]
        提取候选部分 [:, 1:, :]             → [B, C, adapter_dim]
        Linear(adapter_dim, 1).squeeze(-1)  → scores [B, C]

        Returns:
            scores:   Tensor[B, C]   — 精排分数（softmax 后用于 loss）
            best_idx: Tensor[B]      — argmax 候选 ID（推理用）
        """
```

**依赖**: torch, router/memory_bank.py

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_refined_selector_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/refined_selector/`
- [ ] FeatureAdapter 输出稳定（无 NaN/Inf）
- [ ] RefinedSelector best_idx ∈ [0, C)

---

### 1.7 router/model.py — MemoryRouter 整合

**状态**: ⬜ 待实现 · L4 整合 · 前置: §1.4,§1.5,§1.6 · 解锁: §1.10 · 测试: `test_router_model.py`

**文件**: `router/model.py`
**职责**: 组合 ProductKeyMemory + FeatureAdapter + RefinedSelector，提供统一的检索接口。

```python
class MemoryRouter(nn.Module):
    """
    端到端路由器: query text → 最优知识条目 ID。
    可训练参数 (~15M): query_proj + key_proj + FeatureAdapter + RefinedSelector。
    Keys (row_keys/col_keys) 不参与梯度，每 epoch 通过 recluster 更新。
    """

    def __init__(self, config: RouterConfig, encoder: KnowledgeEncoder):
        self.pkm = ProductKeyMemory(config)
        self.adapter = FeatureAdapter(config.dim, config.adapter_dim)
        self.selector = RefinedSelector(config.adapter_dim)
        self.encoder = encoder  # 共享 KnowledgeEncoder（用于候选编码）

    def forward(
        self,
        query_embedding: Tensor,   # [B, D] — 输入问题的 embedding（Qwen3 编码）
        store: DualKnowledgeStore,
    ) -> RouterOutput:
        """
        推理流程:
          1. adapter(query_embedding) → q_adapted [B, 512]
          2. pkm(query_embedding, store) → candidates [B, ~64], scores_1, scores_2
          3. 编码候选: encoder(anchor_bank[candidates]) → cand_vecs [B, 64, 512]
          4. selector(q_adapted, cand_vecs) → scores [B, 64], best_local_idx [B]
          5. best_id = candidates[best_local_idx] → 全局知识 ID

        Returns:
            RouterOutput(
                best_id:       Tensor[B],      # 最优知识条目全局 ID
                candidates:    Tensor[B, ~64], # 粗排候选 ID
                coarse_scores: Tuple[Tensor, Tensor],  # scores_1, scores_2（Router loss 用）
                fine_scores:   Tensor[B, ~64], # 精排分数（Router loss 用）
            )
        """

    @torch.no_grad()
    def retrieve(
        self, query_embedding: Tensor, store: DualKnowledgeStore
    ) -> Tensor:
        """推理专用（无梯度），返回 knowledge_ids [B, K_f]（来自 Fusion Bank）"""
        out = self.forward(query_embedding, store)
        return store.fusion_bank[out.best_id]   # [B, K_f]
```

**依赖**: router/memory_gate.py, router/feature_adapter.py, router/refined_selector.py, router/memory_bank.py

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_router_model_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/router_model/`
- [ ] best_id ∈ [0, N)；可训练参数量 ~15M

---

### 1.8 models/injection_modules.py — 注入模块

**状态**: ⬜ 待实现 · L4 注入（可与 L3 并行）· 前置: 无 · 解锁: §1.9 · 测试: `test_injection_modules.py`

**文件**: `models/injection_modules.py`
**职责**: 定义三种知识注入方式，共享统一接口 `forward(hidden, knowledge, mask) → hidden`。

#### 统一接口

```python
class BaseInjection(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        hidden: Tensor,      # [B, L, D] — 当前层 hidden states
        knowledge: Tensor,   # [B, K_f, D] — 知识编码器输出
        mask: Tensor,        # [B, K_f] — 知识 padding mask（0 为 pad）
    ) -> Tensor:
        """返回注入后的 hidden [B, L, D]"""
```

#### AttentionInjection（主力方案）

```python
class AttentionInjection(BaseInjection):
    """Cross-Attention + Null KV + 零初始化，约 4.2M 参数/层。"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        self.pre_norm = RMSNorm(hidden_dim)
        self.W_q = Linear(hidden_dim, hidden_dim)
        self.W_k = Linear(hidden_dim, hidden_dim)
        self.W_v = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        # 可学习 Null KV（知识全 pad 时退化为 attend to null）
        self.null_k = Parameter(torch.zeros(1, 1, hidden_dim))
        self.null_v = Parameter(torch.zeros(1, 1, hidden_dim))

        # 零初始化：训练初期 attn_out ≈ 0 → 等价于原始模型
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: Tensor, knowledge: Tensor, mask: Tensor) -> Tensor:
        """
        normed = pre_norm(hidden)                       → [B, L, D]
        Q = W_q(normed)                                 → [B, L, D]

        K = cat([null_k, W_k(knowledge)], dim=1)       → [B, K_f+1, D]
        V = cat([null_v, W_v(knowledge)], dim=1)       → [B, K_f+1, D]

        attn_out = MultiHeadAttention(Q, K, V, mask)   → [B, L, D]
        output = hidden + out_proj(attn_out)            # 干净残差

        初始化效果: out_proj=0 → attn_out→0 → output=hidden（无注入退化）
        """
```

#### ConcatProjection（备选，参数量更大）

```python
class ConcatProjection(BaseInjection):
    """mean_pool → concat → MLP(2D→4D→D) + 残差，约 12.6M 参数/层。"""

    def __init__(self, hidden_dim: int):
        self.proj_in = Linear(hidden_dim * 2, hidden_dim * 4)
        self.proj_out = Linear(hidden_dim * 4, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        # 末层零初始化
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, hidden: Tensor, knowledge: Tensor, mask: Tensor) -> Tensor:
        """
        k_pooled = masked_mean(knowledge, mask)       → [B, D]
        k_pooled = k_pooled.unsqueeze(1).expand(-1, L, -1)
        concat = cat([hidden, k_pooled], dim=-1)      → [B, L, 2D]
        delta = proj_out(gelu(proj_in(concat)))       → [B, L, D]
        output = hidden + norm(delta)
        """
```

#### GatedInjection（轻量备选）

```python
class GatedInjection(BaseInjection):
    """per-dim gate × knowledge + 残差，约 1K 参数/层。"""

    def __init__(self, hidden_dim: int):
        self.gate = Parameter(torch.zeros(hidden_dim))  # 零初始化

    def forward(self, hidden: Tensor, knowledge: Tensor, mask: Tensor) -> Tensor:
        """
        k_pooled = masked_mean(knowledge, mask)  → [B, D]
        gate_val = sigmoid(self.gate)             → [D]（近 0.5 初始化）
        output = hidden + gate_val * k_pooled.unsqueeze(1)
        零初始化 gate=0 → gate_val=0.5 → 弱注入，训练中动态调整
        """
```

**三种注入方式对比**:

| 注入方式 | 参数量/层 | 初始化 | 计算复杂度 | 状态 |
|---------|----------|-------|----------|------|
| **AttentionInjection** | ~4.2M | out_proj=0 | O(L·K·d) | 主力 |
| ConcatProjection | ~12.6M | 末层=0 | O(L·d) | 备选（E5-D 消融） |
| GatedInjection | ~1K | gate=0 | O(L·d) | 轻量备选 |

**依赖**: torch

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_injection_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/injection/`
- [ ] 零初始化残差不变性：初始 forward 输出 ≈ hidden

---

### 1.9 models/modified_model.py — ModifiedQwen

**状态**: ⬜ 待实现 · L5 模型 · 前置: §1.3,§1.8 · 解锁: §1.10 · 测试: `test_modified_model.py`

**文件**: `models/modified_model.py`
**职责**: 通过 Hook 机制在 Qwen3 指定层注入知识，实现无侵入式融合。

```python
class ModifiedQwen(nn.Module):
    """
    Qwen3 + Hook 注入，在 injection_layers 位置调用 AttentionInjection。
    基础模型完全冻结，仅 injection_modules 参与训练。
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        knowledge_encoder: KnowledgeEncoder,
        injection_modules: nn.ModuleList,   # 4 个 AttentionInjection（每层共享或独立）
        injection_layers: List[int],        # [6, 12, 18, 24]
    ):
        # 冻结基础模型全量参数
        for p in base_model.parameters():
            p.requires_grad = False
        # 注册 Hook
        for i, layer_idx in enumerate(injection_layers):
            base_model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(i)
            )

    def _make_hook(self, module_idx: int) -> Callable:
        """
        返回 post-hook 函数:
          hook(module, input, output):
            hidden = output[0]         # [B, L, D]
            injected = injection_modules[module_idx](
                hidden, self._current_knowledge, self._current_mask
            )
            return (injected,) + output[1:]
        """

    def forward(
        self,
        input_ids: Tensor,       # [B, L]
        knowledge_ids: Tensor,   # [B, K_f]
        attention_mask: Tensor,  # [B, L]
        labels: Optional[Tensor] = None,  # [B, L]，-100 为忽略位
    ) -> CausalLMOutput:
        """
        Step 1: 知识编码（一次，所有注入层复用）
            knowledge_mask = (knowledge_ids != pad_token_id)  → [B, K_f]
            self._current_knowledge = knowledge_encoder(
                knowledge_ids, knowledge_mask
            )                                                  → [B, K_f, D]
            self._current_mask = knowledge_mask

        Step 2: 基础模型前向（Hook 自动在指定层注入）
            output = base_model(input_ids, attention_mask, labels)

        Step 3: 返回（logits + loss if labels）
        """
```

**前向数据流**:

```
input_ids [B, L]  +  knowledge_ids [B, K_f]
     │                      │
     │           knowledge_encoder  → [B, K_f, D]（知识编码，一次）
     │                      │
     │           Qwen3 Layer 0-5（无注入）
     │                      │
     └──────────────── Hook ─┤ Layer 6 → AttentionInjection(hidden, knowledge)
                             │
                      Layer 7-11
                             │
                        Hook ─┤ Layer 12 → AttentionInjection
                             ...（Layer 18, 24 同理）
                             │
                      lm_head → logits [B, L, V]
```

**可训练 vs 冻结**:

```
冻结:
  ✗ Qwen3-0.6B 全量参数 (600M)
  ✗ embed_tokens（共享词嵌入）
  ✗ LLMLingua-2 (XLM-RoBERTa)

可训练（Fusion, ~20M）:
  ✓ AttentionInjection × 4 层 (~4.2M/层)
  ✓ KnowledgeEncoder（Qwen3 前 6 层，解冻）
```

**依赖**: transformers, models/injection_modules.py, models/qwen_wrapper.py

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_modified_model_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/modified_model/`
- [ ] Hook 正确注册到 4 层；logits 形状 [B,L,V]

---

### 1.10 pipeline.py — 端到端管线

**状态**: ⬜ 待实现 · L6 管线 · 前置: §1.7,§1.9 · 解锁: 训练管线 · 测试: `test_pipeline.py`

**文件**: `pipeline.py`
**职责**: 串联 Router 检索 → 知识编码 → 生成的完整推理流程，提供统一对外接口。

```python
class ExplicitLMPipeline:
    """
    端到端推理管线（无梯度）:
    query text → Router 检索 → 注入知识 → Qwen3 生成 → answer
    """

    def __init__(
        self,
        config: Config,
        modified_qwen: ModifiedQwen,
        router: MemoryRouter,
        store: DualKnowledgeStore,
        tokenizer: AutoTokenizer,
    ): ...

    @classmethod
    def from_checkpoints(
        cls,
        config: Config,
        router_ckpt: str,    # Phase 1 最优权重
        fusion_ckpt: str,    # Phase 2/3 最优权重
        store_path: str,     # 知识库路径
    ) -> "ExplicitLMPipeline":
        """从 checkpoint 加载完整管线"""

    def answer(
        self, question: str, use_real_router: bool = True
    ) -> PipelineOutput:
        """
        Args:
            question:         用户问题（原始文本）
            use_real_router:  True=真实路由, False=Oracle 知识（实验用）

        流程:
          1. 路由检索（如 use_real_router）:
               q_emb = modified_qwen.embed(question)  → [1, D]
               knowledge_ids = router.retrieve(q_emb, store)  → [1, K_f]
             否则（Oracle）:
               knowledge_ids = lookup_oracle_knowledge(question)

          2. 生成:
               output = modified_qwen(input_ids, knowledge_ids, ...)
               answer = tokenizer.decode(output.logits.argmax(-1))

        Returns:
            PipelineOutput(
                answer:        str,
                retrieved_id:  int,   # 检索到的知识条目 ID
                latency_ms:    float,
            )
        """

    def evaluate_loglikelihood(
        self, question: str, choices: List[str], knowledge_ids: Tensor
    ) -> int:
        """
        多选题评测（loglikelihood 方法）:
          对 " A"/" B"/" C"/" D" 各做 forward，取 continuation log-prob 最高者
          → 预测选项 idx (0-3)
        """
```

**依赖**: 所有 models/router 模块，transformers

**验证 Checkpoint**:
- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%
- [ ] `tests/integration/test_pipeline_flow.py` 端到端验证通过
- [ ] Markdown 报告生成到 `tests/outputs/pipeline/`
- [ ] 端到端 query→answer 可运行

---

## 2. 训练管线

### 2.1 阶段总览

```
Phase 0: 知识构建（离线，每个模型独立）
├─ LLMLingua-2 压缩语料 → Fusion Bank [N, K_f=64]
├─ 原文截断 → Anchor Bank [N, K_a=128]
├─ embed(Anchor Bank) → SubspaceClustering → Keys + 倒排索引
└─ 验证: Cluster 负载均衡（max/mean < 3）→ 进入 Phase 1
      ↓
Phase 1: Router 训练（路由能力）
├─ 目标: 学习从输入精确定位知识条目
├─ 损失: CE(粗排) + KL(软标签) + CE(精排)
├─ 每 epoch 重新聚类 + 更新 Keys + 倒排索引
└─ 验证: 【E1】Recall@1 > 50% → 进入 Phase 2
      ↓
Phase 2: Fusion 预训练（融合能力）
├─ 数据: FineWeb-Edu（通用语料，Oracle 知识）
├─ 任务: 语言建模（给定压缩知识，预测原文 token）
├─ 可训练: AttentionInjection × 4 + KnowledgeEncoder（前 6 层）
├─ 冻结: Qwen3 全量 + Router
└─ 验证: 【E2】KS > 20% → 进入 Phase 3
      ↓
Phase 3: 下游 SFT（任务激活）
├─ 数据: MedQA train
├─ 起点: Phase 2 最优权重
├─ 早停: val_loss，patience=3
└─ 输出: phase3_best checkpoint（用于 E4-E7 评测）
```

### 2.2 Phase 1 Router 训练循环

```python
# 每个 Epoch 的两阶段流程
for epoch in range(router_config.max_epochs):

    # ── Phase A: 数据与索引更新（Main Process） ──
    texts = sample(corpus, N=router_config.knowledge_num)
    facts = compressor.compress_batch(texts)            # LLMLingua 压缩
    store.update_all(tokenize(facts), tokenize_truncate(texts))  # 更新双 Bank
    result = clustering.fit(
        encoder.encode_mean(store.anchor_bank).cpu().numpy(),
        num_keys=int(router_config.knowledge_num ** 0.5),
    )
    pkm.update_keys(result.row_keys, result.col_keys)   # 更新 Keys
    store.update_index(result)                          # 重建倒排索引
    broadcast_to_all_ranks(store)

    # ── Phase B: 路由训练（所有 ranks） ──
    for batch in router_dataloader:
        q_emb = get_query_embedding(batch["questions"])    # [B, D]
        target_ids = batch["knowledge_ids"]                # [B]（Ground Truth 条目 ID）

        # 获取 teacher soft labels（KL 监督）
        teacher_row, teacher_col = compute_teacher_labels(target_ids)  # [B, √N]

        out = router(q_emb, store)
        scores_1, scores_2 = out.coarse_scores              # [B, √N] × 2

        # 损失计算
        ce_loss = CE(scores_1, target_row) + CE(scores_2, target_col)
        kl_loss = KL(scores_1.softmax(-1), teacher_row) + \
                  KL(scores_2.softmax(-1), teacher_col)
        coarse_loss = (1 - alpha) * ce_loss + alpha * kl_loss   # alpha=0.2

        fine_loss = CE(out.fine_scores, target_local_idx)

        total_loss = coarse_loss + fine_loss
        total_loss.backward(); optimizer.step()
```

### 2.3 Phase 2/3 Fusion 训练循环

```python
for batch in fusion_dataloader:
    input_ids     = batch["input_ids"]       # [B, 128]
    knowledge_ids = batch["knowledge_ids"]   # [B, 64]（Oracle 知识）
    labels        = batch["labels"]          # [B, 128]，padding=-100

    # 前向传播（Hook 自动注入知识）
    output = modified_qwen(input_ids, knowledge_ids, attention_mask, labels)

    # Phase 2: 语言建模 loss（仅对有效 token）
    # Phase 3: QA SFT loss（同样的 CrossEntropy，标签为答案 token）
    loss = output.loss  # CrossEntropy(logits.view(-1,V), labels.view(-1), ignore=-100)

    accelerator.backward(loss)
    clip_grad_norm_(trainable_params, grad_clip)
    optimizer.step(); scheduler.step()
```

### 2.4 训练超参数

| 参数 | Phase 1 (Router) | Phase 2 (Fusion 预训练) | Phase 3 (SFT) |
|------|-----------------|------------------------|---------------|
| **数据** | 知识库文本对 | FineWeb-Edu | MedQA train |
| **lr** | 1e-3 | 3e-4 | 1e-4 |
| **warmup** | 200 steps | 100 steps | 50 steps |
| **调度器** | Cosine decay | Cosine decay | Cosine decay |
| **batch_size/卡** | 64 | 32 | 16 |
| **梯度累积** | 1 | 4 | 1 |
| **混合精度** | bf16 | bf16 | bf16 |
| **梯度裁剪** | 1.0 | 1.0 | 1.0 |
| **早停** | 无 | 无 | patience=3 |
| **GPU** | 2×（6,7） | 2×（6,7） | 2×（6,7） |
| **可训练** | PKM + FeatureAdapter + RefinedSelector | AttentionInjection × 4 + KnowledgeEncoder | 同 Phase 2 |
| **冻结** | Qwen3 全量 | Qwen3 + Router | Qwen3 + Router |

### 2.5 训练管线构建依赖

训练管线要求所有 §1.1-1.10 模块完成后才可运行：

| Phase | 依赖模块 | main.py 子命令 |
|-------|---------|---------------|
| Phase 0 知识构建 | §1.2,§1.3,§1.5 | `build-knowledge` |
| Phase 1 Router 训练 | §1.7（含 §1.4-1.6） | `train --phase 1` |
| Phase 2 Fusion 预训练 | §1.9（含 §1.8） | `train --phase 2` |
| Phase 3 SFT | §1.10 | `train --phase 3` |
| 评测 | §1.10 | `eval` |

---

## 3. 实验计划

> **状态**：框架已建立，结果待填入。
> **模型矩阵**：Qwen3-0.6B / 4B / 7B（三个基准尺度），14B+ 可选。
> **评测方法**：loglikelihood（对 " A"/" B"/" C"/" D" 取 continuation log-prob 最高者），跨所有实验统一。
> **数据集**：MedQA-USMLE (1,273) / ARC-Challenge (1,165) / MMLU (14,042)

### 0. 核心主张与论证逻辑

```
主张一（前提 A）：Router 能在百万级知识库中精确检索         ← E1 验证
  ↓ 路由质量是端到端系统的前提
主张二（前提 B）：Fusion 模块具有通用的知识融合能力         ← E2 + E3 验证
  ↓ 融合机制本身有效
主张三（核心贡献）：端到端系统（Router+Fusion）> RAG       ← E4 验证
  ↓ 完整系统优于输入拼接方案
主张四（核心卖点）：知识可动态增删更新，不损害性能          ← E6 验证
  ↓ 持续学习能力得证
主张五（效率 Scaling）：效率优势随模型规模增大而显著        ← E7 验证
  → O(L·K·d) vs O(L²·d) 的理论优势在 4B/7B 上得以验证
```

**训练管线与实验的时序关系**：

```
Phase 0 → Phase 1 → 【E1】→ Phase 2 → 【E2】【E3】→ Phase 3 → 【E4】【E5】→ 【E6】【E7】
```

**Oracle 知识协议**：所有 Fusion 相关实验（E2-E5）默认使用预配对知识，端到端实验（E4）同时测试 Oracle 和真实路由。

---

### E1：Router 路由质量验证 — 主张一（端到端的前提）

**设计原理**：Router 是端到端系统的入口。E1 是所有后续实验的必要前提——Phase 1 训练完成后必须立即运行，确认路由可用。

**实验设计**：

```
知识库: N=1M 条目（双 Bank）
查询: 评测集每道题
对比条件:
  ① 粗排 only — MemoryGate Top-16 clusters → ~64 候选
  ② 粗排 + 精排 — RefinedSelector 精选 Top-1
  ③ 随机 baseline（下界参考）
```

**判断标准**：Recall@64 > 80%（粗排合格），Recall@1 > 50%（端到端可用），Recall@1 < 30%（Router 训练失败）

#### E1-1 Recall@K 曲线（MedQA，N=1M）

| 模型 | 方法 | Recall@1 | Recall@4 | Recall@16 | Recall@64 |
|------|------|----------|----------|-----------|-----------|
| 0.6B | 粗排 only | | | | |
| 0.6B | 粗排 + 精排 | | | | |
| 0.6B | 随机 baseline | | | | |
| 4B | 粗排 only | | | | |
| 4B | 粗排 + 精排 | | | | |
| 4B | 随机 baseline | | | | |
| 7B | 粗排 only | | | | |
| 7B | 粗排 + 精排 | | | | |
| 7B | 随机 baseline | | | | |

#### E1-2 Cluster 负载均衡

| 模型 | Max Cluster Size | Mean Cluster Size | Max/Mean 比值 | 判定 |
|------|-----------------|-------------------|---------------|------|
| 0.6B | | | | |
| 4B | | | | |
| 7B | | | | |

> 比值 > 3 则失衡，需调整聚类策略。

#### E1-3 跨数据集 Recall@1（粗排+精排）

| 模型 | MedQA | ARC | MMLU |
|------|-------|-----|------|
| 0.6B | | | |
| 4B | | | |
| 7B | | | |

---

### E2：Fusion 知识驱动性验证（Sanity Check）— 主张二（前半）

**设计原理**：最基础的验证——Fusion 模块**真的在使用**注入的知识，而不是忽略知识仅靠模型本身能力答题。

**核心指标**: Knowledge Sensitivity (KS) = acc(正确) - acc(反事实)
**关键不等式**: acc(正确) > acc(无知识) > random(25%) > acc(反事实)

**判断标准**：KS > 20%（显著有效）；**旧项目参考值**（0.6B Phase 2）：KS = 56.32%

#### E2-1 Knowledge Sensitivity — MedQA

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

#### E2-2 Knowledge Sensitivity — ARC

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

#### E2-3 Knowledge Sensitivity — MMLU

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

---

### E3：跨域通用能力验证 — 主张二（后半）

**设计原理**：E2 证明 Fusion 在训练域有效。E3 证明融合能力是**通用的**——在从未 fine-tune 过的域上也有效。

**使用 Phase 2 权重（通用预训练，无 domain SFT）**

**判断标准**：三个域 Δacc 全部 > 0；**旧项目参考值**（0.6B Phase 2）：MedQA +3.6%, ARC +15.4%, MMLU +7.3%

#### E3-1 跨域通用能力（Phase 2 权重）

| 模型 | 条件 | MedQA | ARC | MMLU |
|------|------|-------|-----|------|
| 0.6B | Baseline（无知识） | | | |
| 0.6B | Fusion + Oracle 知识 | | | |
| 0.6B | Fusion + 空知识（消融） | | | |
| 0.6B | **Δacc（Fusion - Baseline）** | | | |
| 4B | Baseline（无知识） | | | |
| 4B | Fusion + Oracle 知识 | | | |
| 4B | Fusion + 空知识（消融） | | | |
| 4B | **Δacc（Fusion - Baseline）** | | | |
| 7B | Baseline（无知识） | | | |
| 7B | Fusion + Oracle 知识 | | | |
| 7B | Fusion + 空知识（消融） | | | |
| 7B | **Δacc（Fusion - Baseline）** | | | |

---

### E4：端到端系统 vs RAG — 主张三（核心贡献）

**设计原理**：核心实验。严格控制变量，证明完整端到端系统（Router + Fusion）优于传统 RAG（文本前缀拼接）。

**八组对比矩阵**：

```
Oracle 路由组（消除路由误差）:
  G0: Baseline — 无知识（下界）
  G1: RAG-compressed — 64 tokens 前缀拼接
  G2: Fusion-Phase2 + Oracle — 层注入（Phase 2 权重）
  G3: Fusion-Phase3 + Oracle — 层注入（Phase 3 权重）
  G4: RAG-original — ~256 tokens 前缀拼接（准确率上限）

真实路由组（端到端，含路由误差）:
  G5: Fusion-Phase3 + 真实路由
  G6: RAG-compressed + 真实路由
  G7: RAG-original + 真实路由
```

**核心对比关系**：

| 对比 | 控制变量 | 论证目标 |
|------|---------|---------|
| G1 vs G2 | 同 Oracle 知识、同 64 tokens | 层注入 > 文本拼接 |
| G2 vs G3 | 同 Oracle 知识、同注入方式 | Phase 3 SFT 的增益 |
| G3 vs G5 | 同 Phase 3 权重 | 量化路由误差影响 |
| G5 vs G7 | 同真实路由 | 端到端 Fusion vs 端到端 RAG |
| G3 vs G4 | Oracle、Fusion@64 vs RAG@256 | 知识利用效率 |

**旧项目参考值**（0.6B）：G3 MedQA 69.52%，G4 MedQA 86.96%

#### E4-1 八组对比矩阵 — MedQA

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

#### E4-2 八组对比矩阵 — ARC

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

#### E4-3 八组对比矩阵 — MMLU

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

#### E4-4 核心对比指标（MedQA）

| 指标 | 定义 | 0.6B | 4B | 7B |
|------|------|------|----|----|
| **Δacc(G3-G0)** | Fusion-Phase3 Oracle 增益 | | | |
| **Δacc(G4-G0)** | RAG-original 增益（上限） | | | |
| **知识利用效率** | Δacc(G3) / Δacc(G4)，64 tokens 达到 256 tokens 多少效果 | | | |
| **路由损失** | acc(G3) - acc(G5)，路由误差代价 | | | |
| **端到端优势** | acc(G5) - acc(G7)，端到端 Fusion vs 端到端 RAG | | | |
| **同 token Fusion vs RAG** | acc(G2) - acc(G1)，同 64 tokens | | | |

---

### E5：消融与深度分析 — 揭示系统内部机制

#### E5-A 训练阶段消融

**设计原理**：量化 Phase 3 (domain SFT) 相对 Phase 2 (通用预训练) 的增益，验证 SFT 是否引入跨域 trade-off。

| 模型 | 权重阶段 | MedQA | ARC | MMLU |
|------|---------|-------|-----|------|
| 0.6B | Phase 2 | | | |
| 0.6B | Phase 3 | | | |
| 0.6B | **Δ(Phase3 - Phase2)** | | | |
| 4B | Phase 2 | | | |
| 4B | Phase 3 | | | |
| 4B | **Δ(Phase3 - Phase2)** | | | |
| 7B | Phase 2 | | | |
| 7B | Phase 3 | | | |
| 7B | **Δ(Phase3 - Phase2)** | | | |

> 关注：ARC/MMLU 的 Δ 是否为负（跨域 trade-off）。旧项目结论为全面提升，新架构需要验证。

#### E5-B 知识 Token 预算分析

**设计原理**：验证 Fusion@64 > RAG@64（同 token 预算层注入更优），关键点：Fusion@128 是否仍 > RAG@256。

##### E5-B-1 MedQA（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

##### E5-B-2 ARC（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

##### E5-B-3 MMLU（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

#### E5-C 知识相关性分析

**设计原理**：排除"注入任何信号都能提升"的反驳，证明增益完全来自正确的语义对应关系。

```
三种知识条件 (k=64):
  Oracle：正确知识（question → 对应的 compressed answer）
  Shuffled：随机打乱映射（同分布但语义不匹配）
  Empty：无知识（全 pad）
预期: Oracle >> Empty ≈ Shuffled > random(25%)
```

| 模型 | 知识条件 | MedQA | ARC | MMLU |
|------|---------|-------|-----|------|
| 0.6B | Oracle（正确知识） | | | |
| 0.6B | Shuffled（随机打乱映射） | | | |
| 0.6B | Empty（无知识） | | | |
| 4B | Oracle（正确知识） | | | |
| 4B | Shuffled（随机打乱映射） | | | |
| 4B | Empty（无知识） | | | |
| 7B | Oracle（正确知识） | | | |
| 7B | Shuffled（随机打乱映射） | | | |
| 7B | Empty（无知识） | | | |

#### E5-D 注入方式消融

**设计原理**：三种注入方式（§1.8）在三个模型规模上全量对比，不同规模可能有不同最优方式。

**训练要求**: 3 注入方式 × 3 模型 = 9 组独立 Phase 2 + Phase 3 训练

##### E5-D-1 准确率对比（Phase 3 权重）

| 模型 | 注入方式 | 参数量/层 | MedQA | ARC | MMLU |
|------|---------|----------|-------|-----|------|
| 0.6B | AttentionInjection | ~4.2M | | | |
| 0.6B | ConcatProjection | ~12.6M | | | |
| 0.6B | GatedInjection | ~1K | | | |
| 4B | AttentionInjection | | | | |
| 4B | ConcatProjection | | | | |
| 4B | GatedInjection | | | | |
| 7B | AttentionInjection | | | | |
| 7B | ConcatProjection | | | | |
| 7B | GatedInjection | | | | |

##### E5-D-2 推理效率对比

| 模型 | 注入方式 | 总可训练参数 | 延迟 (ms/sample) | 峰值显存 (MB) |
|------|---------|------------|-----------------|--------------|
| 0.6B | AttentionInjection | | | |
| 0.6B | ConcatProjection | | | |
| 0.6B | GatedInjection | | | |
| 4B | AttentionInjection | | | |
| 4B | ConcatProjection | | | |
| 4B | GatedInjection | | | |
| 7B | AttentionInjection | | | |
| 7B | ConcatProjection | | | |
| 7B | GatedInjection | | | |

---

### E6：动态更新验证 — 主张四（核心卖点）

**设计原理**：ExplicitLM 的核心定位是"显式持续学习"。知识在 Bank 中可部署后动态增删更新，无需重训参数。

E6 需要证明：①批量灌入新知识后旧知识性能不受影响（E6-A），②增量增删后系统性能平稳（E6-B），③近似 cluster 分配质量衰减可控（E6-C）。

#### E6-A 批量灌入（跨域知识灌入后原域性能保持）

**流程**：

```
1. Phase 3 权重 + 原始知识库（MedQA）→ 评测 acc_before
2. 批量灌入 ARC 知识 → 全量 recluster
3. 重新评测 MedQA（不应显著下降）+ 评测 ARC（应有提升）
```

| 模型 | 指标 | 灌入前 | 灌入后(MedQA) | 灌入后(ARC) |
|------|------|--------|-------------|------------|
| 0.6B | acc | | | |
| 0.6B | Recall@1 | | | |
| 4B | acc | | | |
| 4B | Recall@1 | | | |
| 7B | acc | | | |
| 7B | Recall@1 | | | |

> **判断标准**：灌入后 MedQA acc 下降 < 2%，ARC 可检索且有提升。

#### E6-B 增量热更新（增删条目后性能变化）

##### E6-B-1 增量 Add 后性能

| 模型 | 新增条目 | acc | Recall@1 | Recall@64 |
|------|---------|-----|----------|-----------|
| 0.6B | +0（基线） | | | |
| 0.6B | +100 | | | |
| 0.6B | +500 | | | |
| 0.6B | +1000 | | | |
| 0.6B | +5000 | | | |
| 4B | +0（基线） | | | |
| 4B | +100 | | | |
| 4B | +500 | | | |
| 4B | +1000 | | | |
| 4B | +5000 | | | |
| 7B | +0（基线） | | | |
| 7B | +100 | | | |
| 7B | +500 | | | |
| 7B | +1000 | | | |
| 7B | +5000 | | | |

##### E6-B-2 增量 Delete 后性能

| 模型 | 删除条目 | acc | Recall@1 | Recall@64 |
|------|---------|-----|----------|-----------|
| 0.6B | -0（基线） | | | |
| 0.6B | -100 | | | |
| 0.6B | -500 | | | |
| 0.6B | -1000 | | | |
| 4B | -0（基线） | | | |
| 4B | -100 | | | |
| 4B | -500 | | | |
| 4B | -1000 | | | |
| 7B | -0（基线） | | | |
| 7B | -100 | | | |
| 7B | -500 | | | |
| 7B | -1000 | | | |

##### E6-B-3 Compact + Recluster 后恢复

| 模型 | 阶段 | acc | Recall@1 | Recall@64 |
|------|------|-----|----------|-----------|
| 0.6B | 初始 | | | |
| 0.6B | 增删后（compact 前） | | | |
| 0.6B | compact + recluster 后 | | | |
| 4B | 初始 | | | |
| 4B | 增删后（compact 前） | | | |
| 4B | compact + recluster 后 | | | |
| 7B | 初始 | | | |
| 7B | 增删后（compact 前） | | | |
| 7B | compact + recluster 后 | | | |

#### E6-C 近似分配质量衰减曲线

**设计原理**：量化近似 cluster 分配的质量衰减，确定合理的 recluster 触发阈值（`change_ratio`）。

```
变量: change_ratio = change_counter / N_valid
流程: 全量 recluster 基线 → 每次 add 1% → 测量 Recall@K → 重复至 20% → 触发 recluster
```

| 模型 | change_ratio | Recall@1 | Recall@4 | Recall@16 | Recall@64 |
|------|-------------|----------|----------|-----------|-----------|
| 0.6B | 0%（recluster 基线） | | | | |
| 0.6B | 1% | | | | |
| 0.6B | 2% | | | | |
| 0.6B | 5% | | | | |
| 0.6B | 10% | | | | |
| 0.6B | 15% | | | | |
| 0.6B | 20% | | | | |
| 0.6B | recluster 后恢复 | | | | |
| 4B | 0%（recluster 基线） | | | | |
| 4B | 5% | | | | |
| 4B | 10% | | | | |
| 4B | 20% | | | | |
| 4B | recluster 后恢复 | | | | |
| 7B | 0%（recluster 基线） | | | | |
| 7B | 5% | | | | |
| 7B | 10% | | | | |
| 7B | 20% | | | | |
| 7B | recluster 后恢复 | | | | |

> **期望**：Recall@K 随 change_ratio 缓慢下降，threshold=0.1（10%）附近仍可接受。recluster 后恢复到初始水平。

---

### E7：Scaling & 推理效率 — 主张五（效率论证）

**设计原理**：Fusion 计算复杂度 O(L·K·d)（K=64 固定），RAG self-attention 为 O(L²·d)。理论上 d 增大时 Fusion 效率优势越明显。旧项目发现 0.6B 上 Fusion 延迟 +30%，必须在更大模型上验证交叉点。

#### E7-A 推理效率基准测试

> **配置**：单 GPU, batch_size=1, N=200 样本, MedQA

| 模型 | 方法 | 延迟 (ms/sample) | 吞吐 (samples/s) | 峰值显存 (MB) |
|------|------|-----------------|-----------------|--------------|
| 0.6B | Baseline | | | |
| 0.6B | RAG-compressed@64 | | | |
| 0.6B | Fusion@64 | | | |
| 0.6B | RAG-original@256 | | | |
| 4B | Baseline | | | |
| 4B | RAG-compressed@64 | | | |
| 4B | Fusion@64 | | | |
| 4B | RAG-original@256 | | | |
| 7B | Baseline | | | |
| 7B | RAG-compressed@64 | | | |
| 7B | Fusion@64 | | | |
| 7B | RAG-original@256 | | | |

> **旧项目参考**（0.6B）：Fusion 延迟 +30%、显存 -17.1%

#### E7-B Scaling 趋势

> **理论预测**：找到 Fusion 延迟 < RAG 的交叉点。若 7B 上仍未交叉 → 外推至 14B+。

| 指标 | 0.6B | 4B | 7B | 14B (可选) |
|------|------|----|----|-----------|
| Fusion 延迟 (ms) | | | | |
| RAG-original 延迟 (ms) | | | | |
| Fusion / RAG 延迟比 | | | | |
| Fusion 显存 (MB) | | | | |
| RAG-original 显存 (MB) | | | | |
| **交叉点（延迟比 < 1.0）** | — | — | — | — |

#### E7-C 六维对比框架

##### Qwen3-0.6B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

##### Qwen3-4B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

##### Qwen3-7B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

---

### 附录 A：训练进度追踪

| Phase | 模型 | 注入方式 | 状态 | 最优 checkpoint | 备注 |
|-------|------|---------|------|----------------|------|
| Phase 0 | 0.6B | — | ⬜ | — | |
| Phase 1 | 0.6B | — | ⬜ | | |
| Phase 2 | 0.6B | AttentionInjection | ⬜ | | |
| Phase 2 | 0.6B | ConcatProjection | ⬜ | | |
| Phase 2 | 0.6B | GatedInjection | ⬜ | | |
| Phase 3 | 0.6B | AttentionInjection | ⬜ | | |
| Phase 3 | 0.6B | ConcatProjection | ⬜ | | |
| Phase 3 | 0.6B | GatedInjection | ⬜ | | |
| Phase 0 | 4B | — | ⬜ | — | |
| Phase 1 | 4B | — | ⬜ | | |
| Phase 2 | 4B | AttentionInjection | ⬜ | | |
| Phase 2 | 4B | ConcatProjection | ⬜ | | |
| Phase 2 | 4B | GatedInjection | ⬜ | | |
| Phase 3 | 4B | AttentionInjection | ⬜ | | |
| Phase 3 | 4B | ConcatProjection | ⬜ | | |
| Phase 3 | 4B | GatedInjection | ⬜ | | |
| Phase 0 | 7B | — | ⬜ | — | |
| Phase 1 | 7B | — | ⬜ | | |
| Phase 2 | 7B | AttentionInjection | ⬜ | | |
| Phase 2 | 7B | ConcatProjection | ⬜ | | |
| Phase 2 | 7B | GatedInjection | ⬜ | | |
| Phase 3 | 7B | AttentionInjection | ⬜ | | |
| Phase 3 | 7B | ConcatProjection | ⬜ | | |
| Phase 3 | 7B | GatedInjection | ⬜ | | |

> ⬜ 待开始 | 🔄 进行中 | ✅ 完成 | ❌ 失败

### 附录 B：实验执行进度

| 实验 | 0.6B | 4B | 7B |
|------|------|----|----|
| E1 Router 路由质量 | ⬜ | ⬜ | ⬜ |
| E2 知识驱动性验证 | ⬜ | ⬜ | ⬜ |
| E3 跨域通用能力 | ⬜ | ⬜ | ⬜ |
| E4 端到端 vs RAG | ⬜ | ⬜ | ⬜ |
| E5-A 训练阶段消融 | ⬜ | ⬜ | ⬜ |
| E5-B Token 预算分析 | ⬜ | ⬜ | ⬜ |
| E5-C 知识相关性 | ⬜ | ⬜ | ⬜ |
| E5-D 注入方式消融 | ⬜ | ⬜ | ⬜ |
| E6-A 批量灌入 | ⬜ | ⬜ | ⬜ |
| E6-B 增量热更新 | ⬜ | ⬜ | ⬜ |
| E6-C 衰减曲线 | ⬜ | ⬜ | ⬜ |
| E7-A 推理效率 | ⬜ | ⬜ | ⬜ |
| E7-B Scaling 趋势 | ⬜ | ⬜ | ⬜ |

### 附录 C：资源估算

#### 主线训练（AttentionInjection）

| Phase | 0.6B | 4B | 7B |
|-------|------|----|----|
| Phase 0 (知识构建) | 1 GPU × 0.5天 | 1 GPU × 1天 | 1 GPU × 1.5天 |
| Phase 1 (Router) | 2 GPU × 1天 | 4 GPU × 2天 | 8 GPU × 3天 |
| Phase 2 (Fusion) | 2 GPU × 2天 | 4 GPU × 4天 | 8 GPU × 5天 |
| Phase 3 (SFT) | 2 GPU × 0.5天 | 4 GPU × 1天 | 8 GPU × 2天 |
| **小计** | **~4天** | **~8天** | **~11.5天** |

#### E5-D 消融额外训练（ConcatProjection + GatedInjection × Phase 2+3）

| 额外训练 | 0.6B | 4B | 7B |
|---------|------|----|----|
| 2 注入方式 × (Phase 2 + Phase 3) | +5天 | +10天 | +14天 |

> **总训练量**：0.6B ~9天 + 4B ~18天 + 7B ~25.5天 ≈ **~52.5 GPU·天**

---

## 4. 文件结构与依赖

### 4.1 目录树

```
ExplicitLM-LoRA/
├── main.py                              # CLI 入口（eval / build-knowledge / train 子命令）
├── pipeline.py                          # §1.10 端到端推理管线
├── config.py                            # §1.1 Config dataclass 定义
├── config/
│   ├── default.yaml                     # 全量非敏感配置（必须写全）
│   └── model_configs/                   # 0.6B/4B/7B 的模型特定配置
├── models/                              # 融合模块
│   ├── __init__.py
│   ├── qwen_wrapper.py                  # §1.3 KnowledgeEncoder（Qwen3 前 6 层）
│   ├── injection_modules.py             # §1.8 AttentionInjection / ConcatProjection / GatedInjection
│   └── modified_model.py               # §1.9 ModifiedQwen（Hook 注入核心）
├── router/                              # 路由模块
│   ├── __init__.py
│   ├── memory_bank.py                   # §1.2 FusionBank + AnchorBank + DualKnowledgeStore
│   ├── memory_gate.py                   # §1.4 ProductKeyMemory（粗排）
│   ├── clustering.py                    # §1.5 SubspaceClustering（聚类 + 近似分配）
│   ├── feature_adapter.py              # §1.6 FeatureAdapter
│   ├── refined_selector.py             # §1.6 RefinedSelector（精排）
│   └── model.py                         # §1.7 MemoryRouter（整合）
├── data_builder/                        # Phase 0 数据构建
│   ├── compressor.py                    # LLMLingua-2 压缩器封装
│   ├── data_loader.py                   # 流式数据加载（FineWeb-Edu）
│   └── parallel_pipeline.py            # 多 GPU 并行构建（N=1M）
├── training/                            # 训练器
│   ├── router_trainer.py               # Phase 1 Router 训练（RouterLoss）
│   ├── fusion_trainer.py               # Phase 2/3 Fusion 训练（FusionLoss）
│   └── dataset.py                       # ExplicitDataset + MedQADataset
├── evaluation/                          # 评测
│   ├── compare_eval.py                  # E4 八组对比实验
│   └── run_eval.py                      # E1-E7 评测入口
├── utils/
│   ├── logger_system.py                 # log_msg, log_json, ensure, log_exception
│   └── helpers.py                       # 通用工具函数
├── scripts/                             # 训练脚本
│   ├── run_phase0_build.sh              # Phase 0 知识构建
│   ├── run_phase1_router.sh             # Phase 1 Router 训练
│   ├── run_phase2_fusion.sh             # Phase 2 Fusion 预训练
│   ├── run_phase3_sft.sh                # Phase 3 SFT
│   └── run_compare_eval.sh             # E4 对比实验
├── tests/
│   ├── unit/
│   │   ├── test_memory_bank.py          # DualKnowledgeStore 增删索引
│   │   ├── test_product_key_memory.py   # PKM 检索正确性
│   │   ├── test_clustering.py           # 聚类平衡性 + 近似分配
│   │   ├── test_attention_injection.py  # 零初始化 + 残差
│   │   └── test_modified_qwen.py        # Hook 注入 + forward pass
│   ├── integration/
│   │   ├── test_router_pipeline.py      # Router 端到端检索
│   │   └── test_full_pipeline.py        # 完整推理管线
│   └── outputs/                         # Agent 测试 MD 输出
├── data/                                # 数据集（不提交）
├── checkpoints/                         # 模型权重（不提交）
│   ├── phase1_best/                     # Router 最优权重
│   ├── phase2_best/                     # Fusion 预训练最优权重
│   └── phase3_best/                     # SFT 最优权重
├── logs/                                # 运行日志（不提交）
│   ├── system.log
│   └── metrics.json
├── .env                                 # API 密钥（不提交）
├── .env.example                         # 环境变量模板
└── requirements.txt
```

### 4.2 模块依赖关系

```
config.py ← (所有模块依赖)

router/memory_bank.py ← router/memory_gate.py
                      ← router/refined_selector.py
                      ← router/model.py
                      ← data_builder/*.py

router/clustering.py ← router/memory_bank.py
                     ← training/router_trainer.py

router/feature_adapter.py ← router/model.py

router/refined_selector.py ← router/model.py

router/model.py ← models/qwen_wrapper.py
                ← pipeline.py
                ← training/router_trainer.py

models/qwen_wrapper.py ← models/modified_model.py
                       ← router/model.py

models/injection_modules.py ← models/modified_model.py

models/modified_model.py ← pipeline.py
                         ← training/fusion_trainer.py

pipeline.py ← main.py
            ← evaluation/compare_eval.py
```

### 4.3 Python 依赖

```
# 核心框架
torch>=2.0
transformers>=4.40
accelerate>=0.28        # 分布式训练

# 知识压缩
llmlingua               # LLMLingua-2 压缩器

# 数据
datasets>=2.0           # FineWeb-Edu 加载
safetensors             # 权重序列化

# 聚类
numpy
scipy                   # SVD for PCA

# 配置
pyyaml
python-dotenv

# 评测
lm-eval>=0.4            # loglikelihood 评测框架

# 代码质量
ruff
pytest
pytest-cov

# 日志
rich                    # 终端输出格式化
```
