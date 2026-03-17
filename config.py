"""
配置管理模块

功能：
    - 定义项目所有配置的 dataclass（无默认值，强制 YAML 写全）
    - 实现三层优先级加载：YAML → .env → CLI args
    - 对外暴露唯一入口函数 load_config()

设计约束：
    - 所有 dataclass 字段严禁默认值，缺字段在加载时立即报错
    - 敏感信息（API Key、绝对路径）仅写 .env，不进 YAML

优先级（高 → 低）：
    CLI args > .env > config/default.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import dotenv_values


@dataclass
class ModelConfig:
    """
    模型结构配置

    参数：
        base_model: HuggingFace 模型路径或名称，如 "Qwen/Qwen3-0.6B"
        hidden_dim: 模型隐藏维度（Qwen3-0.6B 为 1024）
        num_layers: 模型 Transformer 层数（Qwen3-0.6B 为 28）
        injection_method: 知识注入方式，"attention" | "concat" | "gated"
        injection_layers: 注入模块所在层索引列表
        encoder_depth: 知识编码器使用的 Qwen3 前 N 层
        fusion_length: Fusion Bank 每条知识的 token 数（LLMLingua 压缩后）
        anchor_length: Anchor Bank 每条知识的 token 数（原文截断）
    """

    base_model: str
    hidden_dim: int
    num_layers: int
    injection_method: str
    injection_layers: List[int]
    encoder_depth: int
    fusion_length: int
    anchor_length: int


@dataclass
class RouterConfig:
    """
    路由器（Product Key Memory）配置

    参数：
        knowledge_num: 知识库总条目数（默认 1024*1024 = 1M）
        dim: 路由器内部隐藏维度
        query_dim: 查询向量维度
        key_proj_dim: Product Key Memory 子空间维度（= dim // 2）
        adapter_dim: FeatureAdapter 中间维度
        num_candidates: 粗排候选条目数
        temperature: PKM softmax 温度参数
        recluster_threshold: 触发重新聚类的变更比例阈值
        max_candidates_per_cell: 粗排每个 grid cell 最多取多少候选
            -1 = 全量（倒排索引，热更新多条/格场景）
            >0 = 每格上限（=1 时退化为 1:1 简单映射，对齐参考项目行为）
        refined_num_heads: RefinedSelector Transformer 注意力头数（adapter_dim 必须能整除）
        refined_num_layers: RefinedSelector Transformer 层数
    """

    knowledge_num: int
    dim: int
    query_dim: int
    key_proj_dim: int
    adapter_dim: int
    num_candidates: int
    temperature: float
    recluster_threshold: float
    max_candidates_per_cell: int
    refined_num_heads: int
    refined_num_layers: int


@dataclass
class TrainConfig:
    """
    三阶段训练超参数配置

    参数：
        phase1_lr: Phase 1（Router 训练）学习率
        phase2_lr: Phase 2（Fusion 预训练）学习率
        phase3_lr: Phase 3（下游 SFT）学习率
        phase1_batch_size: Phase 1 每卡 batch size
        phase2_batch_size: Phase 2 每卡 batch size
        phase3_batch_size: Phase 3 每卡 batch size
        phase1_max_epochs: Phase 1 最大训练轮次
        phase2_max_epochs: Phase 2 最大训练轮次
        phase3_max_epochs: Phase 3 最大训练轮次（配合早停）
        patience: 早停耐心轮次（Phase 3 val_loss 不下降则停止）
        phase1_warmup_steps: Phase 1 warmup 步数
        phase2_warmup_steps: Phase 2 warmup 步数
        phase3_warmup_steps: Phase 3 warmup 步数
        grad_clip: 梯度裁剪最大范数
        bf16: 是否使用 bfloat16 混合精度训练
        phase1_gradient_accumulation_steps: Phase 1 梯度累积步数（Accelerate）
        phase1_recluster_batch_size: Phase 1 重聚类时 encoder 编码批大小
    """

    phase1_lr: float
    phase2_lr: float
    phase3_lr: float
    phase1_batch_size: int
    phase2_batch_size: int
    phase3_batch_size: int
    phase1_max_epochs: int
    phase2_max_epochs: int
    phase3_max_epochs: int
    patience: int
    phase1_warmup_steps: int
    phase2_warmup_steps: int
    phase3_warmup_steps: int
    grad_clip: float
    bf16: bool
    phase1_gradient_accumulation_steps: int
    phase2_gradient_accumulation_steps: int
    phase3_gradient_accumulation_steps: int
    phase1_recluster_batch_size: int


@dataclass
class DataConfig:
    """
    数据处理配置

    参数：
        fusion_length: 压缩后知识 token 数（应与 ModelConfig.fusion_length 一致）
        anchor_length: 原文截断 token 数（应与 ModelConfig.anchor_length 一致）
        num_workers: DataLoader 并行 worker 数
        train_max_samples: 训练样本上限，-1 表示使用全量数据
        phase1_parquet_dir: Phase 1 Router 训练数据目录（预压缩 FineWeb-Edu Parquet）
        phase1_tokenize_batch_size: Phase 1 tokenize 批大小
        phase1_recluster_chunk_size: Phase 1 重聚类编码分块大小（避免 OOM）
    """

    fusion_length: int
    anchor_length: int
    num_workers: int
    train_max_samples: int
    phase1_parquet_dir: str
    phase1_tokenize_batch_size: int
    phase1_recluster_chunk_size: int
    phase2_n_samples_per_epoch: int
    phase3_max_seq_length: int


@dataclass
class PathsConfig:
    """
    文件系统路径配置

    参数：
        model_dir: 基础模型权重目录（可被 .env MODEL_PATH 覆盖）
        llmlingua_model_dir: LLMLingua 压缩模型目录（可被 .env LLMLINGUA_PATH 覆盖）
        data_dir: 数据根目录
        checkpoint_dir: 训练检查点保存目录
        log_dir: 日志输出目录
        results_dir: 评测结果保存目录
    """

    model_dir: str
    llmlingua_model_dir: str
    data_dir: str
    checkpoint_dir: str
    log_dir: str
    results_dir: str


@dataclass
class SwanLabConfig:
    """
    SwanLab 实验追踪配置

    参数：
        project: SwanLab 项目名称
        enabled: 是否启用 SwanLab 追踪（False 时跳过所有 swanlab 调用）
        log_every_n_steps: 每隔多少训练步上报一次指标
        phase1_log_acc_steps: Phase 1 每隔多少步打印 acc/row_acc/col_acc 并上报 SwanLab
        phase1_log_recall_k: step 级别 row/col recall 使用的 top-K（对齐参考项目 refine_top_k=8）
    """

    project: str
    enabled: bool
    log_every_n_steps: int
    phase1_log_acc_steps: int  # 每多少 step 打印 acc/row_acc/col_acc 并上报 SwanLab
    phase1_log_recall_k: int  # step 级别 row/col recall 使用的 top-K，对齐参考项目 refine_top_k=8


@dataclass
class EvalConfig:
    """
    评测配置

    参数：
        medqa_knowledge_map: MedQA 知识映射文件路径（.jsonl）
        arc_knowledge_map: ARC 知识映射文件路径（.jsonl）
        mmlu_knowledge_map: MMLU 知识映射文件路径（.jsonl）
        lm_eval_tasks: lm-eval 框架任务名列表
        num_fewshot: few-shot 示例数量（0 表示 zero-shot）
    """

    medqa_knowledge_map: str
    arc_knowledge_map: str
    mmlu_knowledge_map: str
    lm_eval_tasks: List[str]
    num_fewshot: int


@dataclass
class Config:
    """
    顶层配置容器，聚合所有子模块配置

    参数：
        model: 模型结构配置
        router: 路由器配置
        train: 训练超参数配置
        data: 数据处理配置
        paths: 文件系统路径配置
        eval: 评测配置
        swanlab: SwanLab 实验追踪配置
    """

    model: ModelConfig
    router: RouterConfig
    train: TrainConfig
    data: DataConfig
    paths: PathsConfig
    eval: EvalConfig
    swanlab: SwanLabConfig


# ─────────────────────────────────────────────
# Phase 1: 内部辅助函数
# ─────────────────────────────────────────────


def _load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    读取 YAML 配置文件，返回原始嵌套 dict。

    参数：
        yaml_path: YAML 文件路径

    返回：
        嵌套 dict，对应 YAML 结构

    异常：
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML 格式错误
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点必须是 dict，实际类型: {type(data)}")
    return data


def _override_from_env(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 .env 文件读取敏感配置，覆盖 config_dict 中对应的 paths 字段。

    支持的环境变量（仅覆盖 paths 段）：
        MODEL_PATH       → paths.model_dir
        LLMLINGUA_PATH   → paths.llmlingua_model_dir

    参数：
        config_dict: 已从 YAML 加载的嵌套 dict（原地修改并返回）

    返回：
        更新后的 config_dict
    """
    # dotenv_values 读取 .env 文件（不影响 os.environ）
    env_file = Path(".env")
    env_vals: Dict[str, Optional[str]] = (
        dotenv_values(env_file) if env_file.exists() else {}
    )

    # 同时查看实际进程环境变量（高优先级）
    env_map = {
        "MODEL_PATH": ("paths", "model_dir"),
        "LLMLINGUA_PATH": ("paths", "llmlingua_model_dir"),
    }

    paths_dict: Dict[str, Any] = config_dict.setdefault("paths", {})

    for env_key, (section, field) in env_map.items():
        # 进程环境变量优先于 .env 文件
        value = os.environ.get(env_key) or env_vals.get(env_key)
        if value:
            assert section == "paths", f"当前仅支持覆盖 paths 段，收到: {section}"
            paths_dict[field] = value

    return config_dict


def _override_from_cli(
    config_dict: Dict[str, Any],
    cli_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    将 CLI 扁平参数（点路径格式）覆盖到嵌套 config_dict。

    示例：
        cli_overrides = {"model.hidden_dim": 2048, "train.phase2_lr": 1e-4}
        → config_dict["model"]["hidden_dim"] = 2048
        → config_dict["train"]["phase2_lr"] = 1e-4

    参数：
        config_dict: 待覆盖的嵌套 dict（原地修改并返回）
        cli_overrides: 扁平点路径 dict，None 表示无 CLI 覆盖

    返回：
        更新后的 config_dict

    异常：
        KeyError: 点路径中某级 key 不存在（提前报错，避免静默）
        ValueError: 点路径格式错误（少于两级）
    """
    if not cli_overrides:
        return config_dict

    for dot_key, value in cli_overrides.items():
        parts = dot_key.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"CLI override 格式错误（需至少两级，如 'model.hidden_dim'）: {dot_key!r}"
            )
        # 逐级深入，最后一级赋值
        node: Dict[str, Any] = config_dict
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(
                    f"CLI override 路径 '{dot_key}' 中 '{part}' 不存在于配置"
                )
            node = node[part]
        node[parts[-1]] = value

    return config_dict


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    将合并后的嵌套 dict 构造为强类型 Config dataclass。

    关键实现：
        - 每个子 dataclass 使用 **dict 解包构造，dataclass 无默认值
        - 缺少任何字段时 Python 会立即抛 TypeError，不会静默使用默认值
        - List[int] 类型字段（如 injection_layers）需确保 YAML 加载为 list

    参数：
        config_dict: 已合并 YAML + .env + CLI 的嵌套 dict

    返回：
        强类型 Config 对象

    异常：
        TypeError: 缺少必填字段或类型不匹配
        KeyError: 顶层 section（model/router/...）缺失
    """
    # Phase 1: 校验顶层 section 存在
    required_sections = ["model", "router", "train", "data", "paths", "eval", "swanlab"]
    for section in required_sections:
        if section not in config_dict:
            raise KeyError(
                f"YAML 缺少顶层 section: '{section}'，已有: {list(config_dict.keys())}"
            )

    try:
        model_cfg = ModelConfig(**config_dict["model"])
    except TypeError as e:
        raise TypeError(f"ModelConfig 构造失败（检查 YAML model 段）: {e}") from e

    try:
        router_cfg = RouterConfig(**config_dict["router"])
    except TypeError as e:
        raise TypeError(f"RouterConfig 构造失败（检查 YAML router 段）: {e}") from e

    try:
        train_cfg = TrainConfig(**config_dict["train"])
    except TypeError as e:
        raise TypeError(f"TrainConfig 构造失败（检查 YAML train 段）: {e}") from e

    try:
        data_cfg = DataConfig(**config_dict["data"])
    except TypeError as e:
        raise TypeError(f"DataConfig 构造失败（检查 YAML data 段）: {e}") from e

    try:
        paths_cfg = PathsConfig(**config_dict["paths"])
    except TypeError as e:
        raise TypeError(f"PathsConfig 构造失败（检查 YAML paths 段）: {e}") from e

    try:
        eval_cfg = EvalConfig(**config_dict["eval"])
    except TypeError as e:
        raise TypeError(f"EvalConfig 构造失败（检查 YAML eval 段）: {e}") from e

    try:
        swanlab_cfg = SwanLabConfig(**config_dict["swanlab"])
    except TypeError as e:
        raise TypeError(f"SwanLabConfig 构造失败（检查 YAML swanlab 段）: {e}") from e

    return Config(
        model=model_cfg,
        router=router_cfg,
        train=train_cfg,
        data=data_cfg,
        paths=paths_cfg,
        eval=eval_cfg,
        swanlab=swanlab_cfg,
    )


# ─────────────────────────────────────────────
# Phase 2: 对外唯一入口
# ─────────────────────────────────────────────


def load_config(
    yaml_path: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    加载并合并配置，返回强类型 Config 对象。

    三层优先级（高 → 低）：
        1. CLI args（cli_overrides 参数，点路径格式）
        2. .env 文件（项目根目录下的 .env）
        3. YAML 文件（yaml_path 指定）

    参数：
        yaml_path: YAML 配置文件路径，如 "config/default.yaml"
        cli_overrides: CLI 覆盖参数，点路径扁平 dict，如
                       {"model.hidden_dim": 2048, "train.phase2_lr": 1e-4}
                       传 None 表示无 CLI 覆盖

    返回：
        强类型 Config 对象，包含所有子模块配置

    异常：
        FileNotFoundError: YAML 文件不存在
        KeyError: YAML 缺少必填 section 或 CLI 路径不存在
        TypeError: dataclass 字段缺失或类型不匹配

    示例：
        >>> cfg = load_config("config/default.yaml")
        >>> print(cfg.model.hidden_dim)    # 1024
        >>> print(cfg.router.temperature)  # 0.1

        >>> cfg = load_config(
        ...     "config/default.yaml",
        ...     cli_overrides={"model.injection_method": "gated"}
        ... )
        >>> assert cfg.model.injection_method == "gated"
    """
    # Phase 1: 加载 YAML 基础配置
    config_dict = _load_yaml(yaml_path)

    # Phase 2: .env 覆盖（敏感路径）
    config_dict = _override_from_env(config_dict)

    # Phase 3: CLI 覆盖（最高优先级）
    config_dict = _override_from_cli(config_dict, cli_overrides)

    # Phase 4: 构造强类型 dataclass
    return _dict_to_config(config_dict)
