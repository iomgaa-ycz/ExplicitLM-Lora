"""
config.py 单元测试

测试覆盖：
    1. 仅 YAML 加载，验证字段类型正确
    2. .env 覆盖 paths.model_dir
    3. CLI override 覆盖 injection_layers
    4. CLI 优先级高于 .env（同字段 CLI 胜出）
    5. YAML 缺少必填字段时抛出 KeyError/TypeError
    6. injection_method 为 "gated" 时正常加载
    7. 点路径格式错误时抛出 ValueError

运行命令：
    conda run -n ExplicitLLM python -m pytest tests/unit/test_config.py -v --tb=short
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
import yaml

from config import (
    Config,
    DataConfig,
    EvalConfig,
    ModelConfig,
    PathsConfig,
    RouterConfig,
    TrainConfig,
    load_config,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

VALID_YAML_CONTENT = textwrap.dedent(
    """\
    model:
      base_model: "Qwen/Qwen3-0.6B"
      hidden_dim: 1024
      num_layers: 28
      injection_method: "attention"
      injection_layers: [6, 12, 18, 24]
      encoder_depth: 6
      fusion_length: 64
      anchor_length: 128

    router:
      knowledge_num: 1048576
      dim: 1024
      query_dim: 1024
      key_proj_dim: 512
      adapter_dim: 512
      num_candidates: 32
      temperature: 0.1
      recluster_threshold: 0.1
      max_candidates_per_cell: -1
      refined_num_heads: 8
      refined_num_layers: 2

    train:
      phase1_lr: 1.0e-3
      phase2_lr: 3.0e-4
      phase3_lr: 1.0e-4
      phase1_batch_size: 64
      phase2_batch_size: 32
      phase3_batch_size: 16
      phase1_max_epochs: 3
      phase2_max_epochs: 5
      phase3_max_epochs: 10
      patience: 3
      phase1_warmup_steps: 200
      phase2_warmup_steps: 100
      phase3_warmup_steps: 50
      grad_clip: 1.0
      bf16: true
      phase1_gradient_accumulation_steps: 8
      phase2_gradient_accumulation_steps: 4
      phase3_gradient_accumulation_steps: 1
      phase1_recluster_batch_size: 10240

    data:
      fusion_length: 64
      anchor_length: 128
      num_workers: 4
      train_max_samples: -1
      phase1_parquet_dir: "data/compressed/v2"
      phase1_tokenize_batch_size: 10000
      phase1_recluster_chunk_size: 512
      phase2_n_samples_per_epoch: 131072
      phase3_max_seq_length: 256

    paths:
      model_dir: "models/qwen3-0.6B"
      llmlingua_model_dir: "models/llmlingua-2"
      data_dir: "data"
      checkpoint_dir: "checkpoints"
      log_dir: "logs"
      results_dir: "results"

    swanlab:
      project: "explicit-lora-test"
      enabled: false
      log_every_n_steps: 50
      phase1_log_acc_steps: 1000
      phase1_log_recall_k: 8

    eval:
      medqa_knowledge_map: "data/medqa_knowledge.jsonl"
      arc_knowledge_map: "data/arc_knowledge.jsonl"
      mmlu_knowledge_map: "data/mmlu_knowledge.jsonl"
      lm_eval_tasks:
        - "medqa_4options"
      num_fewshot: 0
    """
)


@pytest.fixture
def valid_yaml_file(tmp_path: Path) -> str:
    """
    创建有效 YAML 配置文件，返回其路径字符串。
    使用 tmp_path 确保测试间隔离。
    """
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(VALID_YAML_CONTENT, encoding="utf-8")
    return str(yaml_file)


# ─────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────


def test_load_from_yaml_only(valid_yaml_file: str) -> None:
    """
    测试 1：仅从 YAML 加载配置，验证所有字段类型和值正确。

    验证点：
        - 返回类型为 Config
        - 各子 dataclass 类型正确
        - 关键字段值与 YAML 一致
        - injection_layers 为 list[int]
    """
    # 隔离 .env（防止本地 .env 干扰测试）
    with mock.patch.dict(os.environ, {}, clear=False):
        cfg = load_config(valid_yaml_file)

    assert isinstance(cfg, Config)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.router, RouterConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.paths, PathsConfig)
    assert isinstance(cfg.eval, EvalConfig)

    # 关键字段验证
    assert cfg.model.base_model == "Qwen/Qwen3-0.6B"
    assert cfg.model.hidden_dim == 1024
    assert cfg.model.num_layers == 28
    assert cfg.model.injection_layers == [6, 12, 18, 24]
    assert isinstance(cfg.model.injection_layers, list)

    assert cfg.router.knowledge_num == 1048576
    assert cfg.router.temperature == pytest.approx(0.1)

    assert cfg.train.phase2_lr == pytest.approx(3e-4)
    assert cfg.train.bf16 is True
    assert cfg.train.patience == 3

    assert cfg.data.train_max_samples == -1
    assert cfg.paths.model_dir == "models/qwen3-0.6B"
    assert cfg.eval.num_fewshot == 0
    assert cfg.eval.lm_eval_tasks == ["medqa_4options"]


def test_env_override_model_path(valid_yaml_file: str, tmp_path: Path) -> None:
    """
    测试 2：.env 文件中的 MODEL_PATH 覆盖 paths.model_dir。

    验证点：
        - .env 中的 MODEL_PATH 被读取并覆盖 YAML 中的 paths.model_dir
        - 未被 .env 覆盖的字段保持 YAML 原值
    """
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_PATH=/opt/models/qwen3\n", encoding="utf-8")

    # 切换工作目录到 tmp_path，使 load_config 能找到 .env
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # 清除进程环境变量中的 MODEL_PATH，确保只通过 .env 覆盖
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_PATH", None)
            cfg = load_config(valid_yaml_file)
    finally:
        os.chdir(original_cwd)

    assert cfg.paths.model_dir == "/opt/models/qwen3"
    # 其他路径字段不受影响
    assert cfg.paths.data_dir == "data"


def test_cli_override_injection_layers(valid_yaml_file: str) -> None:
    """
    测试 3：CLI override 覆盖 model.injection_layers。

    验证点：
        - 点路径格式 "model.injection_layers" 正确解析
        - 覆盖值替换 YAML 原值
        - 其他字段不受影响
    """
    cli_overrides: Dict[str, Any] = {"model.injection_layers": [4, 8, 12]}
    cfg = load_config(valid_yaml_file, cli_overrides=cli_overrides)

    assert cfg.model.injection_layers == [4, 8, 12]
    # 同 section 其他字段不受影响
    assert cfg.model.hidden_dim == 1024
    assert cfg.model.injection_method == "attention"


def test_cli_priority_over_env(valid_yaml_file: str, tmp_path: Path) -> None:
    """
    测试 4：CLI 优先级高于 .env（同字段 CLI 胜出）。

    场景：
        .env 设置 MODEL_PATH=/from/env
        CLI 设置 paths.model_dir=/from/cli
        最终 cfg.paths.model_dir 应为 /from/cli
    """
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_PATH=/from/env\n", encoding="utf-8")

    cli_overrides: Dict[str, Any] = {"paths.model_dir": "/from/cli"}

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_PATH", None)
            cfg = load_config(valid_yaml_file, cli_overrides=cli_overrides)
    finally:
        os.chdir(original_cwd)

    # CLI 覆盖 .env（CLI 在 _override_from_cli 中最后执行）
    assert cfg.paths.model_dir == "/from/cli"


def test_missing_section_raises_keyerror(tmp_path: Path) -> None:
    """
    测试 5：YAML 缺少必填 section 时抛出 KeyError。

    验证点：
        - 删除 YAML 中的 router 段
        - load_config() 在加载阶段立即抛 KeyError（不使用默认值）
    """
    incomplete_yaml = yaml.safe_load(VALID_YAML_CONTENT)
    del incomplete_yaml["router"]

    yaml_file = tmp_path / "incomplete.yaml"
    yaml_file.write_text(yaml.dump(incomplete_yaml), encoding="utf-8")

    with pytest.raises(KeyError, match="router"):
        load_config(str(yaml_file))


def test_missing_field_raises_typeerror(tmp_path: Path) -> None:
    """
    测试 5b：YAML 中某 section 缺少必填字段时抛出 TypeError。

    验证点：
        - model 段缺少 hidden_dim 字段
        - load_config() 抛 TypeError（dataclass 无默认值）
        - 错误信息包含 ModelConfig 提示
    """
    incomplete_yaml = yaml.safe_load(VALID_YAML_CONTENT)
    del incomplete_yaml["model"]["hidden_dim"]

    yaml_file = tmp_path / "missing_field.yaml"
    yaml_file.write_text(yaml.dump(incomplete_yaml), encoding="utf-8")

    with pytest.raises(TypeError, match="ModelConfig"):
        load_config(str(yaml_file))


def test_injection_method_gated(valid_yaml_file: str) -> None:
    """
    测试 6：injection_method 为 "gated" 时正常加载。

    验证点：
        - 通过 CLI override 将 injection_method 设为 "gated"
        - 配置成功加载，无异常
    """
    cli_overrides: Dict[str, Any] = {"model.injection_method": "gated"}
    cfg = load_config(valid_yaml_file, cli_overrides=cli_overrides)

    assert cfg.model.injection_method == "gated"


def test_cli_invalid_path_format_raises_valueerror(valid_yaml_file: str) -> None:
    """
    测试 7：CLI 点路径格式错误（少于两级）时抛出 ValueError。

    验证点：
        - 传入 "hidden_dim"（非点路径）
        - load_config() 抛 ValueError，提示格式错误
    """
    cli_overrides: Dict[str, Any] = {"hidden_dim": 2048}  # 缺少 section 前缀

    with pytest.raises(ValueError, match="格式错误"):
        load_config(valid_yaml_file, cli_overrides=cli_overrides)


def test_yaml_file_not_found() -> None:
    """
    测试 8：YAML 文件不存在时抛出 FileNotFoundError。
    """
    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        load_config("/nonexistent/path/config.yaml")
