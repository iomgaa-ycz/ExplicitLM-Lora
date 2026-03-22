"""
Explicit-LoRA 生产 CLI 入口

子命令：
    build-knowledge  Phase 0: 知识构建（LLMLingua 压缩 + 双 Bank 写入）
    train            Phase 1-3: 训练管线
    eval             评测入口
    answer           端到端 QA

用法：
    conda run -n ExplicitLLM python main.py build-knowledge --config config/default.yaml
    conda run -n ExplicitLLM python main.py train --phase 1 --config config/default.yaml
    conda run -n ExplicitLLM python main.py eval --config config/default.yaml
    conda run -n ExplicitLLM python main.py answer --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# 路径设置
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from pipeline import ExplicitLMPipeline  # noqa: E402


# ─────────────────────────────────────────────
# CLI 参数解析
# ─────────────────────────────────────────────


def _parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """
    解析 key=value 列表为嵌套 dict。

    参数：
        overrides: action="append" 产生的嵌套列表，如 [["a=1"], ["b=2"]]，
                   或 None（无 --override 时）

    返回：
        扁平 dict，key 为点路径，value 为字符串
    """
    if overrides is None:
        return {}

    # 展平嵌套列表（action="append" 产生嵌套结构）
    flat: list[str] = []
    for item in overrides:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    result: dict[str, Any] = {}
    for item in flat:
        if "=" not in item:
            print(f"[ERROR] 无效的 override 格式（缺少 '='）: {item}")
            sys.exit(1)
        key, value = item.split("=", 1)
        # 尝试转换类型
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


# ─────────────────────────────────────────────
# 子命令实现（占位）
# ─────────────────────────────────────────────


def _cmd_build_knowledge(cfg: Any, args: argparse.Namespace) -> None:
    """Phase 0: 知识构建（LLMLingua 压缩 → Fusion Bank + Anchor Bank）。"""
    print("[WARN] build-knowledge: 前置模块 §1.5 (clustering) 未实现，无法执行。")


def _cmd_train(cfg: Any, args: argparse.Namespace) -> None:
    """Phase 1-3: 训练管线。"""
    device = getattr(args, "device", "cpu")
    if args.phase == 1:
        from training.phase1_router import train_phase1

        train_phase1(cfg, device)
    elif args.phase == 2:
        from training.phase2_fusion import train_phase2

        train_phase2(cfg, device)
    elif args.phase == 3:
        from training.phase3_sft import train_phase3

        phase2_ckpt = getattr(args, "from_phase2", None)
        train_phase3(cfg, device, phase2_ckpt=phase2_ckpt)
    else:
        print(f"[WARN] train --phase {args.phase}: 未知阶段")


def _cmd_eval(cfg: Any, args: argparse.Namespace) -> None:
    """
    评测入口：从 checkpoint 加载 ExplicitLMPipeline，打印加载状态。

    当前实现为管线加载验证（smoke test），完整评测逻辑在训练管线完成后接入。
    """
    device = getattr(args, "device", "cpu")
    router_ckpt = str(Path(cfg.paths.checkpoint_dir) / "phase1_best")
    fusion_ckpt = str(Path(cfg.paths.checkpoint_dir) / "phase2_best")
    store_path = str(Path(cfg.paths.data_dir) / "store.pt")

    if not Path(store_path).exists():
        print(f"[WARN] eval: 知识库文件不存在 ({store_path})，跳过加载。")
        print("[INFO] eval: 请先运行 build-knowledge 构建知识库。")
        return

    pipeline = ExplicitLMPipeline.from_checkpoints(
        config=cfg,
        router_ckpt=router_ckpt,
        fusion_ckpt=fusion_ckpt,
        store_path=store_path,
        device=device,
    )
    print(f"[INFO] eval: ExplicitLMPipeline 加载成功 (device={device})")
    print(f"[INFO] eval: 知识库条目数 next_free={pipeline._store.next_free}")


def _cmd_answer(cfg: Any, args: argparse.Namespace) -> None:
    """
    端到端 QA：从 checkpoint 加载管线，对输入问题生成答案。

    用法：python main.py --device cuda:0 answer --question "What causes pneumonia?"
    """
    question: str = getattr(args, "question", None)
    if not question:
        print("[ERROR] answer: 请通过 --question 指定问题文本。")
        return

    device = getattr(args, "device", "cpu")
    router_ckpt = str(Path(cfg.paths.checkpoint_dir) / "phase1_best")
    fusion_ckpt = str(Path(cfg.paths.checkpoint_dir) / "phase2_best")
    store_path = str(Path(cfg.paths.data_dir) / "store.pt")

    if not Path(store_path).exists():
        print(f"[WARN] answer: 知识库文件不存在 ({store_path})，跳过。")
        print("[INFO] answer: 请先运行 build-knowledge 构建知识库。")
        return

    pipeline = ExplicitLMPipeline.from_checkpoints(
        config=cfg,
        router_ckpt=router_ckpt,
        fusion_ckpt=fusion_ckpt,
        store_path=store_path,
        device=device,
    )
    result = pipeline.answer(question, use_real_router=True)
    print(f"[Question] {question}")
    print(f"[Answer]   {result.answer}")
    print(f"[KnowledgeID] {result.retrieved_id}")
    print(f"[Latency]  {result.latency_ms:.1f} ms")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────


def main() -> None:
    """Explicit-LoRA CLI 入口函数。"""
    parser = argparse.ArgumentParser(description="Explicit-LoRA CLI")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--device", default="cpu", help="设备（cpu / cuda:0 等）")
    parser.add_argument("--override", nargs="?", action="append", help="key=value 配置覆盖（可多次指定）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 生产命令
    subparsers.add_parser("build-knowledge", help="Phase 0: 知识构建")
    train_parser = subparsers.add_parser("train", help="Phase 1-3: 训练管线")
    train_parser.add_argument(
        "--phase", type=int, required=True, choices=[0, 1, 2, 3], help="训练阶段"
    )
    train_parser.add_argument(
        "--from-phase2",
        type=str,
        default=None,
        dest="from_phase2",
        help="Phase 3 专用：Phase 2 最优 checkpoint 目录（默认 checkpoints/phase2_best）",
    )
    subparsers.add_parser("eval", help="评测入口")
    answer_parser = subparsers.add_parser("answer", help="端到端 QA")
    answer_parser.add_argument("--question", type=str, required=False, help="问题文本")

    args = parser.parse_args()
    cli_overrides = _parse_overrides(args.override or [])
    cfg = load_config(args.config, cli_overrides=cli_overrides)

    dispatch = {
        "build-knowledge": _cmd_build_knowledge,
        "train": _cmd_train,
        "eval": _cmd_eval,
        "answer": _cmd_answer,
    }
    dispatch[args.command](cfg, args)


if __name__ == "__main__":
    main()
