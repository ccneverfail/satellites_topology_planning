# inspect_checkpoint.py
import os
import json
from collections import defaultdict
from typing import Dict, Any, Tuple
import torch
import json
from datetime import datetime, timedelta


def _json_default(o):
    if isinstance(o, timedelta):
        # 最通用：用总秒数表示时长（浮点）。需要整数可改成 microseconds。
        return o.total_seconds()
    if isinstance(o, datetime):
        # 若 report 里也有 datetime，一并处理
        return o.isoformat()
    # 若项目里常见 numpy 类型，也顺手处理下（可选）
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    # 兜底：转成字符串，避免再次抛错（按需保留/去掉）
    return str(o)


def _summarize_state_dict(sd: Dict[str, Any], name: str) -> Dict[str, Any]:
    """统计 state_dict：每个键的 shape/dtype/numel，总参数量，按 dtype 分布"""
    n_tensors = 0
    total_params = 0
    by_dtype = defaultdict(int)
    keys_meta = {}

    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            n_tensors += 1
            numel = v.numel()
            total_params += numel
            dt = str(v.dtype)
            by_dtype[dt] += numel
            keys_meta[k] = {
                "shape": list(v.shape),
                "numel": int(numel),
                "dtype": dt,
            }
        else:
            # 极少数 buffer/标量可能不是 Tensor
            keys_meta[k] = {"type": type(v).__name__}

    return {
        "name": name,
        "n_tensors": n_tensors,
        "total_params": int(total_params),
        "by_dtype": dict(by_dtype),
        "keys": keys_meta,  # 如需更小摘要，可去掉这一大段
    }

def _summarize_optimizer(optim_sd: Dict[str, Any]) -> Dict[str, Any]:
    """提取优化器的 param_groups（超参）与 state 里各字段的统计"""
    out: Dict[str, Any] = {}
    out["param_groups"] = optim_sd.get("param_groups", [])
    state = optim_sd.get("state", {})

    field_counts = defaultdict(int)
    n_states = 0
    for _pid, st in state.items():
        n_states += 1
        if isinstance(st, dict):
            for sk, sv in st.items():
                if isinstance(sv, torch.Tensor):
                    field_counts[f"{sk}:{str(sv.dtype)}"] += int(sv.numel())
                else:
                    field_counts[f"{sk}:{type(sv).__name__}"] += 1

    out["n_states"] = n_states
    out["state_field_counts"] = dict(field_counts)
    return out

def read_epoch_checkpoint(ckpt_path: str, save_json: bool = True) -> Dict[str, Any]:
    """
    读取 epoch checkpoint 并返回一个结构化字典。
    - 若是 epoch 保存（包含 model/target/optim/...），逐项汇总；
    - 若是纯 state_dict（只存了模型权重），也能给出摘要。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    report: Dict[str, Any] = {
        "path": ckpt_path,
        "size_bytes": os.path.getsize(ckpt_path),
        "format": type(ckpt).__name__,
    }

    # epoch 风格（推荐的保存格式）
    if isinstance(ckpt, dict) and "model" in ckpt:
        report["has_epoch_checkpoint"] = True
        report["epoch"] = ckpt.get("epoch")
        report["global_step"] = ckpt.get("global_step")
        report["epsilon"] = ckpt.get("epsilon")

        # 输出训练时间（若当时保存了）
        cfg_train_time = ckpt.get("cost_time_epoch")
        if cfg_train_time is not None:
            td = cfg_train_time if isinstance(cfg_train_time, timedelta) else timedelta(seconds=float(cfg_train_time))
            report["cost_time_epoch"] = td


        # 配置快照（若当时保存了）
        cfg_snapshot = ckpt.get("config")
        if cfg_snapshot is not None:
            report["config"] = cfg_snapshot

        # 模型/目标网络
        report["model"] = _summarize_state_dict(ckpt["model"], "model")
        if "target" in ckpt:
            report["target"] = _summarize_state_dict(ckpt["target"], "target")

        # 优化器
        if "optim" in ckpt:
            report["optimizer"] = _summarize_optimizer(ckpt["optim"])

    # 纯 state_dict（例如 step 保存只存了权重）
    elif isinstance(ckpt, dict):

        report["has_epoch_checkpoint"] = False
        report["state_dict"] = _summarize_state_dict(ckpt, "state_dict")

    else:
        # 其它非常规对象
        report["has_epoch_checkpoint"] = False

    # 控制台简要输出
    brief = {k: report.get(k) for k in ["path", "size_bytes", "epoch", "global_step", "epsilon", "has_epoch_checkpoint","cost_time_epoch"]}
    print(json.dumps(brief, indent=2, ensure_ascii=False, default=_json_default))

    # 若有 config，友好打印
    if report.get("config"):
        print("\n[config snapshot]")
        for k, v in report["config"].items():
            print(f"  {k}: {v}")

    # 简单提示参数总量
    model_meta = report.get("model") or report.get("state_dict")
    if model_meta:
        print(f"\n[model] total_params = {model_meta.get('total_params')}")

    # 可选：写出 *.summary.json
    if save_json:
        out_path = ckpt_path + ".summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            # json.dump(report, f, indent=2, ensure_ascii=False)
            json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"[OK] Summary written to: {out_path}")

    return report

# 便捷封装：按目录与文件名读取
def read_epoch_checkpoint_from(dir: str, filename: str, save_json: bool = True) -> Dict[str, Any]:
    return read_epoch_checkpoint(os.path.join(dir, filename), save_json=save_json)

if __name__ == "__main__":
    # 示例：
    # python inspect_checkpoint.py  # 自行修改下面的路径
    # topo_planner_pkg_v2/runs/20250827_113053/epoch_010_step190.pt
    # demo_path = "runs/20250917_165019/epoch_010_step190.pt"
    # demo_path = "runs/20250906_031621/epoch_001_step1439.pt"
    demo_path = "runs/20251128_111254/epoch_035_step4165.pt"
    # demo_path = "runs/20251022_014459/epoch_035_step4165.pt"
    # demo_path = "runs/20251009_132816/epoch_015_step1785.pt"
    if os.path.exists(demo_path):
        read_epoch_checkpoint(demo_path, save_json=True)
    else:
        print(f"Demo checkpoint not found: {demo_path}")
