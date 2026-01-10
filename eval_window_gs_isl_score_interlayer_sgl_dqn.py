# eval_window_gs_isl_score_interlayer_sgl_dqn.py
# DQN, Double DQN都可以用
# with robustness analysis/攻击实验与画图
# -*- coding: utf-8 -*-
"""
评估（含地面站优先级与连通性修复）：
- 支持采用 checkpoint 内保存的 config（use_ckpt_params=True），或使用当前 config.py
- 可用函数参数覆盖 t_begin/t_end/use_lookahead/window_w
- 逐步写出 eval_step.csv 与每步拓扑 JSON（Facility/ 与 Satellite/ 命名）

#  固定层间链路，只规划层间和星地链路
"""

import json, os, math
from typing import Set, Tuple, Dict, Any, Callable, Union, List, Optional
import csv

import networkx as nx
import re

import glob
import random
# import matplotlib
# matplotlib.use("Agg")  # 无显示环境也能出图
# import matplotlib.pyplot as plt


import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from config.config import cfg
from data.loader_adapter import load_everything, EdgeCand
from data.state import StateInterLayer, make_state_interlayer
from models.edge_q import EdgeDQNet
from topo.graph_utils import (
    C_LIGHT, largest_component_ratio, avg_hop_delay, switch_count, llh_to_ecef_distance_m, build_keep_graph, remove_E_core_graph
)
from topo.bmatch import bmatch_with_requirements_interlayer_sgl, safe_greedy_bmatch
from topo.conn_gain import conn_gain_A
from topo.intra_backbone import build_intra_backbone




# ------------ 配置应用/覆盖 ------------
def _apply_cfg_from_ckpt(ckpt_cfg: dict):
    for k, v in ckpt_cfg.items():
        setattr(cfg, k, v)

def _auto_device():
    dev = torch.device('cuda' if getattr(cfg, "USE_AUTO_DEVICE", True) and torch.cuda.is_available() else 'cpu')
    amp = bool(getattr(cfg, "AMP", True) and dev.type == 'cuda')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
    return dev, amp


# ------------ 与训练一致的特征 ------------

def edge_features(cands:List[EdgeCand], N:int,
                  deg_base:List[int], comp_id:List[int],
                  layer:Dict[int,str], plane:Dict[int,int],
                  prev_edges:Set[Tuple[int,int]]):
    feats=[]; pairs=[]
    layer_enc = {"LEO":0,"MEO":1,"GEO":2,"GS":3}
    for e in cands:
        u,v = (e.u, e.v) if e.u<e.v else (e.v, e.u)
        inv_delay = C_LIGHT / max(e.dist_m, 1.0)           # 逆度项，偏好把低度/边缘节点拉进主网，提升连通、降直径。
        rem_vis   = max(e.rem_vis_min, 0.0)
        cg = conn_gain_A(u,v,deg_base,comp_id,layer,plane, # 结构性增益，这条边能否显著增强全网连通/缩短路径”的结构性加分
                         b1=cfg.CG_B1,b2=cfg.CG_B2,b3=cfg.CG_B3,b4=cfg.CG_B4,eps=cfg.CG_EPS)
        is_new = 0.0 if (u,v) in prev_edges else 1.0
        feats.append([inv_delay, rem_vis, cg, is_new,
                      layer_enc.get(layer[u],0), layer_enc.get(layer[v],0)])
        pairs.append((u,v))

    feats = np.asarray(feats, dtype=np.float32)
    if getattr(cfg, "USE_EDGE_FEAT_W", False):
        w = np.asarray(getattr(cfg, "EDGE_FEAT_W", []), dtype=np.float32)
        # print(f"construct edge features, w={w}")
        if w.size > 0:
            if w.size < feats.shape[1]:
                # 长度不够时，后面维度默认权重=1
                w = np.pad(w, (0, feats.shape[1] - w.size), constant_values=1.0)
            elif w.size > feats.shape[1]:
                w = w[:feats.shape[1]]
            feats = feats * w[None, :]  # 逐维缩放
    return feats, pairs
    # return np.asarray(feats, dtype=np.float32), pairs

def make_max_deg(layer_map: Dict[int, str]) -> List[int]:
    out = []
    for i in range(len(layer_map)):
        L = layer_map[i]
        if   L == "LEO": out.append(cfg.MAX_DEG_LEO)
        elif L == "MEO": out.append(cfg.MAX_DEG_MEO)
        elif L == "GEO": out.append(cfg.MAX_DEG_GEO)
        elif L == "GS":  out.append(cfg.MAX_DEG_GS)
        else:            out.append(cfg.MAX_LINK)
    return out

def get_autocast(device='cuda', dtype=None):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):  # 新API
        return torch.amp.autocast(device, dtype=dtype)
    else:  # 旧API回退
        return torch.cuda.amp.autocast(dtype=dtype)

# ------------ 评估动作（含地面站/连通性修复） ------------
@torch.no_grad()
def _choose_action_from_state_eval(model: EdgeDQNet, device: torch.device, amp_enabled: bool,
                                   state: StateInterLayer):
    feats_np, pairs = edge_features(
        cands=state.cand_edges,
        N=len(state.nodes),
        deg_base=state.deg_base,
        comp_id=state.comp_id,
        layer=state.layer_map,
        plane=state.plane_map,
        prev_edges=state.E_prev
    )
    if feats_np.size == 0:
        return set(), {}, {}

    x = torch.tensor(feats_np, dtype=torch.float32, device=device)
    if amp_enabled:
        with torch.cuda.amp.autocast():
        # with get_autocast(device=device, dtype=(
        #         torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16)):
            q = model(x)
    else:
        q = model(x)
    q = torch.nan_to_num(q.view(-1), nan=0.0, posinf=0.0, neginf=0.0)

    scores = {pairs[i]: float(q[i].item()) for i in range(len(pairs))}

    # 提取 rem_vis（第2列，索引1），为每条边建立字典
    # feats_np 形状为 [E, D]；其中 rem_vis 在 feats_np[:, 1]
    # rem_vis = {pairs[i]: float(feats_np[i, 1]) for i in range(len(pairs))}
    rem_vis = {}
    for i, (a, b) in enumerate(pairs):
        u, v = (a, b) if a < b else (b, a)
        rem_vis[(u, v)] = float(feats_np[i, 1])  # 第2列是 rem_vis

    max_deg = make_max_deg(state.layer_map)
    max_repair_steps = int(getattr(cfg, "MAX_REPAIR_STEPS", 256))

    E_t = bmatch_with_requirements_interlayer_sgl(
        scores=scores,
        max_deg=max_deg,
        E_core=state.E_core,
        layer_map=state.layer_map,
        plane_map=state.plane_map,
        deg_isl_used=state.deg_isl_used,
        N=len(state.nodes),
        require_gs_per_layer=True,
        max_repair_steps=256,  # 可按需调整
    )

    if len(E_t) == 0:
        # 保底确保非空
        E_t = safe_greedy_bmatch(scores, max_deg, N_local=len(state.nodes))
    return E_t, scores, rem_vis


# ------------ 前瞻与距离 ------------
def _true_dist_map_for_t(db, t: int) -> Dict[Tuple[int, int], float]:
    nodes = db.nodes_by_t[t]; V = db.vis_matrix[t]; N = len(nodes)
    dist = {}
    for i in range(N):
        ni = nodes[i]
        for j in range(i + 1, N):
            if V[i, j] == 1:
                nj = db.nodes_by_t[t][j]
                d = llh_to_ecef_distance_m(
                    ni.lat, ni.lon, ni.alt_km * 1000.0,
                    nj.lat, nj.lon, nj.alt_km * 1000.0
                )
                dist[(i, j)] = d
    return dist

def _lookahead_score_true(E0, db, t0: int, window_w: int, max_deg_local: List[int]) -> float:
    t1 = min(t0 + window_w, db.time_steps - 1)
    E_prev = set(); E = set(E0); total = 0.0
    for tau, tt in enumerate(range(t0, t1 + 1)):
        V = db.vis_matrix[tt]
        E = {(u, v) for (u, v) in E if V[min(u, v), max(u, v)] == 1}
        dist_m = _true_dist_map_for_t(db, tt)
        N = len(db.nodes_by_t[tt])
        conn = largest_component_ratio(E, N)
        hop, delay = avg_hop_delay(E, N, dist_m, num_anchors=80)
        sw = switch_count(E, E_prev) if tau > 0 else 0
        # r = (cfg.R_A1_CONN * conn
        r = (cfg.R_A1_CONN * (conn-1)
             - cfg.R_A2_DELAY * delay
             - cfg.R_A3_HOP * hop
             - cfg.R_A4_SWITCH * sw)
        total += (cfg.GAMMA ** tau) * r
        E_prev = set(E)
    return total


# ------------ 拓扑保存（命名格式） ------------
def _node_label(local_idx: int, state: StateInterLayer, db) -> str:
    node = state.nodes[local_idx]  # Node(id=原始 id, layer=...)
    oid = int(node.id)
    layer = state.layer_map[local_idx]
    name = None
    if getattr(db, "sat_id_map", None) is not None:
        name = db.sat_id_map.get(oid, None)

    if layer == "GS":
        if name is None:
            name = f"GS{oid}"
        return f"Facility/{name}"
    else:
        if name is None:
            name = f"{layer}{oid}"
        return f"Satellite/{name}"

# 判断某节点是否为地面站（GS）
def _is_gs(local_idx: int, state: StateInterLayer) -> bool:
    return state.layer_map[local_idx] == "GS"



RemVisType = Union[Dict[Tuple[int, int], float], Callable[[int, int], float]]

def _get_rem_vis(u: int, v: int, rem_vis: RemVisType) -> Any:
    """返回 JSON 友好的 rem_vis 值（float 或 None）。"""
    try:
        if callable(rem_vis):
            val = float(rem_vis(u, v))
        else:
            # 统一用无向有序键查；若键方向不确定，双向兜底
            key = (min(u, v), max(u, v))
            if key in rem_vis:
                val = float(rem_vis[key])
            else:
                # 兼容调用方没有归一化键的情况
                val = float(rem_vis.get((u, v), rem_vis.get((v, u), float("nan"))))
    except Exception:
        val = float("nan")

    # JSON 不支持 NaN/Inf，转 None
    if math.isnan(val) or math.isinf(val):
        return None
    return val

def _save_dynamic_topology_json(E_t: Set[Tuple[int, int]],
                        state: StateInterLayer,
                        db,
                        rem_vis: RemVisType,
                        out_path: str):

    E_t_remove_core = remove_E_core_graph(E_core= state.E_core, E_t=E_t)
    # E_t_remove_core = E_t

    arr = []
    # 去重并规范化无向边
    for (u, v) in sorted({(min(a, b), max(a, b)) for (a, b) in E_t_remove_core}):
        edge_type = "0" if (_is_gs(u, state) or _is_gs(v, state)) else "1"
        edge_rem_vis = _get_rem_vis(u, v, rem_vis)
        # is_dynamic = bool(edge_rem_vis)
        arr.append({
            "src": _node_label(u, state, db),
            "dst": _node_label(v, state, db),
            "type": edge_type,
            # "dynamic": is_dynamic,
            "edge_rem_vis": edge_rem_vis
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # <—— 强制刷盘


def _save_static_topology_json(state: StateInterLayer,
                               db,
                               out_path: str):

    arr = []
    # 去重并规范化无向边
    for (u, v) in sorted({(min(a, b), max(a, b)) for (a, b) in state.E_core}):
        # edge_type = "0" if (_is_gs(u, state) or _is_gs(v, state)) else "1"
        edge_type =  "2"# 层内全是静态链路
        # is_dynamic = bool(False)
        arr.append({
            "src": _node_label(u, state, db),
            "dst": _node_label(v, state, db),
            "type": edge_type,
            # "dynamic": is_dynamic,
            "edge_rem_vis": "null"
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # <—— 强制刷盘





# ------------------ robustness等相关指标计算 ------------------

def robustness_curve(G, order, steps=120):
    """返回 (q, S)：移除比例 q 与最大连通分量占比 S(q)。"""
    Gc = G.copy()
    n = Gc.number_of_nodes()
    if n == 0: return [0.0], [0.0]
    step = max(1, math.ceil(n / steps))
    q, S = [], []
    removed = 0
    for _ in range(0, n + 1, step):
        if Gc.number_of_nodes() > 0:
            sizes = [len(c) for c in nx.connected_components(Gc)]
            s = (max(sizes) / n) if sizes else 0.0
        else:
            s = 0.0
        q.append(removed / n)
        S.append(s)
        if removed >= n: break
        batch = order[removed:removed + step]
        Gc.remove_nodes_from(batch)
        removed += len(batch)
    return q, S

def robustness_index(q, S):
    """曲线面积（梯形法，范围[0,1]）。"""
    area = 0.0
    for i in range(1, len(q)):
        dq = q[i] - q[i - 1]
        area += 0.5 * (S[i] + S[i - 1]) * dq
    return float(area)

def moving_average(y, k: int):
    """简单滑动平均；k<=1 时返回原序列。"""
    if k is None or k <= 1:
        return y
    y = np.asarray(y, dtype=float)
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(y, kernel, mode="same").tolist()

def compute_metrics_for_graph(G, steps=120, seed=42):
    """对单个无向图计算冗余/鲁棒性指标。"""
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = list(nx.connected_components(G))
    is_conn = (len(comps) == 1)
    giant = max(comps, key=len) if comps else set()
    # giant_frac = (len(giant) / n) if n else 0.0

    deg_vals = [d for _, d in G.degree()]
    delta = min(deg_vals) if deg_vals else 0
    avg_deg = (sum(deg_vals) / n) if n else 0.0
    if deg_vals:
        q25, q50, q75 = np.percentile(deg_vals, [25, 50, 75]).tolist()
    else:
        q25 = q50 = q75 = 0.0

    bridges = list(nx.bridges(G))
    arts = list(nx.articulation_points(G))
    bridge_frac = (len(bridges) / m) if m else 0.0
    art_frac = (len(arts) / n) if n else 0.0

    two_ec = list(nx.k_edge_components(G, k=2))
    largest_2ec_frac = (max((len(s) for s in two_ec), default=0) / n) if n else 0.0

    try:
        bcc_sets = list(nx.biconnected_components(G))  # 节点集合
    except Exception:
        bcc_sets = []
    largest_bcc_frac = (max((len(s) for s in bcc_sets), default=0) / n) if n else 0.0

    # 全局边连通度（无向全局最小割）
    if is_conn:
        try:
            from networkx.algorithms.connectivity import stoer_wagner
            lam_value, _ = stoer_wagner(G, weight=None)
            lam_algo = "Stoer–Wagner"
        except Exception:
            from networkx.algorithms.flow import shortest_augmenting_path
            lam_value = nx.edge_connectivity(G, flow_func=shortest_augmenting_path)
            lam_algo = "edge_connectivity(flow)"
    else:
        lam_value, lam_algo = 0, "n/a (disconnected)"

    # 代数连通度（需要 SciPy；环境无则返回 "N/A"）
    try:
        fiedler = float(nx.algebraic_connectivity(G)) if is_conn else 0.0
    except Exception:
        fiedler = "N/A"

    global_eff = float(nx.global_efficiency(G))
    avg_spl_giant = float(nx.average_shortest_path_length(G.subgraph(giant))) if len(giant) > 1 else 0.0

    # 鲁棒性曲线：随机 vs. 度攻击（静态）
    nodes = list(G.nodes())
    random.seed(seed)
    rand_order = nodes[:]; random.shuffle(rand_order)
    q_r, S_r = robustness_curve(G, rand_order, steps=steps)
    deg_map = dict(G.degree())
    deg_sorted = sorted(nodes, key=lambda x: deg_map[x], reverse=True)
    q_t, S_t = robustness_curve(G, deg_sorted, steps=steps)
    R_random = robustness_index(q_r, S_r)
    R_target = robustness_index(q_t, S_t)

    return {
        # "n_nodes": n, "n_edges": m,
        # "connected": bool(is_conn), "num_components": len(comps),
        # "giant_component_fraction": float(giant_frac),
        "min_degree": int(delta),   ## 最小度（度：与该节点连的边的数量）
        "avg_degree": float(avg_deg),  # 平均度
        "degree_q25": float(q25), "degree_q50": float(q50), "degree_q75": float(q75),  # 度分布，25/50/75 分位
        "edge_connectivity_lambda": int(lam_value) if isinstance(lam_value, (int, np.integer)) else lam_value,  # 全局边连通度lambda： λ(G)（无向图全局最小割，冗余性，最少删掉几条边，该网络变不连通）
        "edge_connectivity_algo": lam_algo,  # 使用的算法（Stoer–Wagner 算法、edge_connectivity(flow)、 n/a (disconnected)）
        "bridges_fraction": float(bridge_frac),  # 桥占比（0~1），越低越好（=0（无桥）→ 任意一条边坏了都不会分裂）
        "articulation_fraction": float(art_frac),   # 关节点占比（0~1），越低越好
        "largest_2edge_component_fraction": float(largest_2ec_frac),  # 最大 2-边连通块占比（0~1）（= 1.0 → 全图对“单边失效”鲁棒）
        "largest_biconnected_component_fraction": float(largest_bcc_frac),   #最大双连通块占比（0~1）（ > 96% → 至少约 96% 节点对“单点失效”鲁棒）
        "algebraic_connectivity": fiedler,  # 代数连通度
        "global_efficiency": global_eff,  # 全局效率
        "avg_shortest_path_on_giant": avg_spl_giant,  # 最大分量平均最短路
        "robustness_random_R": float(R_random),  # 鲁棒性面积 R（0~1）：随机失效
        "robustness_targeted_by_degree_R": float(R_target),  # 鲁棒性面积 R（0~1）：按度攻击（静态顺序）
    }, (q_r, S_r, q_t, S_t)

# def plot_robustness_curves(q_r, S_r, q_t, S_t, title, out_png, smooth_window=1):
#     """画两条鲁棒性曲线（可选滑动平均），单图无子图、不指定颜色。"""
#     def _ma(y, k):
#         return y if k <= 1 else moving_average(y, k)
#     S_r_sm = _ma(S_r, smooth_window)
#     S_t_sm = _ma(S_t, smooth_window)
#     plt.figure()
#     plt.plot(q_r, S_r_sm, label="Random failure")
#     plt.plot(q_t, S_t_sm, label="Degree-targeted (static)")
#     plt.xlabel("Removed fraction q")
#     plt.ylabel("Largest component fraction S(q)")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=150)
#     plt.close()


# --- 1) 把 E_sel 变成无向图（确保把“孤立点”也放进去） ---
def nx_from_edge_set(E_t, N=None, nodes=None):
    """
    E_sel: 形如 {(u,v), ...} 的无向边集合
    N:     若节点是 0..N-1，传 N 可保证度为 0 的节点也在图里
    nodes: 或者直接传一个节点全集（如 layer_map.keys()）
    """
    G = nx.Graph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    elif N is not None:
        G.add_nodes_from(range(N))      # 把孤立点也纳入
    else:
        # 从边推断节点全集（注意：没有出现在边上的节点会丢失）
        inferred = set()
        for u, v in E_t:
            inferred.add(u); inferred.add(v)
        G.add_nodes_from(inferred)

    # 去掉自环
    G.add_edges_from([(u, v) for (u, v) in E_t if u != v])
    return G

# --- 2) 计算指标与曲线（复用前面提供的函数） ---
# 已定义了这两个函数：
#   compute_metrics_for_graph(G, steps=120, seed=42)
#   plot_robustness_curves(q_r, S_r, q_t, S_t, title, out_png, smooth_window=1)

def compute_robustness_metrics_from_E_t(E_t, N=None, nodes=None, steps=120, seed=42, png_path=None):
    G = nx_from_edge_set(E_t, N=N, nodes=nodes)
    metrics, (q_r, S_r, q_t, S_t) = compute_metrics_for_graph(G, steps=steps, seed=seed)
    # if png_path is not None:
    #     plot_robustness_curves(q_r, S_r, q_t, S_t, title="planned_topology", out_png=png_path, smooth_window=1)
    return metrics, (q_r, S_r, q_t, S_t)

# --- 3) 用你的 E_t 结果直接评估 ---
# metrics, curves = compute_robustness_metrics_from_E_t(E_t, N=N, png_path="./bmatch_topology_robustness.png")
# print(metrics)



# ------------------ 评估主函数 ------------------
def eval_model(dir: str,
               name: str,
               t_begin: Optional[int] = None,
               t_end: Optional[int] = None,
               use_lookahead: Optional[bool] = None,
               window_w: Optional[int] = None,
               use_ckpt_params: bool = True,

               MAX_DEG_ISL_LEO: int = None,
               MAX_DEG_ISL_MEO: int = None,
               MAX_DEG_ISL_GEO: int = None,
               MAX_DEG_SGL_LEO: int = None,
               MAX_DEG_SGL_MEO: int = None,
               MAX_DEG_SGL_GEO: int = None,
               MAX_DEG_SGL_GS: int = None,
               REQ_SGL_LEO: int = None,
               REQ_SGL_MEO: int = None,
               REQ_SGL_GEO: int = None) -> str:
    """
    在目录 dir 下，用 epoch checkpoint 文件 name 进行评估。

    参数
    ----
    dir : str
        模型目录（包含 checkpoint 文件）
    name : str
        checkpoint 文件名，例如 'epoch_010_step2380.pt'
    t_begin : Optional[int]
        评估起始时刻（分钟）。不传则用 cfg.DATA_T_START（或默认 0）
    t_end : Optional[int]
        评估结束时刻（不含）。不传则用 cfg.DATA_T_END（或默认 120）
    use_lookahead : Optional[bool]
        是否启用动作滑窗前瞻。优先于 cfg.USE_LOOKAHEAD
    window_w : Optional[int]
        前瞻窗口大小（分钟）。优先于 cfg.WINDOW_W
    use_ckpt_params : bool
        若 True 且 checkpoint 中包含 'config'，则采用其中的配置；否则使用当前 config.py

    返回
    ----
    out_dir : str
        评估输出目录路径：{dir}/eval_<name_without_ext>
    """
    ckpt_path = os.path.join(dir, name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device, amp_enabled = _auto_device()

    # 1) 载入 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # 2) 采用 checkpoint 中的 config（可选）
    if use_ckpt_params and isinstance(ckpt, dict) and ("config" in ckpt):
        _apply_cfg_from_ckpt(ckpt["config"])

    # 3) 覆盖函数参数到 cfg
    if t_begin is not None: setattr(cfg, "DATA_T_START", int(t_begin))
    if t_end  is not None: setattr(cfg, "DATA_T_END",   int(t_end))
    if use_lookahead is not None: setattr(cfg, "USE_LOOKAHEAD", bool(use_lookahead))
    if window_w is not None:      setattr(cfg, "WINDOW_W",      int(window_w))

    if MAX_DEG_ISL_LEO is not None: setattr(cfg, "MAX_DEG_ISL_LEO", int(MAX_DEG_ISL_LEO))
    if MAX_DEG_ISL_MEO is not None: setattr(cfg, "MAX_DEG_ISL_MEO", int(MAX_DEG_ISL_MEO))
    if MAX_DEG_ISL_GEO is not None: setattr(cfg, "MAX_DEG_ISL_GEO", int(MAX_DEG_ISL_GEO))
    if MAX_DEG_SGL_LEO is not None: setattr(cfg, "MAX_DEG_SGL_LEO", int(MAX_DEG_SGL_LEO))
    if MAX_DEG_SGL_MEO is not None: setattr(cfg, "MAX_DEG_SGL_MEO", int(MAX_DEG_SGL_MEO))
    if MAX_DEG_SGL_GEO is not None: setattr(cfg, "MAX_DEG_SGL_GEO", int(MAX_DEG_SGL_GEO))
    if MAX_DEG_SGL_GS is not None: setattr(cfg, "MAX_DEG_SGL_GS", int(MAX_DEG_SGL_GS))
    if REQ_SGL_LEO is not None: setattr(cfg, "REQ_SGL_LEO", int(REQ_SGL_LEO))
    if REQ_SGL_MEO is not None: setattr(cfg, "REQ_SGL_MEO", int(REQ_SGL_MEO))
    if REQ_SGL_GEO is not None: setattr(cfg, "REQ_SGL_GEO", int(REQ_SGL_GEO))

    # 4) 数据加载
    db = load_everything()

    # 5) 构建模型并加载权重
    edge_dim = 6
    hidden = int(getattr(cfg, "HIDDEN_DIM", 256))
    model_net = EdgeDQNet(edge_dim=edge_dim, hidden=hidden).to(device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_net.load_state_dict(ckpt["model"])
    else:
        model_net.load_state_dict(ckpt)   # 兼容直接保存 state_dict 的情况
    model_net.eval()

    # 6) 评估区间
    T_all = db.time_steps
    tb = int(getattr(cfg, "DATA_T_START", 0)) if t_begin is None else int(t_begin)
    te = int(min(getattr(cfg, "DATA_T_END", 120) if t_end is None else int(t_end), T_all))
    assert te - tb >= 1, "评估时间区间过短"

    # 7) 输出目录
    stem = os.path.splitext(os.path.basename(name))[0]
    # out_dir = os.path.join(dir, f"eval_{stem}")
    out_dir = os.path.join(dir, f"eval_{stem}_{t_end}")
    topo_dir = os.path.join(out_dir, "steps")
    os.makedirs(topo_dir, exist_ok=True)

    # metrics CSV（覆盖写）
    csv_path = os.path.join(out_dir, "eval_step.csv")
    # with open(csv_path, "w", encoding="utf-8") as f:
    #     # f.write("t,num_edges,conn,hop,delay_s,switch,reward\n")
    #     f.write("t,num_edges,conn,hop,delay_s,switch,reward\n")

    # ===== 把“已有的列”与“metrics”合并写入同一 CSV（若不存在就写表头） =====
    BASE_COLS = ["t", "num_edges", "conn", "hop", "delay_s", "switch", "reward"]
    METRIC_COLS = [
        # "n_nodes", "n_edges_calc", "connected", "giant_component_fraction",
        "min_degree", "avg_degree", "degree_q25", "degree_q50", "degree_q75",
        "edge_connectivity_lambda", "edge_connectivity_algo","bridges_fraction",
        "articulation_fraction","largest_2edge_component_fraction", "largest_biconnected_component_fraction",
        "algebraic_connectivity", "global_efficiency", "avg_shortest_path_on_giant",
        "robustness_random_R", "robustness_targeted_by_degree_R"
    ]


    ALL_COLS = BASE_COLS + METRIC_COLS


    # 8) 逐步评估
    E_prev: Set[Tuple[int, int]] = set()
    E_core = build_intra_backbone(nodes=db.nodes_by_t[tb],
                                  layer_map=db.layer_map,
                                  plane_map=db.plane_map)
    s_t = make_state_interlayer(db, tb, E_prev=set(), E_core=set(E_core))

    use_la = bool(getattr(cfg, "USE_LOOKAHEAD", False)) if use_lookahead is None else bool(use_lookahead)
    W = int(getattr(cfg, "WINDOW_W", 0)) if window_w is None else int(window_w)

    _save_static_topology_json(state=s_t, db=db, out_path=os.path.join(topo_dir, f"edges_static.json"))

    for t in tqdm(range(tb, te), desc=f"eval {stem}"):
        # 基于 Q 的匹配（含地面站与连通性修复）
        E_t_q, _, rem_vis = _choose_action_from_state_eval(model_net, device, amp_enabled, s_t)

        # 可选：动作前瞻
        if use_la and W > 0:
            candidates = [("q", E_t_q)]
            N = len(db.nodes_by_t)
            E_keep = build_keep_graph(N=N,
                                      E_core=E_core,
                                      E_prev=E_prev,
                                      visible=s_t.visible_pairs)
            # E_keep_ = {(min(u, v), max(u, v)) for (u, v) in s_t.E_prev
            #           if (min(u, v), max(u, v)) in s_t.visible_pairs}
            if E_keep:
                candidates.append(("keep", E_keep))

            max_deg_local = make_max_deg(s_t.layer_map)
            scored = []
            for _, E0 in candidates:
                score = _lookahead_score_true(E0, db, s_t.t, W, max_deg_local)
                scored.append((score, E0))
            scored.sort(key=lambda x: x[0], reverse=True)
            E_t = scored[0][1]
            if not E_t:
                # 保底确保非空
                simple_scores = {(min(e.u, e.v), max(e.u, e.v)): 1.0 / max(e.dist_m, 1.0) for e in s_t.cand_edges}
                E_t = safe_greedy_bmatch(simple_scores, make_max_deg(s_t.layer_map), len(s_t.nodes))
        else:
            E_t = E_t_q

        # 指标
        dist_m_t = _true_dist_map_for_t(db, t)
        conn = largest_component_ratio(E_t, len(s_t.nodes))
        hop, delay = avg_hop_delay(E_t, len(s_t.nodes), dist_m_t, num_anchors=80)
        sw = switch_count(E_t, E_prev) if (t > tb) else 0
        # r = (cfg.R_A1_CONN * conn
        r = (cfg.R_A1_CONN * (conn-1)
             - cfg.R_A2_DELAY * delay
             - cfg.R_A3_HOP * hop
             - cfg.R_A4_SWITCH * sw)
        # robustness metrics
        # 用 E_t 结果直接评估 ---
        remove_curve_dir = os.path.join(topo_dir, "metrics/remove_curve_plots")# eval_epoch_xxx_stepxxx/steps/metrics/remove_curve_plots
        os.makedirs(remove_curve_dir, exist_ok=True)
        png_path = os.path.join(remove_curve_dir, f"random_attack_{t}.png")# eval_epoch_xxx_stepxxx/steps/metrics/remove_curve_plots/random_attack_{t}.png
        metrics, curves = compute_robustness_metrics_from_E_t(E_t, N=None, png_path=png_path)
        # print(metrics)

        # 2) 拼成一行
        row = {
            "t": t,
            "num_edges": len(E_t),
            "conn": float(conn),
            "hop": float(hop),
            "delay_s": float(delay),
            "switch": int(sw),
            "reward": float(r),
        }
        row.update(metrics)

        # 3) # 记录 CSV,若文件不存在则写表头；存在就直接追加
        file_exists = os.path.isfile(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ALL_COLS)
            if not file_exists:
                w.writeheader()
            w.writerow(row)

        # # 记录 CSV
        # with open(csv_path, "a", encoding="utf-8") as f:
        #     f.write(f"{t},{len(E_t)},{conn:.6f},{hop:.6f},{delay:.6f},{sw},{r:.6f}\n")

        # 保存拓扑 JSON

        _save_dynamic_topology_json(E_t, s_t, db, rem_vis, os.path.join(topo_dir, f"edges_t{t:05d}.json"))

        # 滚动
        s_t = (make_state_interlayer(db, t + 1, E_t, E_core=set(E_core)))
        E_prev = E_t

    print(f"[EVAL DONE] metrics: {csv_path}")
    print(f"[EVAL DONE] topologies: {topo_dir}")
    return out_dir


# 可选命令行入口（便于单脚本跑）
if __name__ == "__main__":
    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--dir", type=str, required=True)
    # ap.add_argument("--name", type=str, required=True)
    # ap.add_argument("--t_begin", type=int, default=None)
    # ap.add_argument("--t_end", type=int, default=None)
    # ap.add_argument("--use_lookahead", type=int, default=None)  # 1/0
    # ap.add_argument("--window_w", type=int, default=None)
    # ap.add_argument("--use_ckpt_params", type=int, default=1)         # 1: use ckpt config
    # args = ap.parse_args()

    # eval_model(
    #     dir=args.dir,
    #     name=args.name,
    #     t_begin=args.t_begin,
    #     t_end=args.t_end,
    #     use_lookahead=(None if args.use_lookahead is None else bool(args.use_lookahead)),
    #     window_w=args.window_w,
    #     use_ckpt_params=bool(args.use_ckpt_params),
    # )

    start_time = datetime.now()

    out = eval_model(
        # dir="./runs/20250904_211256",#因为度限制leo=3，大多数都不连通,ckpt=348_9_111
        # dir="./runs/20250829_003834",# 因为度限制leo=4，有不连通
        # dir="./runs/20250829_003244",#ckpt=568_9_111
        # dir="runs/20251013_145146", # DQN
        # dir="runs/20251013_195526", # DDQN
        # dir="runs/20251018_164658", # LMSR-DDQN
        dir="runs/20251106_204423", # fw=FALSE,DQN
        # name="epoch_050_step950.pt",
        name="epoch_035_step4165.pt",
        t_begin=0,  # 不传则用 ckpt/config

        t_end=5,  # 不传则用 ckpt/config
        # use_lookahead=True,  # 不传则用 ckpt/config
        # window_w=3,  # 不传则用 ckpt/config
        use_ckpt_params=True  # 采用 checkpoint 中保存的 config
        ,#下面的变量若不需要替换ckpt/config，注释掉即可
        # 星间度约束
        MAX_DEG_ISL_LEO = 50,  # 5   LEO层卫星星间天线数量（大于等于5，静态链路已经占用了4个）
        MAX_DEG_ISL_MEO = 60,  # 6   MEO层卫星星间天线数量（大于等于5，静态链路已经占用了4个）
        MAX_DEG_ISL_GEO = 80,  # 8   GEO层卫星星间天线数量（大于等于3，静态链路已经占用了2个）

        #  星地度约束，SGL 在 卫星端/地面端占用 GSL 预算
        MAX_DEG_SGL_LEO = 30,  # 3   LEO层卫星星地天线数量
        MAX_DEG_SGL_MEO = 30,  # 3  LEO层卫星星地天线数量
        MAX_DEG_SGL_GEO = 30,  # 3  LEO层卫星星地天线数量
        MAX_DEG_SGL_GS = 30,  # 3  # 地面站星地天线数量  （每个地面站最多并发 SGL 条数）

        # 整个地面站在星地链路(SGL)上的“分层保底”配额，修改于20250930
        REQ_SGL_LEO = 3,  # 1  地面站层对LEO层卫星保底天线数量,大于等于1
        REQ_SGL_MEO = 3,  # 1  地面站层对MEO层卫星保底天线数量,大于等于1
        REQ_SGL_GEO = 3   # 1  地面站层对GEO层卫星保底天线数量,大于等于1

    )
    end_time = datetime.now()
    cost = end_time - start_time  # timedelta
    print(f"eval cost time:{cost.total_seconds():.3f}s")
    print("eval results at:", out)




