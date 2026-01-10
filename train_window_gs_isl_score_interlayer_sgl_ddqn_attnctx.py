# TODO:attn-DDQN (Double attention DQN)
#  Double: train_step里，online 选动作，target 估值
#  attnctx DQN:
#  确保地面站与卫星能连上，
#  确保地面站的度约束与星间度约束分开，
#  星地链路占用地面站的度约束！！和卫星的度约束
#  边特征打分修改权重！def edge_features
#  固定层间链路，只规划层间和星地链路
#  星地链路按地面站层与LEO/MEO/GEO连，而不是单个的地面站分别与EdgeQAttnCtx


# -*- coding: utf-8 -*-
"""
模式 A 训练主程序：
- 在线动作：可选“滑窗前瞻”对多个候选拓扑打分，仅执行当下最优
- 训练：当步奖励 + n-step 目标；目标动作用 b-匹配在 s_{t+n} 上求
- 自动自适应 CUDA/AMP；定期 CSV 落盘 & flush；定期保存模型
"""
from dataclasses import asdict
import json

import os, random
from datetime import datetime
from typing import Dict, Tuple, Set, List
import numpy as np
import torch
import torch.nn.functional as F

from config.config import cfg
from utils.seed import set_seed
from utils.logger import MetricLogger

from data.loader_adapter import load_everything, build_edge_candidates_at_t, EdgeCand
from topo.conn_gain import conn_gain_A
from topo.graph_utils import (C_LIGHT, build_keep_graph, remove_E_core_graph, largest_component_ratio,
                              avg_hop_delay, switch_count, llh_to_ecef_distance_m)
from topo.bmatch import bmatch_greedy, two_opt_placeholder, safe_greedy_bmatch
from models.edge_q import EdgeQAttnCtx
from rl.replay import ReplaySeq, Transition
from data.state import StateInterLayer, make_state_interlayer
import copy
from tqdm import tqdm
from planner.rollout import fast_rollout_score
from topo.bmatch import bmatch_with_requirements_interlayer_sgl
from topo.intra_backbone import build_intra_backbone



def _auto_device():
    dev = torch.device('cuda' if (cfg.USE_AUTO_DEVICE and torch.cuda.is_available()) else 'cpu')
    amp = cfg.AMP and (dev.type=='cuda')
    if dev.type=='cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
    return dev, amp

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

def make_max_deg(layer_map:Dict[int,str]) -> List[int]:
    out=[]
    for i in range(len(layer_map)):
        L = layer_map[i]
        if   L=="LEO": out.append(cfg.MAX_DEG_LEO)
        elif L=="MEO": out.append(cfg.MAX_DEG_MEO)
        elif L=="GEO": out.append(cfg.MAX_DEG_GEO)
        elif L=="GS":  out.append(cfg.MAX_DEG_GS)
        else: out.append(cfg.MAX_LINK)
    # print("out:",out)

    return out

def get_autocast(device, dtype=None):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):  # 新API
        return torch.amp.autocast(device, dtype=dtype)
    else:  # 旧API回退
        return torch.cuda.amp.autocast(dtype=dtype)

def make_scaler(enabled=True):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device='cuda', enabled=enabled)
    else:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def choose_action_from_state(model: EdgeQAttnCtx, device: torch.device, amp_enabled: bool,
                             state: StateInterLayer):
    """
    基于 State 进行动作选择：
    1) GNN/MLP 对候选边打分
    2) b-匹配选边
    3) 空解兜底（很关键）
    4) 2-opt 细化（仅在非空时）
    """
    # 1) 构特征
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
        if not hasattr(choose_action_from_state, "_warn0"):
            print("[warn] edge_features empty; return ∅.")  # 只打印一次
            choose_action_from_state._warn0 = True
        return set(), {}, {}

    # 2) 前向得到每条边的 Q 分数
    x = torch.tensor(feats_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        if amp_enabled:
            with torch.cuda.amp.autocast():
            # with get_autocast(device=device, dtype=(
            # torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16)):
                q = model(x)
        else:
            q = model(x)

    # 形状/数值保护
    q = q.view(-1)  # 确保是一维 [E]
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    # 分数字典
    scores = {pairs[i]: float(q[i].item()) for i in range(len(pairs))}
    # 提取 rem_vis（第2列，索引1），为每条边建立字典
    rem_vis = {}
    # for i, (a, b) in enumerate(pairs):
    #     u, v = (a, b) if a < b else (b, a)
    #     rem_vis[(u, v)] = float(feats_np[i, 1])  # 第2列是 rem_vis

    # 3) b-匹配 + 空解兜底
    max_deg = make_max_deg(state.layer_map)           # 本地索引的度上限
    # E_t = bmatch_greedy(scores, max_deg)
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
    # E_t = bmatch_with_dual_budgets(
    #     scores=scores,
    #     nodes=state.nodes,
    #     layer_map=state.layer_map,
    #     deg_isl_used=state.deg_isl_used,  # 由骨干占用
    #     require_gs_per_layer=True,
    #     max_repair_steps=256
    # )

    # 再做 2-opt 细化（非必需）
    # E_t = two_opt_placeholder(E_t, scores, max_deg)

    if len(E_t) == 0:
        print("len(E_t)=0")
        # # —— 关键兜底：按分数从高到低、仅用度约束挑边，保证非空 ——
        # # 若你已实现 safe_greedy_bmatch 放在 topo/bmatch.py，直接 import 用
        # try:
        #     from topo.bmatch import safe_greedy_bmatch_typed
        # except Exception:
        #     # 就地给个极简兜底（防万一）
        #     def safe_greedy_bmatch(scores_: dict, max_deg_: list, N_local: int):
        #         deg = [0]*N_local; E = set()
        #         for (u,v), s in sorted(scores_.items(), key=lambda kv: kv[1], reverse=True):
        #             if 0 <= u < N_local and 0 <= v < N_local and deg[u] < max_deg_[u] and deg[v] < max_deg_[v]:
        #                 E.add((u,v)); deg[u]+=1; deg[v]+=1
        #         return E
        #
        # E_t = safe_greedy_bmatch(scores, max_deg, N_local=len(state.nodes))
        # if not hasattr(choose_action_from_state, "_fallback_print"):
        #     print(f"[fallback@online] bmatch_greedy empty → safe_greedy picked {len(E_t)} edges")
        #     choose_action_from_state._fallback_print = True

    # # 4) 2-opt 仅在非空时运行
    # if len(E_t) > 0:
    #     E_t = two_opt_placeholder(E_t, scores, max_deg)

    return E_t, scores, rem_vis


def compute_step_metrics(E_t:Set[Tuple[int,int]], E_tm1:Set[Tuple[int,int]],
                         db, t:int, dist_m:Dict[Tuple[int,int],float]):
    N = len(db.nodes_by_t[t])
    conn = largest_component_ratio(E_t, N)
    hop, delay = avg_hop_delay(E_t, N, dist_m, num_anchors=80)
    sw = switch_count(E_t, E_tm1) if t>0 else 0
    r = (cfg.R_A1_CONN*(conn-1.0)
    # r = (cfg.R_A1_CONN*conn
         - cfg.R_A2_DELAY*delay
         - cfg.R_A3_HOP*hop
         - cfg.R_A4_SWITCH*sw)
    return conn, delay, hop, sw, r

# —— 每个时刻 t 的距离（仅对当时可视边计算）——
def build_true_dist_map_for_t(db, t: int):
    nodes = db.nodes_by_t[t]
    V = db.vis_matrix[t]   # [N,N], 0/1
    N = len(nodes)
    dist = {}
    for i in range(N):
        ni = nodes[i]
        for j in range(i+1, N):
            if V[i, j] == 1:
                nj = nodes[j]
                d = llh_to_ecef_distance_m(
                    ni.lat, ni.lon, ni.alt_km * 1000.0,
                    nj.lat, nj.lon, nj.alt_km * 1000.0
                )
                dist[(i, j)] = d
    return dist

# —— 滑窗前瞻打分（不改训练目标，只影响动作选择）——
def lookahead_score_true(E0, db, t0: int, window_w: int, max_deg_local):
    t1 = min(t0 + window_w, db.time_steps - 1)
    E_prev = set()
    E = set(E0)
    total = 0.0
    for tau, tt in enumerate(range(t0, t1 + 1)):
        V = db.vis_matrix[tt]
        # 1) 清除不可见边
        E = {(u, v) for (u, v) in E if V[min(u, v), max(u, v)] == 1}
        # 2) 本分钟距离
        dist_m = build_true_dist_map_for_t(db, tt)
        # 3) 指标 + 折扣累计
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

def main():
    s = json.dumps(asdict(cfg), indent=2, ensure_ascii=False)  # 不会自动排序
    print("\n[config snapshot]")
    print(s)

    start_time = datetime.now()

    # ——— 初始化 ———
    set_seed(cfg.SEED)
    device, amp_enabled = _auto_device()
    print(f"[INFO] device={device}, amp={amp_enabled}")
    print("[INFO] cuda available:", torch.cuda.is_available(), "device:", torch.cuda.current_device(),
          torch.cuda.get_device_name(0))

    run_name = cfg.RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.LOG_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = MetricLogger(run_dir)

    db = load_everything()

    EDGE_DIM = 6
    model = EdgeQAttnCtx(edge_dim=EDGE_DIM, hidden=cfg.HIDDEN_DIM).to(device)
    target = EdgeQAttnCtx(edge_dim=EDGE_DIM, hidden=cfg.HIDDEN_DIM).to(device)
    target.load_state_dict(model.state_dict())
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    # scaler = make_scaler(enabled=amp_enabled)
    replay = ReplaySeq(capacity=1_000_000)

    # ——— 训练范围 & 轮次设置（固定前120分钟，多轮）———
    T_all = db.time_steps
    # 若 config 里没有就用默认：0~120
    t_begin = getattr(cfg, "DATA_T_START", 0)
    t_end   = min(getattr(cfg, "DATA_T_END", 120), T_all)   # 不含 t_end
    assert t_end - t_begin >= 2, "训练区间太短（至少需要两个时刻）"

    EPOCHS = getattr(cfg, "EPOCHS", 10)   # 多轮重复训练
    total_steps_cap = cfg.NUM_STEPS       # 全局步数上限（两者共同限制）

    global_step = 0
    epsilon = cfg.EPS_START

    for epoch in range(EPOCHS):
        if global_step >= total_steps_cap:
            break

        # 每轮从区间起点重新开始，不把上轮拓扑带入下一轮
        E_prev = set()
        E_core = build_intra_backbone(nodes=db.nodes_by_t[t_begin],
                                   layer_map=db.layer_map,
                                   plane_map=db.plane_map)
        # s_t = make_state(db, t_begin, E_prev, E_core=set())
        # s_t = make_state_interlayer(db, t_begin, E_prev, E_core=set())
        s_t = make_state_interlayer(db, t_begin, E_prev, E_core=set(E_core))


        # （可选）每轮记录累计奖励
        epoch_total_r = 0.0

        pbar = tqdm(range(t_begin, t_end - 1), desc=f'epoch {epoch+1}/{EPOCHS}')
        for t in pbar:

            num_cand_to_gs = sum(1 for e in s_t.cand_edges
                                 if s_t.layer_map[e.u] == "GS" or s_t.layer_map[e.v] == "GS")
            if t % 100 == 0:
                print(f"[check] t={t} cand_to_GS={num_cand_to_gs}/{len(s_t.cand_edges)}")

            if global_step >= total_steps_cap:
                break

            # ——— 1) 先得到基线动作 E_t_q（ε-greedy）———
            if random.random() < epsilon:
                # 随机在当前可视候选边上打分 + b-匹配（使用本地索引的 layer_map）
                rnd_scores = {(min(e.u, e.v), max(e.u, e.v)): random.random()
                              for e in s_t.cand_edges}
                max_deg_local = make_max_deg(s_t.layer_map)
                # E_t_q = bmatch_greedy(rnd_scores, max_deg_local)

                E_t_q = bmatch_with_requirements_interlayer_sgl(
                    scores=rnd_scores,
                    max_deg=max_deg_local,
                    E_core = s_t.E_core,
                    layer_map=s_t.layer_map,
                    plane_map=s_t.plane_map,
                    deg_isl_used=s_t.deg_isl_used,
                    N=len(s_t.nodes),
                    require_gs_per_layer=True,
                    max_repair_steps=256,  # 可按需调整
                )

                # E_t_q = bmatch_with_dual_budgets(
                #     scores=rnd_scores,
                #     nodes=s_t.nodes,
                #     layer_map=s_t.layer_map,
                #     deg_isl_used=s_t.deg_isl_used,  # # 由骨干+上一时刻仍可用边占用
                #     require_gs_per_layer=True,
                #     max_repair_steps=256
                # )

                if len(E_t_q) == 0:
                    E_t_q = safe_greedy_bmatch(rnd_scores, max_deg_local, N_local=len(s_t.nodes))
                    print(f"[fallback] bmatch empty → safe_greedy selected {len(E_t_q)} edges.")
            else:
                # Q 打分 + b-匹配
                E_t_q, _, _ = choose_action_from_state(model, device, amp_enabled, s_t)
                if len(E_t_q) == 0:
                    # 直接对 choose_action_from_state 里那批 pairs/scores 再跑一次兜底
                    # 或者重新算一次 scores（和探索分支同样做法）
                    from topo.bmatch import safe_greedy_bmatch
                    # 你如果拿不到 scores，就退回到 rnd_scores 也行，先跑通
                    rnd_scores = {(min(e.u, e.v), max(e.u, e.v)): 1.0 for e in s_t.cand_edges}  # 全 1 也能选出边
                    E_t_q = safe_greedy_bmatch(rnd_scores, make_max_deg(s_t.layer_map), N_local=len(s_t.nodes))
                    print(f"[fallback] Q-branch empty → safe_greedy selected {len(E_t_q)} edges.")

            # ——— 2) （可选）滑窗前瞻重排序———
            if getattr(cfg, "USE_LOOKAHEAD", False) and cfg.WINDOW_W > 0:##当属性不存在时返回的False，如果属性存在返回该属性
                candidates = [("q", E_t_q)]

                # 尽量保留上一步仍可见的边，降低切换
                # E_keep = {(min(u, v), max(u, v)) for (u, v) in s_t.E_prev
                #           if (min(u, v), max(u, v)) in s_t.visible_pairs}
                N = len(db.nodes_by_t)
                E_keep = build_keep_graph(N=N,
                                          E_core=E_core,
                                          E_prev=E_prev,
                                          visible=s_t.visible_pairs)
                if E_keep:
                    candidates.append(("keep", E_keep))

                max_deg_local = make_max_deg(s_t.layer_map)
                scored = []
                for tag, E0 in candidates:
                    score = lookahead_score_true(E0, db, s_t.t, cfg.WINDOW_W, max_deg_local)
                    scored.append((score, tag, E0))
                scored.sort(reverse=True)
                _, best_tag, E_t = scored[0]

                if not E_t:
                    # 用简单分数填充，确保非空
                    simple_scores = {(min(e.u, e.v), max(e.u, e.v)): 1.0 / max(e.dist_m, 1.0) for e in s_t.cand_edges}
                    E_t = safe_greedy_bmatch(simple_scores, make_max_deg(s_t.layer_map), len(s_t.nodes))
                    print(f"[force] t={t}: final empty → filled {len(E_t)} edges.")

            else:
                E_t = E_t_q

            # # === Sanity check ===
            # V_t = db.vis_matrix[t]
            # num_vis_pairs = int(V_t.sum() // 2)  # 可视对数量
            # num_cand = len(s_t.cand_edges)  # 候选边数量
            # num_bm = len(E_t)  # b-匹配选中的边数
            # N_local = len(s_t.nodes)
            # bad_idx = sum(1 for (u, v) in E_t if not (0 <= u < N_local and 0 <= v < N_local))
            #
            # if (t < 5) or (t % 20 == 0):
            #     print(f"[sanity] t={t} N={N_local} vis_pairs={num_vis_pairs} "
            #           f"cand={num_cand} bmatch={num_bm} bad_idx={bad_idx} eps={epsilon:.3f}")

            # ——— 3) 环境反馈———
            dist_m_t = build_true_dist_map_for_t(db, t)
            conn, delay, hop, sw, r = compute_step_metrics(E_t, E_prev, db, t, dist_m_t)
            epoch_total_r += r

            # ——— 4) 写入回放（深拷贝，避免后续修改污染样本）———
            # s_next = make_state(db, t + 1, E_t, E_core=set())
            s_next = make_state_interlayer(db, t + 1, E_t, E_core=set(E_core))
            replay.add(Transition(s=copy.deepcopy(s_t),
                                  E_t=set(E_t),
                                  r=r,
                                  s_next=copy.deepcopy(s_next),
                                  done=False))

            # ——— 5) 训练更新触发：基于回放池门槛 + UPDATE_EVERY + WARMUP ———
            did_update = False
            need = cfg.BATCH_SIZE * (cfg.N_STEP + 1)  # 至少能采样一个 n-step 批次
            if (len(replay.buf) >= need) and (global_step % cfg.UPDATE_EVERY == 0) and (global_step >= cfg.WARMUP_STEPS):
                batch = replay.sample_sequences(cfg.BATCH_SIZE, cfg.N_STEP)
                loss = train_step(model, target, optim, scaler, device, amp_enabled,
                                  batch=batch, replay=replay, db=db)
                did_update = True

            # ——— 6) 日志（仅在更新时写 loss，避免“看起来一直是0”的错觉）———
            logger.log_step(global_step, conn, delay, hop, sw,
                            loss=(float(loss) if did_update else None),
                            reward=r)


            # ——— 7) 软更新 & ε退火 & 周期保存 ———
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.data.mul_(1 - cfg.TARGET_TAU).add_(p.data, alpha=cfg.TARGET_TAU)

            epsilon = max(cfg.EPS_END,
                          cfg.EPS_START - (cfg.EPS_START - cfg.EPS_END) * global_step / cfg.EPS_DECAY_STEPS)

            if (global_step > 0) and (global_step % cfg.SAVE_EVERY_STEPS == 0):
                tmp = os.path.join(run_dir, f"edge_q_step{global_step}.pt")
                torch.save(model.state_dict(), tmp)
                pbar.set_postfix_str(f'saved {tmp}')

            # ——— 8) 状态滚动 ———
            s_t = s_next
            E_prev = E_t
            global_step += 1

        # logger.log_epoch(epoch, conn, delay, hop, sw, epoch_total_r)
        # logger.log_epoch(epoch, epoch_total_r)
        end_time_epoch = datetime.now()
        cost_time = end_time_epoch - start_time  # timedelta
        print(f"cost time epoch-{epoch}:", cost_time)
        # —— 按 epoch 保存：可恢复训练的完整 checkpoint ——
        if getattr(cfg, "SAVE_EVERY_EPOCH", False) and ((epoch + 1) % getattr(cfg, "EPOCH_SAVE_INTERVAL", 1) == 0):
            ckpt = {
                "model": model.state_dict(),
                "target": target.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "epsilon": epsilon,
                "config": vars(cfg),  # 方便复现实验
                "cost_time_epoch": cost_time, # 训练时长 s
            }
            save_path = os.path.join(run_dir, f"epoch_{epoch + 1:03d}_step{global_step}.pt")
            torch.save(ckpt, save_path)
            print(f"[EPOCH SAVE] {save_path}")

    # ——— 收尾 ———
    logger.flush()
    torch.save(model.state_dict(), os.path.join(run_dir, "edge_q_final.pt"))
    print(f"[DONE] Run dir: {run_dir}")





def train_step(model, target, optim, scaler, device, amp_enabled, batch, replay=None, db=None):
    '''
    （Double DQN：online 选动作，target 估值）
    :param model:
    :param target:
    :param optim:
    :param scaler:
    :param device:
    :param amp_enabled:
    :param batch:
    :param replay:
    :param db:
    :return:
    '''
    model.train()

    valid_losses = []   # 收集每条样本的 loss，再平均一次反传
    for seq in batch:
        # 1) n-step 回报
        R = 0.0
        for k in range(cfg.N_STEP):
            R += (cfg.GAMMA ** k) * seq[k].r

        # 2) 目标 Q（Double DQN：online 选动作，target 估值）
        s_tpn: StateInterLayer = seq[cfg.N_STEP - 1].s_next
        feats_next, pairs_next = edge_features(
            cands=s_tpn.cand_edges, N=len(s_tpn.nodes),
            deg_base=s_tpn.deg_base, comp_id=s_tpn.comp_id,
            layer=s_tpn.layer_map, plane=s_tpn.plane_map,
            prev_edges=s_tpn.E_prev
        )

        if feats_next.size == 0:
            Q_boot = 0.0
        else:
            x_next = torch.tensor(feats_next, dtype=torch.float32, device=device)

            with torch.no_grad():
                # 1) 用 online(model) 打分来“选动作集合” A_star
                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        q_next_online = model(x_next)  # 只用于选
                else:
                    q_next_online = model(x_next)

                scores_next_online = {pairs_next[i]: float(q_next_online[i].item())
                                      for i in range(len(pairs_next))}
                max_deg_next = make_max_deg(s_tpn.layer_map)
                A_star = bmatch_with_requirements_interlayer_sgl(
                    scores=scores_next_online,  # ★ online 分数用于选
                    max_deg=max_deg_next,
                    E_core=s_tpn.E_core,
                    layer_map=s_tpn.layer_map,
                    plane_map=s_tpn.plane_map,
                    deg_isl_used=s_tpn.deg_isl_used,
                    N=len(s_tpn.nodes),
                    require_gs_per_layer=True,
                    max_repair_steps=256,
                )

                if not A_star:
                    A_star = safe_greedy_bmatch(
                        scores=scores_next_online,
                        max_deg=max_deg_next,
                        N_local=len(s_tpn.nodes)
                    )
                    if not hasattr(train_step, "_fallback_tpn_print"):
                        print("[fallback@tpn] bmatch_greedy empty at s_{t+n} → safe_greedy used.")
                        train_step._fallback_tpn_print = True

                A_star_interlayer = remove_E_core_graph(E_core=s_tpn.E_core, E_t=A_star)

                # 2) 用 target(target) 对“online 选出的边集”估值
                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        q_next_tgt = target(x_next)
                else:
                    q_next_tgt = target(x_next)

                scores_next_tgt = {pairs_next[i]: float(q_next_tgt[i].item())
                                   for i in range(len(pairs_next))}

                Q_boot = (sum(scores_next_tgt[e] for e in A_star_interlayer)
                          if A_star_interlayer else 0.0)

        y_t = torch.tensor(R + (cfg.GAMMA ** cfg.N_STEP) * Q_boot,
                           dtype=torch.float32, device=device)

        # 3) 估计 Q（在 s_t 上，用 online Q 对“本条样本实际执行的边集”求和）
        s0: StateInterLayer = seq[0].s
        feats_t, pairs_t = edge_features(
            cands=s0.cand_edges, N=len(s0.nodes),
            deg_base=s0.deg_base, comp_id=s0.comp_id,
            layer=s0.layer_map, plane=s0.plane_map,
            prev_edges=s0.E_prev
        )
        if feats_t.size == 0:
            # 当步没有候选边，无法训练——跳过该样本
            continue

        x_t = torch.tensor(feats_t, dtype=torch.float32, device=device)
        if amp_enabled:
            with torch.cuda.amp.autocast():
            # with get_autocast(device=device, dtype=(
            # torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16)):
                q_t = model(x_t)      # [E]

        else:
            q_t = model(x_t)

        # 建立 pair -> index 的映射，按选中边集把对应 q_t 累加（保持图不断）
        edge_index = {p: i for i, p in enumerate(pairs_t)}
        sel_idx = []
        for (u, v) in seq[0].E_t:
            e_ = (min(u, v), max(u, v))
            if e_ in edge_index:
                sel_idx.append(edge_index[e_])

        if not sel_idx:
            # 该样本执行的边在当前特征里不存在，训练信号为空——跳过
            continue

        idx_tensor = torch.as_tensor(sel_idx, device=device, dtype=torch.long)
        q_est = q_t.index_select(0, idx_tensor).sum()   # Tensor，带 grad

        # 4) 单样本损失
        # 保证 loss 计算用 FP32，避免 Half/Float 混用导致的 RuntimeError
        q_est = q_est.float()
        y_t = y_t.float()
        loss_i = F.huber_loss(q_est, y_t, delta=1.0)
        valid_losses.append(loss_i)

    # 5) 批量反传（若本批无有效样本，则不更新）
    if not valid_losses:
        return 0.0

    loss = torch.stack(valid_losses).mean()

    optim.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim)
    scaler.update()

    return float(loss.item())


if __name__ == "__main__":
    main()
