# topo/bmatch.py
# -*- coding: utf-8 -*-
'''
# ========= 约束友好的选择器：必选边预留 + 贪心补全 + 连通修复 =========
'''

from typing import Dict, Tuple, Set, List
import random
from config.config import cfg
from collections import defaultdict

# --------- 小工具 ---------
def _norm(u:int, v:int) -> Tuple[int,int]:
    return (u, v) if u < v else (v, u)

def _is_sgl(u:int, v:int, layer_map:Dict[int,str]) -> bool:
    # SGL：只要一端是 GS
    return (layer_map[u] == "GS") or (layer_map[v] == "GS")

def _is_sgl_interlayer(u:int, v:int, layer_map:Dict[int,str], nodes)->bool:
    Lu = layer_map[nodes[u].id]; Lv = layer_map[nodes[v].id]
    return (Lu=="GS") ^ (Lv=="GS")


def _is_interlayer_isl(u:int, v:int, layer_map:Dict[int,str])->bool:
    Lu = layer_map[u]; Lv = layer_map[v]
    return (Lu!=Lv) and (Lu!="GS") and (Lv!="GS")

def _max_isl_for_node(local_idx:int, nodes, layer_map)->int:
    L = layer_map[nodes[local_idx].id]
    if   L=="LEO": return cfg.MAX_ISL_LEO
    elif L=="MEO": return cfg.MAX_ISL_MEO
    elif L=="GEO": return cfg.MAX_ISL_GEO
    elif L=="GS":  return cfg.MAX_ISL_GS
    else:          return 0


def _make_max_deg_isl(layer_map:Dict[int,str], fallback:List[int]) -> List[int]:
    """
    ISL（星间）预算：
    - 若 cfg 有专门的 MAX_DEG_ISL_*，则使用；
    - 否则回退为传入的 fallback（即你原来的 max_deg）。
    """
    out = []
    has_typed = all(hasattr(cfg, k) for k in [
        "MAX_DEG_ISL_LEO", "MAX_DEG_ISL_MEO", "MAX_DEG_ISL_GEO"
    ])
    for i in range(len(layer_map)):
        L = layer_map[i]
        if has_typed:
            if   L == "LEO": out.append(cfg.MAX_DEG_ISL_LEO)
            elif L == "MEO": out.append(cfg.MAX_DEG_ISL_MEO)
            elif L == "GEO": out.append(cfg.MAX_DEG_ISL_GEO)
            elif L == "GS":  out.append(0)  # 通常 0
            else:            out.append(0)
        else:
            # 回退到原有单一预算
            out.append(fallback[i] if i < len(fallback) else 0)
    return out

def _make_max_deg_sgl_gs(layer_map:Dict[int,str]) -> List[int]:
    """
    SGL（星地）预算：仅对 GS 生效（其它节点为 0）
    - 若 cfg 有 MAX_DEG_SGL_GS，则使用；
    - 否则回退为 MAX_DEG_GS（若也没有，则 0）。
    """
    fallback_val = getattr(cfg, "MAX_DEG_SGL_GS",
                     getattr(cfg, "MAX_DEG_GS", 0))
    # print(f"MAX_DEG_SGL_GS: {fallback_val}")
    out = []
    for i in range(len(layer_map)):
        out.append(fallback_val if layer_map[i] == "GS" else 0)
    return out


def _make_max_deg_sgl_for_node(layer_map:Dict[int,str]) -> List[int]:
    out = []
    for i in range(len(layer_map)):
        L = layer_map[i]
        if L=="LEO": out.append(cfg.MAX_DEG_SGL_LEO)
        elif L=="MEO": out.append(cfg.MAX_DEG_SGL_MEO)
        elif L=="GEO": out.append(cfg.MAX_DEG_SGL_GEO)
        elif L=="GS": out.append(cfg.MAX_DEG_SGL_GS)
        else: out.append(0)
    return out


def _require_edges_for_geo_meo(scores: Dict[Tuple[int,int], float],
                               layer_map: Dict[int,str],
                               max_deg_isl: List[int],
                               deg_isl_used: List[int],
                               ordered: List[Tuple[Tuple[int,int], float]],
                               *,
                               plane_map: Dict[int,int] = None,
                               diag: Dict[int,dict] = None) -> Set[Tuple[int,int]]:
    """
    每个 MEO 轨道至少 1 条 GEO–MEO，取该轨道内可见边按分数排序，选度未满的第一个边。
    若未找到可选边，则告警并区分原因（没有可见边或容量不足）。
    """
    geo_ids = sorted([nid for nid, L in layer_map.items() if L == "GEO"])
    meo_ids = sorted([nid for nid, L in layer_map.items() if L == "MEO"])
    if not geo_ids or not meo_ids:
        return set()

    plane_to_meos: Dict[int, List[int]] = defaultdict(list)
    if plane_map:
        for m in meo_ids:
            plane_to_meos[plane_map.get(m, 0)].append(m)
    else:
        plane_to_meos[0] = meo_ids
    planes = sorted(plane_to_meos.keys())

    def _score_of(u:int, v:int) -> float:
        e = (u, v) if u < v else (v, u)
        return scores.get(e, float("-inf"))

    picked: Set[Tuple[int,int]] = set()

    # —— 统计器 —— #
    for p in planes:
        meos = plane_to_meos[p]
        best_e, best_s = None, float("-inf")

        # 统计信息
        visible_total = 0                # 可见边计数（不管容量）
        visible_cap_ok = 0               # 可见且容量充足的计数
        geo_cap_block = 0                # 因 GEO 端满而被挡的“可见边”数
        meo_cap_block = 0                # 因 MEO 端满而被挡的“可见边”数
        both_cap_block = 0               # 两端都满

        for m in meos:
            # 仅考虑度未满的 MEO
            if deg_isl_used[m] >= max_deg_isl[m]:
                continue
            # 当前轨道的所有 MEO 和 GEO 的边按分数降序排序
            visible_edges = [
                (g, m) if g < m else (m, g) for g in geo_ids
                if deg_isl_used[g] < max_deg_isl[g] and _score_of(g, m) != float("-inf")
            ]
            # 按分数从高到低排序
            visible_edges.sort(key=lambda edge: _score_of(*edge), reverse=True)

            # 遍历已排序的边，找到第一个两端度数未满的边
            for u, v in visible_edges:
                if deg_isl_used[u] < max_deg_isl[u] and deg_isl_used[v] < max_deg_isl[v]:
                    best_e = (u, v)
                    best_s = _score_of(u, v)
                    visible_cap_ok += 1
                    break

            # 统计容量不足的情况
            if best_e is None:
                if deg_isl_used[m] >= max_deg_isl[m]:
                    meo_cap_block += 1
                elif all(deg_isl_used[g] >= max_deg_isl[g] for g in geo_ids):
                    geo_cap_block += 1
                else:
                    both_cap_block += 1

        # —— 记录诊断 —— #
        if diag is not None:
            diag[p] = {
                "visible_total": visible_total,
                "visible_cap_ok": visible_cap_ok,
                "geo_cap_block": geo_cap_block,
                "meo_cap_block": meo_cap_block,
                "both_cap_block": both_cap_block,
                "picked_edge": best_e,
                "picked_score": (best_s if best_e is not None else None),
            }

        # —— 选择或告警 —— #
        if best_e is None:
            if visible_total == 0:
                print(f"[geo-meo][warn] plane={p}: no visible GEO–MEO edges.")
            else:
                # 有可见边但容量不够，细分原因
                msg_detail = []
                if both_cap_block: msg_detail.append(f"both_full={both_cap_block}")
                if geo_cap_block:  msg_detail.append(f"geo_full={geo_cap_block}")
                if meo_cap_block:  msg_detail.append(f"meo_full={meo_cap_block}")
                detail = ", ".join(msg_detail) if msg_detail else "capacity_exhausted"
                print(f"[geo-meo][warn] plane={p}: visible={visible_total}, "
                      f"cap_ok=0 ({detail}).")
            continue

        u, v = best_e
        # 二次容量确认并扣减
        if deg_isl_used[u] < max_deg_isl[u] and deg_isl_used[v] < max_deg_isl[v]:
            picked.add((u, v))
            deg_isl_used[u] += 1
            deg_isl_used[v] += 1
        else:
            # 理论上不会走到这里，留兜底打印
            print(f"[geo-meo][warn] plane={p}: capacity exhausted at pick-time for edge {(u,v)}.")

    return picked


def _require_edges_for_geo_meo2(scores: Dict[Tuple[int,int], float],
                               layer_map: Dict[int,str],
                               max_deg_isl: List[int],
                               deg_isl_used: List[int],
                               ordered: List[Tuple[Tuple[int,int], float]],
                               *,
                               plane_map: Dict[int,int] = None,
                               diag: Dict[int,dict] = None) -> Set[Tuple[int,int]]:
    """
    每个 MEO 轨道至少 1 条 GEO–MEO，取该轨道内可见边最高分。
    失败时区分：无可见边 vs. 容量不足（并细分哪一侧容量不足）。
    若传入 diag，将写入每个 plane 的统计信息。
    """
    geo_ids = sorted([nid for nid, L in layer_map.items() if L == "GEO"])
    meo_ids = sorted([nid for nid, L in layer_map.items() if L == "MEO"])
    if not geo_ids or not meo_ids:
        return set()

    plane_to_meos: Dict[int, List[int]] = defaultdict(list)
    if plane_map:
        for m in meo_ids:
            plane_to_meos[plane_map.get(m, 0)].append(m)
    else:
        plane_to_meos[0] = meo_ids
    planes = sorted(plane_to_meos.keys())

    def _score_of(u:int, v:int) -> float:
        e = (u, v) if u < v else (v, u)
        return scores.get(e, float("-inf"))

    picked: Set[Tuple[int,int]] = set()

    for p in planes:
        meos = plane_to_meos[p]
        best_e, best_s = None, float("-inf")

        # —— 统计器 —— #
        visible_total = 0                # 可见边计数（不管容量）
        visible_cap_ok = 0               # 可见且容量充足的计数
        geo_cap_block = 0                # 因 GEO 端满而被挡的“可见边”数
        meo_cap_block = 0                # 因 MEO 端满而被挡的“可见边”数
        both_cap_block = 0               # 两端都满

        for m in meos:
            meo_cap_ok = (0 <= m < len(max_deg_isl)) and (deg_isl_used[m] < max_deg_isl[m])
            for g in geo_ids:
                geo_cap_ok = (0 <= g < len(max_deg_isl)) and (deg_isl_used[g] < max_deg_isl[g])
                s = _score_of(g, m)
                if s == float("-inf"):
                    continue  # 不可见
                visible_total += 1
                if geo_cap_ok and meo_cap_ok:
                    visible_cap_ok += 1
                    if s > best_s:
                        best_s = s
                        best_e = (g, m) if g < m else (m, g)
                else:
                    # 统计是谁卡了容量
                    if not geo_cap_ok and not meo_cap_ok:
                        both_cap_block += 1
                    elif not geo_cap_ok:
                        geo_cap_block += 1
                    elif not meo_cap_ok:
                        meo_cap_block += 1

        # —— 记录诊断 —— #
        if diag is not None:
            diag[p] = {
                "visible_total": visible_total,
                "visible_cap_ok": visible_cap_ok,
                "geo_cap_block": geo_cap_block,
                "meo_cap_block": meo_cap_block,
                "both_cap_block": both_cap_block,
                "picked_edge": best_e,
                "picked_score": (best_s if best_e is not None else None),
            }

        # —— 选择或告警 —— #
        if best_e is None:
            if visible_total == 0:
                print(f"[geo-meo][warn] plane={p}: no visible GEO–MEO edges.")
            else:
                # 有可见边但容量不够，细分原因
                msg_detail = []
                if both_cap_block: msg_detail.append(f"both_full={both_cap_block}")
                if geo_cap_block:  msg_detail.append(f"geo_full={geo_cap_block}")
                if meo_cap_block:  msg_detail.append(f"meo_full={meo_cap_block}")
                detail = ", ".join(msg_detail) if msg_detail else "capacity_exhausted"
                print(f"[geo-meo][warn] plane={p}: visible={visible_total}, "
                      f"cap_ok=0 ({detail}).")
            continue

        u, v = best_e
        # 二次容量确认并扣减
        if deg_isl_used[u] < max_deg_isl[u] and deg_isl_used[v] < max_deg_isl[v]:
            picked.add((u, v))
            deg_isl_used[u] += 1
            deg_isl_used[v] += 1
        else:
            # 理论上不会走到这里，留兜底打印
            print(f"[geo-meo][warn] plane={p}: capacity exhausted at pick-time for edge {(u,v)}.")

    return picked


def _require_edges_for_geo_meo1(scores: Dict[Tuple[int,int], float],
                               layer_map: Dict[int,str],
                               max_deg_isl: List[int],
                               deg_isl_used: List[int],
                               ordered: List[Tuple[Tuple[int,int], float]],
                               *,
                               plane_map: Dict[int,int] = None) -> Set[Tuple[int,int]]:
    """
    为 GEO–MEO 添加“每个 MEO 轨道至少 1 条，且取该轨道内可见边最高分”的必选边。
    - MEO 集合：从 layer_map 中搜 L=='MEO' 的节点。
    - 轨道划分：若提供 plane_map（如 state.plane_map），按 plane_map[node_id] 分组；
      否则兜底把全部 MEO 放在同一轨道（0）。
    - GEO 候选：不再均分；对每个轨道，GEO 候选即所有 GEO 节点。
    - 选择：每轨道仅选 1 条可见且两端有剩余 ISL 端口的 GEO–MEO 边；即时扣两端 ISL 端口。
    """
    # —— 收集 GEO / MEO —— #
    geo_ids = sorted([nid for nid, L in layer_map.items() if L == "GEO"])
    meo_ids = sorted([nid for nid, L in layer_map.items() if L == "MEO"])
    print("geo_ids:", geo_ids)
    print("meo_ids:", meo_ids)
    if not geo_ids or not meo_ids:
        return set()

    # —— 按 plane_map 分轨（没有就把所有 MEO 放到轨道 0）—— #
    plane_to_meos: Dict[int, List[int]] = defaultdict(list)
    if plane_map:
        for m in meo_ids:
            p = plane_map.get(m, 0)  # 未给 plane 的记到 0
            plane_to_meos[p].append(m)
    else:
        plane_to_meos[0] = meo_ids

    planes = sorted(plane_to_meos.keys())
    print("plane_to_meos:", plane_to_meos)

    def _score_of(u:int, v:int) -> float:
        e = (u, v) if u < v else (v, u)
        return scores.get(e, float("-inf"))

    picked: Set[Tuple[int,int]] = set()

    # —— 每轨道挑 1 条最高分 GEO–MEO 边 —— #
    for p in planes:
        meos = plane_to_meos[p]
        best_e = None
        best_s = float("-inf")

        for m in meos:
            # MEO 端口满了跳过该 MEO
            if not (0 <= m < len(max_deg_isl)) or deg_isl_used[m] >= max_deg_isl[m]:
                continue
            for g in geo_ids:
                if not (0 <= g < len(max_deg_isl)) or deg_isl_used[g] >= max_deg_isl[g]:
                    continue
                s = _score_of(g, m)
                if s != float("-inf") and s > best_s:
                    best_s = s
                    best_e = (g, m) if g < m else (m, g)

        if best_e is None:
            print(f"[geo-meo][warn] plane={p}: no visible/capacity edge to pick.")
            continue

        u, v = best_e
        # 二次容量确认并扣减
        if deg_isl_used[u] < max_deg_isl[u] and deg_isl_used[v] < max_deg_isl[v]:
            picked.add((u, v))
            deg_isl_used[u] += 1
            deg_isl_used[v] += 1
        else:
            print(f"[geo-meo][warn] plane={p}: capacity exhausted at pick-time for edge {(u,v)}.")

    return picked




def _required_edges_for_gs_typed(scores: Dict[Tuple[int,int], float],
                                 layer_map: Dict[int,str],
                                 max_deg_sgl_gs: List[int],
                                 deg_sgl_gs: List[int],
                                 N: int,
                                 ordered: List[Tuple[Tuple[int,int], float]] = None
                                 ) -> Set[Tuple[int,int]]:
    """
    每个地面站 GS 对 LEO/MEO/GEO 分别至少选 K_LEO/K_MEO/K_GEO 条 SGL（若可见且有端口）。
    只占 GS 端口，不占卫星 ISL 端口。
    选完每层若未达标，会打印告警（端口不足或可见不足）。
    """
    need_per_layer = {
        "LEO": getattr(cfg, "REQ_SGL_LEO", 0),
        "MEO": getattr(cfg, "REQ_SGL_MEO", 0),
        "GEO": getattr(cfg, "REQ_SGL_GEO", 0),
    }
    if (need_per_layer["LEO"] <= 0 and
        need_per_layer["MEO"] <= 0 and
        need_per_layer["GEO"] <= 0):
        return set()

    if ordered is None:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    req: Set[Tuple[int,int]] = set()

    for g in range(N):
        if layer_map[g] != "GS":
            continue
        # 每个 GS 单独的需求副本
        need_g = dict(need_per_layer)

        for L in ("LEO", "MEO", "GEO"):
            k_needed_init = max(int(need_g[L]), 0)
            if k_needed_init == 0:
                continue
            k_needed = k_needed_init

            # 扫描降序候选，拿该 GS→该层 的 top-K
            for (u, v), s in ordered:
                if deg_sgl_gs[g] >= max_deg_sgl_gs[g]:
                    break  # GS 端口已满
                u_, v_ = _norm(u, v)
                if g not in (u_, v_):
                    continue
                j = v_ if u_ == g else u_
                if layer_map[j] != L:
                    continue
                e = (u_, v_)
                if e in req:
                    continue
                # 只要在 scores 中出现就视为可见
                req.add(e)
                deg_sgl_gs[g] += 1
                k_needed -= 1
                if k_needed == 0 or deg_sgl_gs[g] >= max_deg_sgl_gs[g]:
                    break

            # —— 未达标则告警 —— #
            if k_needed > 0:
                # 统计该 GS 对该层的可见候选总数（便于区分“可见不足”）
                cand_total = 0
                for (u, v), _ in ordered:
                    u_, v_ = _norm(u, v)
                    if g not in (u_, v_):
                        continue
                    j = v_ if u_ == g else u_
                    if layer_map[j] == L:
                        cand_total += 1

                picked = k_needed_init - k_needed
                gs_full = (deg_sgl_gs[g] >= max_deg_sgl_gs[g])
                reason = "ports_exhausted" if gs_full else "insufficient_visible_edges"

                print(
                    f"[quota][warn] GS#{g} for layer {L}: need={k_needed_init}, "
                    f"got={picked}, visible_candidates={cand_total}, "
                    f"gs_ports={deg_sgl_gs[g]}/{max_deg_sgl_gs[g]}, reason={reason}"
                )

    return req




def _required_edges_for_gs_layers(
    scores: Dict[Tuple[int, int], float],
    layer_map: Dict[int, str],          # 每个节点所属层：{"GS","LEO","MEO","GEO"}
    max_deg_sgl: List[int],          # 各 节点 的 SGL 端口上限（长度>=N）
    deg_sgl_used: List[int],              # 各 节点 当前已占用的 SGL 端口数（会被原地更新）
    ordered: List[Tuple[Tuple[int,int], float]] = None
) -> Set[Tuple[int,int]]:
    """
    GS“整层”对 LEO/MEO/GEO 分别至少选 REQ_SGL_LEO / REQ_SGL_MEO / REQ_SGL_GEO  条 SGL（若可见且两端有端口）。
    占用两端的 SGL 端口：一端为 GS（受 max_deg_sgl_gs 限制），另一端为卫星（受按层统一上限限制）。
    —— 不涉及 ISL 端口计数 ——

    读取 cfg：
      REQ_SGL_LEO / REQ_SGL_MEO / REQ_SGL_GEO  : GS层对各层至少需要的连边数
      MAX_DEG_SGL_LEO / MAX_DEG_SGL_MEO / MAX_DEG_SGL_GEO : 单星按层 SGL 上限（默认 3）
    """
    # 需求（整层配额）
    need = {
        "LEO": getattr(cfg, "REQ_SGL_LEO", 1),
        "MEO": getattr(cfg, "REQ_SGL_MEO", 1),
        "GEO": getattr(cfg, "REQ_SGL_GEO", 1),
    }
    if all(int(need[L]) <= 0 for L in ("LEO","MEO","GEO")):
        return set()

    # 评分降序
    if ordered is None:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _norm(u: int, v: int) -> Tuple[int,int]:
        return (u, v) if u <= v else (v, u)

    req: Set[Tuple[int,int]] = set()

    # 预分桶：仅保留“GS↔目标层”的候选
    buckets = {"LEO": [], "MEO": [], "GEO": []}
    for (u, v), s in ordered:
        a, b = _norm(u, v)
        La, Lb = layer_map.get(a), layer_map.get(b)
        if La == "GS" and Lb in buckets:
            buckets[Lb].append(((a, b), s))
        elif Lb == "GS" and La in buckets:
            buckets[La].append(((a, b), s))

    # 逐层满足配额；挑边时同时占用 GS 与卫星的 SGL 端口
    for L in ("LEO","MEO","GEO"):
        need_L = int(max(0, need[L]))
        if need_L == 0:
            continue

        for (u, v), s in buckets[L]:
            if need_L == 0:
                break
            e = _norm(u, v)
            if e in req:
                continue

            # —— 双端端口检查（两端都必须有剩余）——
            if deg_sgl_used[u] >= max_deg_sgl[u] or deg_sgl_used[v] >= max_deg_sgl[v]:
                continue

            # 满足条件：选择该边，并占用两端 SGL 端口
            req.add(e)
            deg_sgl_used[u]  += 1
            deg_sgl_used[v] += 1
            need_L -= 1

        # 未达标打印诊断
        if need_L > 0:
            vis_total = len(buckets[L])
            gs_ok = sat_ok = both_ok = 0
            for (u, v), _ in buckets[L]:
                if layer_map[u] == "GS":
                    gs, sat = u, v
                else:
                    gs, sat = v, u
                cond_gs  = (deg_sgl_used[gs]  < max_deg_sgl[gs])
                cond_sat = (deg_sgl_used[sat] < max_deg_sgl[sat])
                gs_ok  += 1 if cond_gs else 0
                sat_ok += 1 if cond_sat else 0
                both_ok += 1 if (cond_gs and cond_sat) else 0

            picked = int(need[L]) - need_L
            if vis_total == 0:
                reason = "insufficient_visible_edges"
            elif both_ok == 0:
                if gs_ok == 0 and sat_ok == 0:
                    reason = "gs_and_satellite_ports_exhausted"
                elif gs_ok == 0:
                    reason = "gs_ports_exhausted"
                elif sat_ok == 0:
                    reason = "satellite_ports_exhausted"
                else:
                    reason = "mixed_constraints"
            else:
                reason = "mixed_constraints"

            print(
                f"[quota][warn] GS-layer -> {L}: need={int(need[L])}, "
                f"got={picked}, visible={vis_total}, gs_ok={gs_ok}, "
                f"sat_ok={sat_ok}, both_ok={both_ok}, reason={reason}"
            )

    return req



# DSU / 可视连通性检查函数
class _DSU:
    def __init__(self, n:int):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.comp = n
    def find(self, x:int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a:int, b:int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        self.comp -= 1

def _vis_graph_connected(N:int, scores:Dict[Tuple[int,int],float]) -> bool:
    # 仅按“可见边”（即出现在 scores 的边）检查可能连通性
    dsu = _DSU(N)
    for (u, v) in scores.keys():
        u_, v_ = _norm(u, v)
        dsu.union(u_, v_)
    return dsu.comp == 1


# ------------------------------------
def bmatch_with_requirements_interlayer_sgl(scores: Dict[Tuple[int,int], float],
                             max_deg: List[int],
                             E_core: Set[Tuple[int,int]],
                             layer_map: Dict[int,str],
                             plane_map: Dict[int,int],
                             deg_isl_used: List[int],      # 由 E_base 得到的已占用 ISL
                             N: int,
                             require_gs_per_layer: bool = True,
                             max_repair_steps: int = 256) -> Set[Tuple[int,int]]:
    """
    类型化 b-匹配（兼容旧签名）：
      - ISL（卫星-卫星）占用两端 ISL 端口（deg_isl <= max_deg_isl）
      - SGL（卫星-地面）只占地面站 SGL 端口（deg_sgl_gs <= max_deg_sgl_gs）
      - 可选“GS 对 LEO/MEO/GEO 保底 1 条”约束
      - 若不连通，则用“跨分量高分边”进行连通性修复（限步数）
    注意：
      * 若 cfg 未配置 MAX_DEG_ISL_* / MAX_DEG_SGL_GS，会自动回退用传入的 max_deg 和 MAX_DEG_GS。
    """
    # 1) 度预算准备（带回退）
    max_deg_isl    = _make_max_deg_isl(layer_map, fallback=max_deg)          # ISL 预算
    max_deg_sgl = _make_max_deg_sgl_for_node(layer_map)                 # SGL预算

    # 2) 候选边按分数降序
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # 3) 可视图是否本身连通（仅用于提示）
    possible_conn = _vis_graph_connected(N, scores)
    if not possible_conn:
        print(f"possible_conn:{possible_conn}")

    # 4) 选择集与度计数
    E_sel: Set[Tuple[int,int]] = set(E_core)  # 层内拓扑打底
    deg_sgl_used = [0]*N           # 所有节点 的 SGL 使用

    # 添加函数_require_edges_for_geo_meo
    # 给E_sel添加GEO-MEO的边
    E_geo_meo = _require_edges_for_geo_meo2(
        scores=scores,
        layer_map=layer_map,
        max_deg_isl=max_deg_isl,
        deg_isl_used=deg_isl_used,  # 就地扣减端口
        ordered=ordered,
        plane_map=plane_map  # ← 传入轨道编号
    )
    for (u, v) in E_geo_meo:
        E_sel.add((u, v))

    # 5) 必选：每 GS 对每层（LEO/MEO/GEO）各 1 条（若可见且有 SGL 端口）
    if require_gs_per_layer:
        E_req = _required_edges_for_gs_layers(
            scores=scores,
            layer_map=layer_map,
            max_deg_sgl=max_deg_sgl,
            deg_sgl_used=deg_sgl_used,
            ordered=ordered,  # ← 复用已排序的候选，加速
        )

        for (u, v) in E_req:
            u_, v_ = _norm(u, v)
            E_sel.add((u_, v_))

    # 6) 贪心补全（区分 ISL / SGL）
    for (u, v), s in ordered:
        u_, v_ = _norm(u, v)
        if (u_, v_) in E_sel:
            continue
        if _is_sgl(u_, v_, layer_map):
            if deg_sgl_used[u_] < max_deg_sgl[u_] and deg_sgl_used[v_] < max_deg_sgl[v_]:
                E_sel.add((u_, v_))
                deg_sgl_used[u_] += 1
                deg_sgl_used[v_] += 1
        elif _is_interlayer_isl(u_,v_, layer_map):
            # ISL：两端都非 GS,且layer不同
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1

    # 7) 连通性检查
    dsu = _DSU(N)
    for (u, v) in E_sel:
        dsu.union(u, v)
    if dsu.comp == 1:
        return E_sel  # 已连通

    # 8) 连通性修复：挑“跨分量高分边”，并遵守各自端口预算
    steps = 0
    for (u, v), s in ordered:
        if dsu.comp == 1 or steps >= max_repair_steps:
            break
        u_, v_ = _norm(u, v)
        ru, rv = dsu.find(u_), dsu.find(v_)
        if ru == rv:
            continue  # 同分量内无助于连通

        if _is_sgl(u_, v_, layer_map):

            if deg_sgl_used[u_] < max_deg_sgl[u_] and deg_sgl_used[v_] < max_deg_sgl[v_]:
                E_sel.add((u_, v_))
                deg_sgl_used[u_] += 1
                deg_sgl_used[v_] += 1
                dsu.union(u_, v_)
                steps += 1
        else:
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1
                dsu.union(u_, v_)
                steps += 1

    # 9) 提示
    if dsu.comp > 1 and not hasattr(bmatch_with_requirements, "_warn_unconnected"):
        if possible_conn:
            print("[repair] warning: still not fully connected after repair (degree-limited).")
        else:
            print("[repair] info: visibility graph itself is disconnected this minute; full connectivity impossible.")
        bmatch_with_requirements._warn_unconnected = True

    return E_sel

# ------------------------------------
def bmatch_with_requirements_interlayer(scores: Dict[Tuple[int,int], float],
                             max_deg: List[int],
                             E_core: Set[Tuple[int,int]],
                             layer_map: Dict[int,str],
                             plane_map: Dict[int,int],
                             deg_isl_used: List[int],      # 由 E_base 得到的已占用 ISL
                             N: int,
                             require_gs_per_layer: bool = True,
                             max_repair_steps: int = 256) -> Set[Tuple[int,int]]:
    """
    类型化 b-匹配（兼容旧签名）：
      - ISL（卫星-卫星）占用两端 ISL 端口（deg_isl <= max_deg_isl）
      - SGL（卫星-地面）只占地面站 SGL 端口（deg_sgl_gs <= max_deg_sgl_gs）
      - 可选“GS 对 LEO/MEO/GEO 保底 1 条”约束
      - 若不连通，则用“跨分量高分边”进行连通性修复（限步数）
    注意：
      * 若 cfg 未配置 MAX_DEG_ISL_* / MAX_DEG_SGL_GS，会自动回退用传入的 max_deg 和 MAX_DEG_GS。
    """
    # 1) 度预算准备（带回退）
    max_deg_isl    = _make_max_deg_isl(layer_map, fallback=max_deg)          # ISL 预算
    max_deg_sgl_gs = _make_max_deg_sgl_gs(layer_map)                          # SGL(GS)预算

    # 2) 候选边按分数降序
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # 3) 可视图是否本身连通（仅用于提示）
    possible_conn = _vis_graph_connected(N, scores)
    # print(f"possible_conn:{possible_conn}")

    # 4) 选择集与度计数
    E_sel: Set[Tuple[int,int]] = set(E_core)  # 层内拓扑打底
    # deg_isl    = [0]*N           # 仅 ISL 使用
    deg_sgl_gs = [0]*N           # 仅 GS 的 SGL 使用

    # 添加函数_require_edges_for_geo_meo
    # 给E_sel添加GEO-MEO的边
    E_geo_meo = _require_edges_for_geo_meo2(
        scores=scores,
        layer_map=layer_map,
        max_deg_isl=max_deg_isl,
        deg_isl_used=deg_isl_used,  # 就地扣减端口
        ordered=ordered,
        plane_map=plane_map  # ← 传入轨道编号
    )
    for (u, v) in E_geo_meo:
        E_sel.add((u, v))

    # 5) 必选：每 GS 对每层（LEO/MEO/GEO）各 1 条（若可见且有 SGL 端口）
    if require_gs_per_layer:
        # 原：E_req = _required_edges_for_gs_typed(scores, layer_map, max_deg_sgl_gs, deg_sgl_gs, N)
        E_req = _required_edges_for_gs_typed(
            scores=scores,
            layer_map=layer_map,
            max_deg_sgl_gs=max_deg_sgl_gs,
            deg_sgl_gs=deg_sgl_gs,
            N=N,
            ordered=ordered,  # ← 复用已排序的候选，加速
        )

        for (u, v) in E_req:
            u_, v_ = _norm(u, v)
            E_sel.add((u_, v_))

    # 6) 贪心补全（区分 ISL / SGL）
    for (u, v), s in ordered:
        u_, v_ = _norm(u, v)
        if (u_, v_) in E_sel:
            continue
        if _is_sgl(u_, v_, layer_map):
            # 只检查 GS 端口
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
        elif _is_interlayer_isl(u_,v_, layer_map):
            # ISL：两端都非 GS,且layer不同
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1

    # 7) 连通性检查
    dsu = _DSU(N)
    for (u, v) in E_sel:
        dsu.union(u, v)
    if dsu.comp == 1:
        return E_sel  # 已连通

    # 8) 连通性修复：挑“跨分量高分边”，并遵守各自端口预算
    steps = 0
    for (u, v), s in ordered:
        if dsu.comp == 1 or steps >= max_repair_steps:
            break
        u_, v_ = _norm(u, v)
        ru, rv = dsu.find(u_), dsu.find(v_)
        if ru == rv:
            continue  # 同分量内无助于连通

        if _is_sgl(u_, v_, layer_map):
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
                dsu.union(u_, v_)
                steps += 1
        else:
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1
                dsu.union(u_, v_)
                steps += 1

    # 9) 提示
    if dsu.comp > 1 and not hasattr(bmatch_with_requirements, "_warn_unconnected"):
        if possible_conn:
            print("[repair] warning: still not fully connected after repair (degree-limited).")
        else:
            print("[repair] info: visibility graph itself is disconnected this minute; full connectivity impossible.")
        bmatch_with_requirements._warn_unconnected = True

    return E_sel



# ------------------------------------
def bmatch_with_requirements_interlayer1(scores: Dict[Tuple[int,int], float],
                             max_deg: List[int],
                             E_core: Set[Tuple[int,int]],
                             layer_map: Dict[int,str],
                             deg_isl_used: List[int],      # 由 E_base 得到的已占用 ISL
                             N: int,
                             require_gs_per_layer: bool = True,
                             max_repair_steps: int = 256) -> Set[Tuple[int,int]]:
    """
    类型化 b-匹配（兼容旧签名）：
      - ISL（卫星-卫星）占用两端 ISL 端口（deg_isl <= max_deg_isl）
      - SGL（卫星-地面）只占地面站 SGL 端口（deg_sgl_gs <= max_deg_sgl_gs）
      - 可选“GS 对 LEO/MEO/GEO 保底 1 条”约束
      - 若不连通，则用“跨分量高分边”进行连通性修复（限步数）
    注意：
      * 若 cfg 未配置 MAX_DEG_ISL_* / MAX_DEG_SGL_GS，会自动回退用传入的 max_deg 和 MAX_DEG_GS。
    """
    # 1) 度预算准备（带回退）
    max_deg_isl    = _make_max_deg_isl(layer_map, fallback=max_deg)          # ISL 预算
    max_deg_sgl_gs = _make_max_deg_sgl_gs(layer_map)                          # SGL(GS)预算

    # 2) 候选边按分数降序
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # 3) 可视图是否本身连通（仅用于提示）
    possible_conn = _vis_graph_connected(N, scores)
    # print(f"possible_conn:{possible_conn}")

    # 4) 选择集与度计数
    E_sel: Set[Tuple[int,int]] = set(E_core)  # 层内拓扑打底
    # deg_isl    = [0]*N           # 仅 ISL 使用
    deg_sgl_gs = [0]*N           # 仅 GS 的 SGL 使用

    # 添加函数_require_edges_for_geo_meo
    # 给E_sel添加GEO-MEO的边




    # 5) 必选：每 GS 对每层（LEO/MEO/GEO）各 1 条（若可见且有 SGL 端口）
    if require_gs_per_layer:
        # 原：E_req = _required_edges_for_gs_typed(scores, layer_map, max_deg_sgl_gs, deg_sgl_gs, N)
        E_req = _required_edges_for_gs_typed(
            scores=scores,
            layer_map=layer_map,
            max_deg_sgl_gs=max_deg_sgl_gs,
            deg_sgl_gs=deg_sgl_gs,
            N=N,
            ordered=ordered,  # ← 复用已排序的候选，加速
        )

        for (u, v) in E_req:
            u_, v_ = _norm(u, v)
            E_sel.add((u_, v_))

    # 6) 贪心补全（区分 ISL / SGL）
    for (u, v), s in ordered:
        u_, v_ = _norm(u, v)
        if (u_, v_) in E_sel:
            continue
        if _is_sgl(u_, v_, layer_map):
            # 只检查 GS 端口
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
        elif _is_interlayer_isl(u_,v_, layer_map):
            # ISL：两端都非 GS,且layer不同
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1

    # 7) 连通性检查
    dsu = _DSU(N)
    for (u, v) in E_sel:
        dsu.union(u, v)
    if dsu.comp == 1:
        return E_sel  # 已连通

    # 8) 连通性修复：挑“跨分量高分边”，并遵守各自端口预算
    steps = 0
    for (u, v), s in ordered:
        if dsu.comp == 1 or steps >= max_repair_steps:
            break
        u_, v_ = _norm(u, v)
        ru, rv = dsu.find(u_), dsu.find(v_)
        if ru == rv:
            continue  # 同分量内无助于连通

        if _is_sgl(u_, v_, layer_map):
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
                dsu.union(u_, v_)
                steps += 1
        else:
            if deg_isl_used[u_] < max_deg_isl[u_] and deg_isl_used[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl_used[u_] += 1
                deg_isl_used[v_] += 1
                dsu.union(u_, v_)
                steps += 1

    # 9) 提示
    if dsu.comp > 1 and not hasattr(bmatch_with_requirements, "_warn_unconnected"):
        if possible_conn:
            print("[repair] warning: still not fully connected after repair (degree-limited).")
        else:
            print("[repair] info: visibility graph itself is disconnected this minute; full connectivity impossible.")
        bmatch_with_requirements._warn_unconnected = True

    return E_sel






# ------------------ b-matching with gs degree split ------------------
def bmatch_with_requirements(scores: Dict[Tuple[int,int], float],
                             max_deg: List[int],
                             layer_map: Dict[int,str],
                             N: int,
                             require_gs_per_layer: bool = True,
                             max_repair_steps: int = 256) -> Set[Tuple[int,int]]:
    """
    类型化 b-匹配（兼容旧签名）：
      - ISL（卫星-卫星）占用两端 ISL 端口（deg_isl <= max_deg_isl）
      - SGL（卫星-地面）只占地面站 SGL 端口（deg_sgl_gs <= max_deg_sgl_gs）
      - 可选“GS 对 LEO/MEO/GEO 保底 1 条”约束
      - 若不连通，则用“跨分量高分边”进行连通性修复（限步数）
    注意：
      * 若 cfg 未配置 MAX_DEG_ISL_* / MAX_DEG_SGL_GS，会自动回退用传入的 max_deg 和 MAX_DEG_GS。
    """
    # 1) 度预算准备（带回退）
    max_deg_isl    = _make_max_deg_isl(layer_map, fallback=max_deg)          # ISL 预算
    max_deg_sgl_gs = _make_max_deg_sgl_gs(layer_map)                          # SGL(GS)预算

    # 2) 候选边按分数降序
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # 3) 可视图是否本身连通（仅用于提示）
    possible_conn = _vis_graph_connected(N, scores)
    # print(f"possible_conn:{possible_conn}")

    # 4) 选择集与度计数
    E_sel: Set[Tuple[int,int]] = set()
    deg_isl    = [0]*N           # 仅 ISL 使用
    deg_sgl_gs = [0]*N           # 仅 GS 的 SGL 使用

    # 5) 必选：每 GS 对每层（LEO/MEO/GEO）各 1 条（若可见且有 SGL 端口）
    if require_gs_per_layer:
        # 原：E_req = _required_edges_for_gs_typed(scores, layer_map, max_deg_sgl_gs, deg_sgl_gs, N)
        E_req = _required_edges_for_gs_typed(
            scores=scores,
            layer_map=layer_map,
            max_deg_sgl_gs=max_deg_sgl_gs,
            deg_sgl_gs=deg_sgl_gs,
            N=N,
            ordered=ordered,  # ← 复用已排序的候选，加速
        )

        for (u, v) in E_req:
            u_, v_ = _norm(u, v)
            E_sel.add((u_, v_))

    # 6) 贪心补全（区分 ISL / SGL）
    for (u, v), s in ordered:
        u_, v_ = _norm(u, v)
        if (u_, v_) in E_sel:
            continue
        if _is_sgl(u_, v_, layer_map):
            # 只检查 GS 端口
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
        else:
            # ISL：两端都非 GS
            if deg_isl[u_] < max_deg_isl[u_] and deg_isl[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl[u_] += 1
                deg_isl[v_] += 1

    # 7) 连通性检查
    dsu = _DSU(N)
    for (u, v) in E_sel:
        dsu.union(u, v)
    if dsu.comp == 1:
        return E_sel  # 已连通

    # 8) 连通性修复：挑“跨分量高分边”，并遵守各自端口预算
    steps = 0
    for (u, v), s in ordered:
        if dsu.comp == 1 or steps >= max_repair_steps:
            break
        u_, v_ = _norm(u, v)
        ru, rv = dsu.find(u_), dsu.find(v_)
        if ru == rv:
            continue  # 同分量内无助于连通

        if _is_sgl(u_, v_, layer_map):
            g = u_ if layer_map[u_] == "GS" else v_
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E_sel.add((u_, v_))
                deg_sgl_gs[g] += 1
                dsu.union(u_, v_)
                steps += 1
        else:
            if deg_isl[u_] < max_deg_isl[u_] and deg_isl[v_] < max_deg_isl[v_]:
                E_sel.add((u_, v_))
                deg_isl[u_] += 1
                deg_isl[v_] += 1
                dsu.union(u_, v_)
                steps += 1

    # 9) 提示
    if dsu.comp > 1 and not hasattr(bmatch_with_requirements, "_warn_unconnected"):
        if possible_conn:
            print("[repair] warning: still not fully connected after repair (degree-limited).")
        else:
            print("[repair] info: visibility graph itself is disconnected this minute; full connectivity impossible.")
        bmatch_with_requirements._warn_unconnected = True

    return E_sel


def bmatch_with_dual_budgets(scores: Dict[Tuple[int,int], float],
                             nodes,
                             layer_map: Dict[int,str],
                             deg_isl_used: List[int],      # 由 E_base 得到的已占用 ISL
                             require_gs_per_layer: bool=True,
                             max_repair_steps:int=256) -> Set[Tuple[int,int]]:
    """
    只在候选里挑‘层间 ISL’ & ‘SGL’，并用两套度池约束：
      - ISL: 按层取 cfg.MAX_ISL_*，并扣除 E_fixed 已占用的 deg_isl_used
      - SGL: 卫星端用 cfg.MAX_SGL_SAT；地面站端用 cfg.MAX_SGL_GS
    仍保留：
      - 地面站每层最小连接数（cfg.REQ_GS_PER_LAYER）
      - 若不连通，跨分量修复（有限步）
    """
    N = len(nodes)
    # —— 初始化可用预算 ——
    isl_cap = [max(0, _max_isl_for_node(i, nodes, layer_map) - deg_isl_used[i]) for i in range(N)]
    sgl_cap_sat = [cfg.MAX_SGL_SAT if layer_map[nodes[i].id]!="GS" else 0 for i in range(N)]
    sgl_cap_gs  = [cfg.MAX_SGL_GS if layer_map[nodes[i].id]=="GS" else 0 for i in range(N)]

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    E_sel: Set[Tuple[int,int]] = set()

    # —— (A) 先满足“GS 每层至少若干条”的硬约束 ——
    if require_gs_per_layer:
        need = {gs: dict(cfg.REQ_GS_PER_LAYER) for gs in range(N) if layer_map[nodes[gs].id]=="GS"}
        # 扫一遍候选（按高分优先），尽力满足
        for (u,v), s in ordered:
            is_sgl = _is_sgl_interlayer(u,v,layer_map,nodes)
            if not is_sgl:
                continue
            gs = u if layer_map[nodes[u].id]=="GS" else v
            sat = v if gs==u else u
            Lsat = layer_map[nodes[sat].id]  # 卫星层
            if Lsat not in ("LEO","MEO","GEO"):
                continue
            if need.get(gs,{}).get(Lsat,0) <= 0:
                continue
            # 预算检查（SGL 双端）
            if sgl_cap_gs[gs] <= 0 or sgl_cap_sat[sat] <= 0:
                continue
            # 选边
            a,b = (u,v) if u<v else (v,u)
            if (a,b) in E_sel:
                continue
            E_sel.add((a,b))
            sgl_cap_gs[gs]  -= 1
            sgl_cap_sat[sat]-= 1
            need[gs][Lsat]  -= 1

    # —— (B) 贪心补全：先用 ISL，再用 SGL（或你想要的顺序）——
    # 你可以混合遍历，只要在选择时按边类型扣对应度池即可
    for (u,v), s in ordered:
        a,b = (u,v) if u<v else (v,u)
        if (a,b) in E_sel:
            continue
        if _is_interlayer_isl(a,b, layer_map):
            if isl_cap[a] > 0 and isl_cap[b] > 0:
                E_sel.add((a,b))
                isl_cap[a] -= 1; isl_cap[b] -= 1
        elif _is_sgl_interlayer(a,b, layer_map, nodes):
            gs  = a if layer_map[nodes[a].id]=="GS" else b
            sat = b if gs==a else a
            if sgl_cap_gs[gs] > 0 and sgl_cap_sat[sat] > 0:
                E_sel.add((a,b))
                sgl_cap_gs[gs]  -= 1
                sgl_cap_sat[sat]-= 1

    # —— (C) 连通性修复（只在不连通时）——
    # 仍然只允许选择候选集里的边；按类型扣对应度池
    dsu = _DSU(N)
    for (u,v) in E_sel:
        dsu.union(u,v)
    if dsu.comp > 1:
        steps = 0
        for (u,v), s in ordered:
            if steps>=max_repair_steps or dsu.comp==1:
                break
            a,b = (u,v) if u<v else (v,u)
            if (a,b) in E_sel:
                continue
            if _is_interlayer_isl(a,b, layer_map):
                if isl_cap[a] > 0 and isl_cap[b] > 0 and dsu.find(a)!=dsu.find(b):
                    E_sel.add((a,b))
                    isl_cap[a]-=1; isl_cap[b]-=1
                    dsu.union(a,b); steps+=1
            elif _is_sgl_interlayer(a,b, layer_map, nodes):
                gs  = a if layer_map[nodes[a].id]=="GS" else b
                sat = b if gs==a else a
                if sgl_cap_gs[gs] > 0 and sgl_cap_sat[sat] > 0 and dsu.find(a)!=dsu.find(b):
                    E_sel.add((a,b))
                    sgl_cap_gs[gs]-=1; sgl_cap_sat[sat]-=1
                    dsu.union(a,b); steps+=1

    return E_sel



def bmatch_typed(
    scores: Dict[Tuple[int,int], float],
    layer_map: Dict[int, str],
    max_deg_isl: List[int],      # 每节点的 ISL 预算
    max_deg_sgl_gs: List[int],   # 仅 GS 的 SGL 预算（其它节点为 0）
) -> Set[Tuple[int,int]]:
    """
    类型化 b-匹配（贪心）：区分 ISL 与 SGL 两类边
    - ISL（非 GS-GS 且不含 GS）：两端用 deg_isl 计数，受 max_deg_isl 约束
    - SGL（含 GS 的边）：只在 GS 端用 deg_sgl_gs 计数，卫星端不占用其 ISL 预算
    """
    # 预备
    N = len(layer_map)
    def norm(u,v): return (u,v) if u<v else (v,u)

    deg_isl = [0]*N
    deg_sgl_gs = [0]*N
    E: Set[Tuple[int,int]] = set()

    # 从高分到低分挑边
    for (u,v), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        u_, v_ = norm(u,v)
        Lu, Lv = layer_map[u_], layer_map[v_]
        # SGL：任一端为 GS
        if Lu=="GS" or Lv=="GS":
            g = u_ if Lu=="GS" else v_
            # 只在 GS 端检查 SGL 预算
            if deg_sgl_gs[g] < max_deg_sgl_gs[g]:
                E.add((u_, v_))
                deg_sgl_gs[g] += 1
        else:
            # ISL：两端都不是 GS
            if deg_isl[u_] < max_deg_isl[u_] and deg_isl[v_] < max_deg_isl[v_]:
                E.add((u_, v_))
                deg_isl[u_] += 1
                deg_isl[v_] += 1
    return E


def safe_greedy_bmatch_typed(scores, layer_map, max_deg_isl, max_deg_sgl_gs):
    # 与 bmatch_typed 同逻辑，只是更宽松/不报错；可直接调用 bmatch_typed 即可
    return bmatch_typed(scores, layer_map, max_deg_isl, max_deg_sgl_gs)





"""
b-匹配：贪心 + 2-opt
"""

def bmatch_greedy(scores:Dict[Tuple[int,int], float],
                  max_deg:List[int]) -> Set[Tuple[int,int]]:

    # # ——调试统计，仅打印一次——
    # if getattr(bmatch_greedy, "_seen_", False) is False:
    #     bmatch_greedy._seen_ = True
    #     vals = list(scores.values())
    #     if vals:
    #         vals_sorted = sorted(vals)
    #         mid = vals_sorted[len(vals)//2]
    #         print(f"[bmatch] scores: n={len(vals)} min={vals_sorted[0]:.4f} med={mid:.4f} max={vals_sorted[-1]:.4f}")
    #     else:
    #         print("[bmatch] scores: EMPTY")
    #     print(f"[bmatch] max_deg: n={len(max_deg)} zeros={sum(1 for d in max_deg if d==0)}")

    deg=[0]*len(max_deg)
    E_sel=set()
    for (u,v),w in sorted(scores.items(), key=lambda kv: -kv[1]):
        if deg[u]<max_deg[u] and deg[v]<max_deg[v]:
            E_sel.add((u,v)); deg[u]+=1; deg[v]+=1
    return E_sel


# 安全兜底：分数从大到小，满足度约束就选
def safe_greedy_bmatch(scores: dict, max_deg: list, N_local: int):
    # 检查并修正度数组
    if not isinstance(max_deg, list) or len(max_deg) != N_local or all(d == 0 for d in max_deg):
        print("[warn] max_deg invalid; fallback to cfg.MAX_LINK for all nodes.")
        from ..config import cfg
        max_deg = [cfg.MAX_LINK] * N_local

    deg = [0] * N_local
    E = set()
    # 分数从高到低遍历
    for (u, v), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        # 本地索引域保护
        if not (0 <= u < N_local and 0 <= v < N_local):
            continue
        # 度约束
        if deg[u] < max_deg[u] and deg[v] < max_deg[v]:
            E.add((u, v))
            deg[u] += 1
            deg[v] += 1
    return E




def two_opt_placeholder(E_sel: Set[Tuple[int, int]],
                        scores: Dict[Tuple[int, int], float],
                        max_deg: List[int],
                        max_swaps: int = 80) -> Set[Tuple[int, int]]:
    """
    简化版 2-opt 改进，用于在当前选择的边集 E_sel 上做局部搜索优化。

    Args:
        E_sel: 已选择的边集，形如 {(i,j), (k,l), ...}，i<j
        scores: 每条边的打分字典 {(i,j):score}
        max_deg: 每个节点的最大度约束
        max_swaps: 最大迭代次数

    Returns:
        新的边集，经过若干次 2-opt 改进
    """
    E_sel = set(E_sel)  # 拷贝，避免原地修改
    # 统计度数
    deg = [0] * (max(max(u, v) for u, v in E_sel) + 1)
    for (u, v) in E_sel:
        deg[u] += 1
        deg[v] += 1

    def edge_score(e):
        return scores.get(e, 0.0)

    for _ in range(max_swaps):
        improved = False
        # 随机抽两条边尝试交换
        e1, e2 = random.sample(E_sel, 2)
        u1, v1 = e1
        u2, v2 = e2
        # 确保 i<j 规范化
        if u1 > v1: u1, v1 = v1, u1
        if u2 > v2: u2, v2 = v2, u2

        # 尝试 cross 方式：(u1,u2),(v1,v2)
        new_edges = [(min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))]
        # 保证不重复、不自环
        if new_edges[0] in E_sel or new_edges[1] in E_sel:
            continue
        if new_edges[0][0] == new_edges[0][1] or new_edges[1][0] == new_edges[1][1]:
            continue

        # 检查度约束
        deg_copy = deg[:]
        valid = True
        for u, v in [e1, e2]:
            deg_copy[u] -= 1
            deg_copy[v] -= 1
        for u, v in new_edges:
            deg_copy[u] += 1
            deg_copy[v] += 1
            if deg_copy[u] > max_deg[u] or deg_copy[v] > max_deg[v]:
                valid = False
                break
        if not valid:
            continue

        # 计算分数提升
        old_score = edge_score(e1) + edge_score(e2)
        new_score = sum(edge_score(e) for e in new_edges)
        if new_score > old_score:
            # 接受交换
            E_sel.remove(e1)
            E_sel.remove(e2)
            E_sel.update(new_edges)
            deg = deg_copy
            improved = True

        if not improved:
            continue

    return E_sel
