# data/state.py
# -*- coding: utf-8 -*-
"""
结构化状态对象：
- 采用“本地索引”（0..N-1）来与可视矩阵一致，避免 sat 原始 id 与矩阵索引不一致的问题
"""
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from data.loader_adapter import Node, EdgeCand, build_edge_candidates_at_t,build_edge_candidates_interlayer_sgl
from topo.graph_utils import build_base_graph

@dataclass
class State:
    t: int                                   # 当前分钟
    nodes: List[Node]                        # 当分钟节点列表（顺序定义本地索引 0..N-1）
    visible_pairs: Set[Tuple[int,int]]       # {(i,j) | i<j 且可视}
    cand_edges: List[EdgeCand]               # 候选边（来自可视矩阵，附距离/剩余可见时长）
    E_prev: Set[Tuple[int,int]]              # 上一分钟已连边（本地索引）
    E_core: Set[Tuple[int,int]]              # 先验骨干边+上一时刻仍可见的边,其实是E_base
    layer_map: Dict[int,str]                 # 本地索引 -> 层级（"LEO"/"MEO"/"GEO"/"GS"）
    plane_map: Dict[int,int]                 # 本地索引 -> 轨道面编号（非LEO可为-1）
    deg_base: List[int]                      # 基准图度
    comp_id: List[int]                       # 基准图连通分量 id

@dataclass
class StateInterLayer:
    t: int                                   # 当前分钟
    nodes: List[Node]                        # 当分钟节点列表（顺序定义本地索引 0..N-1）
    visible_pairs: Set[Tuple[int,int]]       # {(i,j) | i<j 且可视}
    cand_edges: List[EdgeCand]               # 候选边（来自可视矩阵，附距离/剩余可见时长）
    E_prev: Set[Tuple[int,int]]              # 上一分钟已连边（本地索引）
    E_core: Set[Tuple[int,int]]              # 先验骨干边+上一时刻仍可见的边,其实是E_base
    layer_map: Dict[int,str]                 # 本地索引 -> 层级（"LEO"/"MEO"/"GEO"/"GS"）
    plane_map: Dict[int,int]                 # 本地索引 -> 轨道面编号（非LEO可为-1）
    deg_base: List[int]                      # 基准图度
    comp_id: List[int]                       # 基准图连通分量 id
    deg_isl_used: List[int]           # 由 E_fixed 统计出来的 ISL 已占用度

def count_isl_degree_from_fixed(E_fixed:Set[Tuple[int,int]], nodes, layer_map)->List[int]:
    deg = [0]*len(nodes)
    for (u,v) in E_fixed:
        Lu = layer_map[u]; Lv = layer_map[v]
        if Lu!="GS" and Lv!="GS":   # 只有卫星-卫星的固定边算 ISL
            deg[u]+=1; deg[v]+=1
    return deg



def make_state(db, t:int,
               E_prev:Set[Tuple[int,int]],
               E_core:Set[Tuple[int,int]] = frozenset()) -> State:
    """
    构造某分钟 t 的结构化状态：
    - 使用本地索引（0..N-1）与 vis_matrix[t] 对齐
    - 计算 visible_pairs、候选边、基准图的 deg/comp
    """
    nodes = db.nodes_by_t[t]
    N = len(nodes)
    V_t = db.vis_matrix[t]  # [N,N] 0/1

    # 1) 本地可视边集合（i<j）
    visible_pairs = {(i, j) for i in range(N) for j in range(i+1, N) if V_t[i, j] == 1}

    # 2) 候选边（带距离/未来可视分钟）
    cand_edges = build_edge_candidates_at_t(db, t)

    # 3) 本地层级/轨道面映射（用 nodes 的顺序定义本地索引）
    layer_map = {i: nodes[i].layer for i in range(N)}
    plane_map = {i: nodes[i].plane for i in range(N)}

    # 4) 基准图（骨干+上分钟仍可见）→ 度/分量
    deg_base, comp_id, E_base = build_base_graph(
        N, set(E_core), set(E_prev), visible_pairs
    )


    return State(
        t=t,
        nodes=nodes,
        visible_pairs=visible_pairs,
        cand_edges=cand_edges,
        E_prev=set(E_prev),
        E_core=set(E_core),
        layer_map=layer_map,
        plane_map=plane_map,
        deg_base=deg_base,
        comp_id=comp_id,
    )

def make_state_interlayer(db, t:int,
               E_prev:Set[Tuple[int,int]],
               E_core:Set[Tuple[int,int]] = frozenset()) -> StateInterLayer:
    """
    构造某分钟 t 的结构化状态：
    - 使用本地索引（0..N-1）与 vis_matrix[t] 对齐
    - 计算 visible_pairs、候选边、基准图的 deg/comp
    """
    nodes = db.nodes_by_t[t]
    N = len(nodes)
    V_t = db.vis_matrix[t]  # [N,N] 0/1

    # 1) 本地可视边集合（i<j）
    visible_pairs = {(i, j) for i in range(N) for j in range(i+1, N) if V_t[i, j] == 1}

    # 2) 候选边（带距离/未来可视分钟）
    # cand_edges = build_edge_candidates_at_t(db, t)
    cand_edges = build_edge_candidates_interlayer_sgl(db, t)

    # 侯选边需要满足可视


    # 3) 本地层级/轨道面映射（用 nodes 的顺序定义本地索引）
    layer_map = {i: nodes[i].layer for i in range(N)}
    plane_map = {i: nodes[i].plane for i in range(N)}

    # 4) 基准图（骨干+上分钟仍可见）→ 度/分量
    deg_base, comp_id, E_base = build_base_graph(
        N, set(E_core), set(E_prev), visible_pairs
    )

    deg_isl_used = count_isl_degree_from_fixed(E_core, db.nodes_by_t[t], db.layer_map)


    return StateInterLayer(
        t=t,
        nodes=nodes,
        visible_pairs=visible_pairs,
        cand_edges=cand_edges,
        E_prev=set(E_prev),
        E_core=set(E_core),
        layer_map=layer_map,
        plane_map=plane_map,
        deg_base=deg_base,
        comp_id=comp_id,
        deg_isl_used=deg_isl_used,
    )

