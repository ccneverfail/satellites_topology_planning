
# -*- coding: utf-8 -*-
"""
几何/图工具：
- 经纬高 -> 直线距离（ECEF 近似）
- 快速评估指标（Conn/AvgHop/AvgDelay/Switch）
- 构建基准图（骨干+上分钟仍可见）→ 度/分量
- rem_vis（未来剩余可见分钟）
"""
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set

C_LIGHT = 299_792_458.0  # m/s
R_EARTH = 6371000.0      # m

def llh_to_ecef_distance_m(lat1, lon1, alt1_m, lat2, lon2, alt2_m) -> float:
    la1, lo1 = math.radians(lat1), math.radians(lon1)
    la2, lo2 = math.radians(lat2), math.radians(lon2)
    r1 = R_EARTH + alt1_m
    r2 = R_EARTH + alt2_m
    x1 = r1 * math.cos(la1) * math.cos(lo1)
    y1 = r1 * math.cos(la1) * math.sin(lo1)
    z1 = r1 * math.sin(la1)
    x2 = r2 * math.cos(la2) * math.cos(lo2)
    y2 = r2 * math.cos(la2) * math.sin(lo2)
    z2 = r2 * math.sin(la2)
    dx,dy,dz = x1-x2, y1-y2, z1-z2
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def largest_component_ratio(E:Set[Tuple[int,int]], N:int) -> float:
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)
    if G.number_of_nodes()==0: return 0.0
    comps = list(nx.connected_components(G))
    if not comps: return 0.0
    return max(len(c) for c in comps)/N

def avg_hop_delay(E:Set[Tuple[int,int]], N:int, dist_m:Dict[Tuple[int,int], float],
                  num_anchors:int=80) -> Tuple[float, float]:
    if N==0: return 0.0, 0.0
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (u,v) in E:
        d = dist_m.get((u,v), dist_m.get((v,u), 1.0))
        G.add_edge(u,v,weight=d/C_LIGHT)
    if G.number_of_edges()==0: return 0.0, 0.0
    anchors = np.random.choice(N, size=min(num_anchors, N), replace=False)
    total_hop = 0.0
    total_delay = 0.0
    pairs = 0
    for a in anchors:
        length, path = nx.single_source_dijkstra(G, a, weight='weight')
        for b, delay in length.items():
            if b==a: continue
            total_delay += delay
            total_hop += max(0, len(path[b])-1)
            pairs += 1
    if pairs==0: return 0.0, 0.0
    return total_hop/pairs, total_delay/pairs

def switch_count(E_t:Set[Tuple[int,int]], E_tm1:Set[Tuple[int,int]]) -> int:
    diff = (E_t - E_tm1) | (E_tm1 - E_t)
    return len(diff)

class DSU:
    def __init__(self, n):
        self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b: return
        if self.r[a]<self.r[b]: a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]: self.r[a]+=1

def build_base_graph(N:int,
                     E_core:Set[Tuple[int,int]],
                     E_prev:Set[Tuple[int,int]],
                     visible:Set[Tuple[int,int]]):
    adj=[set() for _ in range(N)]
    E_base=set()
    def add(u,v):
        if u==v: return
        if u>v: u,v=v,u
        if (u,v) in E_base: return
        if (u,v) not in visible: return
        E_base.add((u,v)); adj[u].add(v); adj[v].add(u)
    for (u,v) in E_core: add(u,v)
    for (u,v) in E_prev: add(u,v)
    deg_base=[len(adj[i]) for i in range(N)]
    dsu=DSU(N)
    for (u,v) in E_base: dsu.union(u,v)
    comp_id=[dsu.find(i) for i in range(N)]
    return deg_base, comp_id, E_base

def build_keep_graph(N:int,
                     E_core:Set[Tuple[int,int]],
                     E_prev:Set[Tuple[int,int]],
                     visible:Set[Tuple[int,int]]):

    E_keep=set(E_core)
    def add(u,v):
        if u==v: return
        if u>v: u,v=v,u
        if (u,v) in E_keep: return
        if (u,v) not in visible: return
        E_keep.add((u,v))
    # for (u,v) in E_core: add(u,v)
    for (u,v) in E_prev: add(u,v)
    return E_keep

def remove_E_core_graph(E_core:Set[Tuple[int,int]],
                        E_t:Set[Tuple[int,int]]):
    E_interlayer=set(E_t)
    def remove(u,v):
        if u==v: return
        if u>v: u,v=v,u
        if (u,v)  not in E_interlayer: return
        E_interlayer.remove((u,v))
    for (u,v) in E_core: remove(u,v)
    return E_interlayer

def build_future_rem_visibility(vis:np.ndarray, t:int) -> np.ndarray:
    T, N, _ = vis.shape
    out = np.zeros((N, N), dtype=np.int32)
    cur = vis[t]
    for i in range(N):
        for j in range(i+1, N):
            if cur[i, j] == 0:
                continue
            rem = 0
            for tau in range(t, T):
                if vis[tau, i, j] == 1:
                    rem += 1
                else:
                    break
            out[i, j] = rem
    return out
