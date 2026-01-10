# topo/intra_backbone.py
from typing import Dict, List, Set, Tuple, Iterable
from collections import defaultdict

def _id_pairs_to_local_pairs(
    id_pairs: Iterable[Tuple[int,int]],
    id2local: Dict[int, int],
    *,
    strict: bool = False,      # True: 碰到不在 nodes 里的 id 直接报错
    ignore_self: bool = True,  # 忽略自环
    normalize: bool = True,    # 归一化为 (min,max)
) -> Set[Tuple[int,int]]:
    out: Set[Tuple[int,int]] = set()
    for a, b in id_pairs:
        if a not in id2local or b not in id2local:
            if strict:
                raise KeyError(f"id {a} 或 {b} 不在当前 nodes 中")
            else:
                continue
        u, v = id2local[a], id2local[b]
        if ignore_self and u == v:
            continue
        if normalize and u > v:
            u, v = v, u
        out.add((u, v))
    return out

def merge_extra_edges_by_id(
    E_fixed: Set[Tuple[int,int]],
    nodes,
    id_pairs: Iterable[Tuple[int,int]],
    *,
    strict: bool = False
) -> Set[Tuple[int,int]]:
    id2local = {n.id: i for i, n in enumerate(nodes)}
    extra_local = _id_pairs_to_local_pairs(id_pairs, id2local, strict=strict)
    return E_fixed | extra_local

# # 使用：
# E_fixed = build_intra_backbone(nodes, layer_map, plane_map)
# E_fixed = merge_extra_edges_by_id(E_fixed, nodes, [(101,205), (205,309)])

def build_intra_backbone(nodes, layer_map: Dict[int, str], plane_map: Dict[int, int]) -> Set[Tuple[int,int]]:
    """
    构造“层内固定骨干网” E_fixed（忽略可见性）：
      - GEO：相邻成环
      - MEO/LEO： (1) 同平面相邻成环；(2) 相邻平面相同编号互连
    返回：局部索引（按 nodes 的顺序）下的无向边集 {(u,v), ...}
    """
    # 1) 把局部索引 -> 全局 id 做个映射
    local2id = {i: n.id for i, n in enumerate(nodes)}

    # print("local2id:", local2id)

    # 2) 先按层、再按平面分组（局部索引）
    layer_groups = defaultdict(list)     # {layer: [local_idx]}
    plane_groups = defaultdict(list)     # {(layer, plane): [local_idx]}

    for i, n in enumerate(nodes):
        L = layer_map[local2id[i]]
        P = plane_map.get(local2id[i], -1)
        layer_groups[L].append(i)
        # print("layer_groups[L]:", layer_groups[L])
        if L in ("LEO", "MEO"):   # 对 LEO/MEO 再做平面细分
            plane_groups[(L, P)].append(i)
        # print("plane_groups[(L, P)]:", plane_groups[(L, P)])



    E_fixed: Set[Tuple[int,int]] = set()

    def add(u,v):
        if u==v: return
        if u>v: u,v=v,u
        if (u,v) in E_fixed: return
        E_fixed.add((u,v))

    # 3) GEO：相邻成环，GEO11-GEO16,id分别为0-5

    # E_fixed.add((0, 1))
    add(0, 1)
    # E_fixed.add((1, 2))
    add(1, 2)
    # E_fixed.add((2, 3))
    add(2, 3)
    # E_fixed.add((3, 4))
    add(3, 4)
    # E_fixed.add((4, 5))
    add(4, 5)
    # E_fixed.add((0, 5))
    add(0, 5)

    # if "GEO" in layer_groups:
    #     idxs = sorted(layer_groups["GEO"])  # 用局部索引排序
    #     m = len(idxs)
    #     if m >= 2:
    #         for k in range(m):
    #             u = idxs[k]; v = idxs[(k+1) % m]
    #             if u > v: u, v = v, u
    #             # E_fixed.add((u, v))
    #             add(u, v)

    # 4) MEO/LEO：同平面成环 + 邻面同序号连边
    for L in ("GEO","MEO", "LEO"):
        # 收集层内的所有平面号
        planes = sorted({P for (LL, P) in plane_groups.keys() if LL == L})
        # 4.1 同平面相邻成环
        for P in planes:
            idxs = sorted(plane_groups[(L, P)])  # 用局部索引的升序代表“同平面内部编号”
            m = len(idxs)
            if m >= 2:
                for k in range(m):
                    u = idxs[k]; v = idxs[(k+1) % m]
                    # if u > v: u, v = v, u
                    # E_fixed.add((u, v))
                    add(u, v)

        # 4.2 邻面相同序号互连（mesh）——让平面首尾相接
        nplanes = len(planes)
        if nplanes >= 2:
            for i in range(nplanes):
                P = planes[i]
                Pn = planes[(i + 1) % nplanes]  # 环接：最后一个连回第一个
                A = sorted(plane_groups[(L, P)])
                B = sorted(plane_groups[(L, Pn)])
                m = min(len(A), len(B))
                for k in range(m):
                    u, v = A[k], B[k]
                    # if u > v: u, v = v, u
                    # E_fixed.add((u, v))
                    add(u, v)


    # E_fixed = merge_extra_edges_by_id(E_fixed, nodes, [(0, 5), (0, 1),(1, 2), (2, 3), (3, 4), (4, 5)])

    return E_fixed
