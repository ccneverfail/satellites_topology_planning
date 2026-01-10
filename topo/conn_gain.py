
# -*- coding: utf-8 -*-
"""
conn_gain 方案
"""
from typing import List, Dict

def conn_gain_A(i:int, j:int,
                deg_base:List[int],
                comp_id:List[int],
                layer:Dict[int,str],
                plane:Dict[int,int],
                *,
                b1=1.0, b2=1.2, b3=0.5, b4=0.3, eps=0.5) -> float:
    invdeg = 1.0/(deg_base[i]+eps) + 1.0/(deg_base[j]+eps)   # 逆度项（第一项）：偏好把低度/边缘节点拉进主网，提升连通、降直径。
    cross_comp  = 1.0 if comp_id[i]!=comp_id[j] else 0.0     # 跨分量指示：在“当前已选/骨干” E_core上做并查集；若 i,j 不同分量，连上会立刻合并两块图，给大额奖励。
    cross_layer = 1.0 if layer[i]!=layer[j] else 0.0         # 跨层指示
    cross_plane = 1.0 if (layer[i]=="LEO" and layer[j]=="LEO" and plane[i]!=plane[j]) else 0.0    # 跨面指示
    return b1*invdeg + b2*cross_comp + b3*cross_layer + b4*cross_plane
