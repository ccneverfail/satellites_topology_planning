
# -*- coding: utf-8 -*-
"""
滑窗前瞻（只在动作层用于“候选选优”，不进入训练目标）：
- 给若干候选拓扑 E_t，利用未来 W 分钟可视矩阵，做快速前滚评估（Conn/Delay/Hop/Switch）
- 仅执行“当下最佳”的候选
"""
from typing import Dict, Tuple, Set, List
import numpy as np
from topo_planner_pkg_v2.topo.graph_utils import largest_component_ratio, avg_hop_delay, switch_count
from topo_planner_pkg_v2.config.config import cfg

def fast_rollout_score(E0:Set[Tuple[int,int]],
                       vis_seq:np.ndarray,
                       dist_m:Dict[Tuple[int,int],float],
                       max_deg:List[int]) -> float:
    E_prev = set()
    E = set(E0)
    N = vis_seq.shape[1]
    total = 0.0
    for tau in range(vis_seq.shape[0]):
        V = vis_seq[tau]
        E = {(u,v) for (u,v) in E if V[min(u,v), max(u,v)]==1}
        conn = largest_component_ratio(E, N)
        hop, delay = avg_hop_delay(E, N, dist_m, num_anchors=80)
        sw = switch_count(E, E_prev) if tau>0 else 0
        r = (cfg.R_A1_CONN * conn
             - cfg.R_A2_DELAY * delay
             - cfg.R_A3_HOP * hop
             - cfg.R_A4_SWITCH * sw)
        total += (cfg.GAMMA**tau)*r
        E_prev = set(E)
    return total
