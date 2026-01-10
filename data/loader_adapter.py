
# -*- coding: utf-8 -*-
"""
数据加载适配器：
- 优先尝试导入你现有的 data_loader.py 并使用其接口（若可用）
- 否则，提供最小的占位实现/读取 npy 的可视矩阵
- 负责把“每分钟候选边”构造成 EdgeCand（含 dist, rem_vis）
"""
import importlib
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from config.config import cfg
from topo.graph_utils import llh_to_ecef_distance_m, build_future_rem_visibility

# ---- 尝试加载你现有的 data_loader.py（若同目录或 PYTHONPATH 可见）
_dl_mod = None
try:
    _dl_mod = importlib.import_module("data_loader")
    print("[INFO] Using existing data_loader.py for loading. ")
except Exception as _:
    _dl_mod = None

@dataclass
class Node:
    id: int
    layer: str      # "LEO"/"MEO"/"GEO"/"GS"
    plane: int      # 轨道面编号，非 LEO 可设 -1
    lat: float
    lon: float
    alt_km: float

@dataclass
class EdgeCand:
    u: int
    v: int
    dist_m: float
    rem_vis_min: float
    visible: bool = True

class DataBundle:
    """把训练需要的关键数据打成包"""
    def __init__(self):
        self.time_steps = 0
        self.nodes_by_t: Dict[int, List[Node]] = {}
        self.vis_matrix: np.ndarray = None   # [T, N, N] 上三角 0/1
        self.layer_map: Dict[int, str] = {}
        self.plane_map: Dict[int, int] = {}
        self.sat_id_map: Dict[int, int] = {}

def load_everything() -> DataBundle:
    db = DataBundle()
    _dl_mod = importlib.import_module("data_loader")
    print("[INFO] Using existing data_loader.py for loading. ")
    if _dl_mod is not None:
        satellites_data, sat_id_map = _dl_mod.load_satellites()   # data_loader已定义
        last_sat_id, _ = list(sat_id_map.items())[-1]
        ground_stations, gs_id_map = _dl_mod.load_ground_stations(last_sat_id)
        sat_id_map.update(gs_id_map)
        db.sat_id_map = sat_id_map

        vis_matrix = _dl_mod.load_access(sat_id_map=sat_id_map,
                                         output_npy_path=cfg.ACCESS_MATRIX_NPY)

        vis_matrix = vis_matrix[cfg.DATA_T_START:(cfg.DATA_T_END+1),:,:]

        db.vis_matrix = vis_matrix.astype(np.uint8)
        db.time_steps = db.vis_matrix.shape[0]

        for t in range(len(satellites_data)):
            nodes_t = []
            for sat in satellites_data[t]:
                name = str(getattr(sat, "name", ""))
                if name.startswith("LEO"):
                    layer, plane = "LEO", safe_plane_id_from_name(name)
                elif name.startswith("MEO"):
                    layer, plane = "MEO", safe_plane_id_from_name_MEO_GEO(name)
                elif name.startswith("GEO"):
                    layer, plane = "GEO", safe_plane_id_from_name_MEO_GEO(name)
                else:
                    layer, plane = "Facility", -1
                nodes_t.append(Node(
                    id=int(sat.id), layer=layer, plane=plane,
                    lat=float(sat.lat), lon=float(sat.lon), alt_km=float(sat.alt)
                ))
            for gs_id, gs_name in gs_id_map.items():
                lat, lon = _dl_mod.GROUND_STATIONS[gs_name]
                nodes_t.append(Node(id=int(gs_id), layer="GS", plane=-1,
                                    lat=float(lat), lon=float(lon), alt_km=0.0))
            db.nodes_by_t[t] = nodes_t

        layers = {}
        planes = {}
        for n in db.nodes_by_t[0]:
            layers[n.id] = n.layer
            planes[n.id] = n.plane
        db.layer_map = layers
        db.plane_map = planes

    else:
        print("[WARN] data_loader.py not found, using simplified loader with npy.")
        vis = np.load(cfg.ACCESS_MATRIX_NPY)   # 形状 [T,N,N]
        db.vis_matrix = vis.astype(np.uint8)
        db.time_steps = vis.shape[0]
        N = vis.shape[1]
        nodes = [Node(i, "LEO", -1, lat=0.0, lon=0.0, alt_km=550.0) for i in range(N)]
        for t in range(db.time_steps):
            db.nodes_by_t[t] = nodes
        db.layer_map = {i:"LEO" for i in range(N)}
        db.plane_map = {i:-1 for i in range(N)}

    return db

def safe_plane_id_from_name(name:str)->int:
    try:
        return int(name[3:5])  # 例如 LEO2310 -> 23
    except:
        return -1

def safe_plane_id_from_name_MEO_GEO(name:str)->int:
    try:
        return int(name[3:4])  # 例如 MEO23 -> 2
    except:
        return -1

def build_edge_candidates_at_t(db:DataBundle, t:int) -> List[EdgeCand]:
    """把本分钟可视矩阵中为1的上三角(i<j)对转成 EdgeCand，补充 dist_m、rem_vis"""
    V = db.vis_matrix[t]  # [N,N]
    nodes = db.nodes_by_t[t]
    N = len(nodes)
    edge_list: List[EdgeCand] = []
    rem_vis = build_future_rem_visibility(db.vis_matrix, t)
    for i in range(N):
        for j in range(i+1, N):
            if V[i, j] == 1:
                a = nodes[i]; b = nodes[j]
                dist_m = llh_to_ecef_distance_m(a.lat, a.lon, a.alt_km*1000.0,
                                                b.lat, b.lon, b.alt_km*1000.0)
                edge_list.append(EdgeCand(
                    u=i, v=j, dist_m=dist_m, rem_vis_min=float(rem_vis[i, j])
                ))
    return edge_list

def build_edge_candidates_interlayer_sgl(db:DataBundle, t:int) -> List[EdgeCand]:
    """只对“层间 & 星地”构建候选边与特征"""
    nodes = db.nodes_by_t[t]
    V = db.vis_matrix[t]
    layer = db.layer_map
    N = len(nodes)
    edge_list: List[EdgeCand] = []
    rem_vis = build_future_rem_visibility(db.vis_matrix, t)
    for i in range(N):
        for j in range(i+1, N):
            if V[i,j] != 1:
                continue
            a = nodes[i]
            b = nodes[j]
            dist_m = llh_to_ecef_distance_m(a.lat, a.lon, a.alt_km * 1000.0,
                                            b.lat, b.lon, b.alt_km * 1000.0)
            Li, Lj = layer[i], layer[j]
            # 仅层间 ISL 或 SGL
            is_sgl = (Li=="GS") ^ (Lj=="GS")
            is_interlayer_isl = (Li != Lj) and (Li!="GS") and (Lj!="GS")
            if not (is_sgl or is_interlayer_isl):
                continue
            # 构建 EdgeCand
            edge_list.append(EdgeCand(
                u=i, v=j, dist_m=dist_m, rem_vis_min=float(rem_vis[i, j])
                ))
    return edge_list

