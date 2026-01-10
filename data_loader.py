# data_loader.py

from datetime import datetime

from pyproj import Transformer
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from config_data_loader import SATELLITE_POSITION_FILE, ACCESS_FILE, USER_TRAFFIC_FILE, GROUND_STATIONS, START_TIME, TIME_STEPS, NODE_NUM
from config_data_loader import USER_POSITION_FILE, USER_COVERAGE_LEO_CSV


#定义Satellite对象
class Satellite:
    def __init__(self, sat_id, name, lat, lon, alt):
        self.id = sat_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def __repr__(self):#自定义名片，打印用
        return f"{self.name} (ID {self.id}): ({self.lat:.2f}, {self.lon:.2f}, {self.alt:.1f} km)"



class GroundStation:
    def __init__(self, id, name, lat, lon):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = 0.0

    def __repr__(self):
        return f"{self.name} (ID {self.id}): ({self.lat:.2f}, {self.lon:.2f})"




# 地心直角 (ECEF) → 经纬度高程 (WGS84)
transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)




def ecef_to_llh(x, y, z):
    lon, lat, alt = transformer.transform(x*1000, y*1000, z*1000)  # km → m
    alt /= 1000  # m → km
    return lat, lon, alt

def load_satellites():
    '''
    :param:SATELLITE_POSITION_FILE
    {
      "start_time": "...",
      "stop_time": "...",
      "time_step": 60,
      "count": 1441,
      "positions": {
        "GEO11: {
          "pos": [[x1, y1, z1], [x2, y2, z2], ...]  # 长度1441
        },
        "GEO12": {
          "pos": [[x1, y1, z1], [x2, y2, z2], ...]
        },
        ...
      }
    }

    :return:所有时刻卫星信息satellites[t][],从卫星名到ID的映射sat_id_map
    '''
    with open(SATELLITE_POSITION_FILE, 'r') as f:
        data = json.load(f)

    positions = data["positions"]
    time_steps = data["count"]

    sat_names = sorted(positions.keys())  # 保持顺序一致
    num_sats = len(sat_names)

    # TODO: sat_id_map
    #  GEO:0
    #  GEO1-GEO6:[1:7)
    #  LEO:7
    #  LEO0101-LEO2330:[8:698)
    #  MEO:698
    #  MEO11-MEO38:[699:723)

    # sat_id_map = {name: idx for idx, name in enumerate(sat_names)}  #  从卫星名到ID的映射  name: id
    sat_id_map = {idx: name for idx, name in enumerate(sat_names)}  #  从卫星名到ID的映射  id: name
    satellites = {}  # {t: [Satellite, ...]}

    for t in range(time_steps):
        sats_this_step = []

        for sat_id, sat_name in enumerate(sat_names):
            try:
                x, y, z = positions[sat_name]["pos"][t]
                lat, lon, alt = ecef_to_llh(x, y, z)
                sats_this_step.append(Satellite(sat_id, sat_name, lat, lon, alt))#存入Satellite对象
            except Exception as e:
                print(f"[ERROR] {sat_name} @ t={t}: {e}")

        satellites[t] = sats_this_step

    # timestamps = list(satellites.keys())
    print(f"[OK] Loaded {time_steps} time steps × {num_sats} satellites.")
    # return satellites, time_steps, sat_names
    return satellites, sat_id_map




def parse_time_str(t_str, start_dt):
    t = datetime.fromisoformat(t_str)
    return int((t - start_dt).total_seconds() // 60)  # 每分钟一帧

def parse_time_str_v2(t_str, start_dt):
    # 处理格式："22 May 2025 00:00:00.000000"
    t = datetime.strptime(t_str, "%d %b %Y %H:%M:%S.%f")
    return int((t - start_dt).total_seconds() // 60)


def fill_visibility_matrix(vis_json, sat_id_map, vis_matrix, start_time_str):
    start_dt = datetime.strptime(start_time_str, "%d %b %Y %H:%M:%S.%f")

    for record in vis_json["access_data"]:
        src = record["src"]
        dst = record["dst"]

        if src not in sat_id_map or dst not in sat_id_map:
            continue

        src_id = sat_id_map[src]
        dst_id = sat_id_map[dst]

        for t_start_str, t_end_str in record["access_interval"]:
            try:
                t_start = parse_time_str_v2(t_start_str, start_dt)
                t_end = parse_time_str_v2(t_end_str, start_dt)

                for t in range(max(t_start, 0), min(t_end + 1, vis_matrix.shape[0])):
                    i, j = sorted((src_id, dst_id))  # 自动获得 i < j 的顺序
                    if i != j:
                        vis_matrix[t, i, j] = 1

            except Exception as e:
                print(f"[ERROR] Bad time interval for {src}->{dst}: {e}")



# 读取卫星的可见性窗口
def load_access(sat_id_map, output_npy_path):
    '''
    ACCESS_FILE.items()的json文件格式：
    {
      "time_interval": "22 May 2025 00:00:00.000000/23 May 2025 00:00:00.000000",
      "access_data": [
        {
          "src": "MEO",
          "dst": "MEO11",
          "access_interval": [
            ["22 May 2025 00:00:00.000000", "23 May 2025 00:00:00.000000"]
          ]
        },
        ...
      ]
    }

    :param sat_id_map:不仅satellites还包括ground_stations
    :return:
    '''


    if os.path.exists(output_npy_path):
        print(f"[INFO] Found ：{output_npy_path}, reading...")
        # vis_matrix =  np.load(output_npy_path)
        # 尝试用内存映射方式读取，避免一次性分配整块内存
        try:
            vis_matrix = np.load(output_npy_path, mmap_mode='r')  # 关键：mmap
        except MemoryError:
            # 极少数情况下 np.load+mmap 也会触发异常，就退回到 np.memmap
            file_size = os.path.getsize(output_npy_path)
            print(
                f"[warn] np.load with mmap failed; fallback np.memmap. file={output_npy_path}, size={file_size / 1024 / 1024:.1f} MiB")
            vis_matrix = np.memmap(output_npy_path, mode='r')  # 若是 .npy，仍可映射
        # （通常 .npy 都包含形状信息，不需手动 reshape；若你历史文件是扁平一维，可在此按 (T,N,N) reshape）

        return vis_matrix

    print(f"[INFO] Not found, vis_matrix is computing...")

    # 初始化矩阵
    vis_matrix = np.zeros((TIME_STEPS, NODE_NUM, NODE_NUM), dtype=np.uint8)

    # vis_json = {}
    for k, p in tqdm(ACCESS_FILE.items(), desc="Loading access files"):
    # for k, p in ACCESS_FILE.items():
        print(f"[INFO] Loading access file {k}...")
        with open(p, 'r') as f:
            vis_json = json.load(f)
            start_time = START_TIME
            fill_visibility_matrix(vis_json, sat_id_map, vis_matrix, start_time)

    np.save(output_npy_path, vis_matrix)
    print(f"[INFO] vis_matrix is computed, and saved to：{output_npy_path}")
    return vis_matrix



# 读取地面站
# def load_ground_stations():
#     stations = []
#     for name, (lat, lon) in GROUND_STATIONS.items():
#         stations.append(GroundStation(name, lat, lon))
#     return stations

# 读取地面站
def load_ground_stations(max_sat_id):
    gs_names = sorted(GROUND_STATIONS.keys())  # 确保顺序一致
    # print(gs_names)
    stations = []
    gs_id_map = {}

    for idx, name in enumerate(gs_names):
        lat, lon = GROUND_STATIONS[name]
        stations.append(GroundStation(idx + max_sat_id + 1, name, lat, lon))
        # gs_id_map[name] = idx + max_sat_id + 1
        gs_id_map[idx + max_sat_id + 1] = name

    print(f"[OK] Loaded {len(stations)} ground stations.")

    return stations, gs_id_map





# 读取用户位置、流量
def load_users_traffic():
    # 加载
    user_positions_df = pd.read_csv(USER_POSITION_FILE)

    user_traffic = np.load(USER_TRAFFIC_FILE, allow_pickle=True)  # 允许反序列化

    # print(f"user_positions: {user_positions_df.head()}")
    # print(f"user_traffic: {user_traffic[2]}")
    return user_positions_df, user_traffic




 # 加载 提供用户覆盖的 LEO
def load_user_cover_leo():
    # 加载用户覆盖DataFrame
    user_cover_leo = pd.read_csv(USER_COVERAGE_LEO_CSV, index_col=0)

    return user_cover_leo


# 将地面用户流量聚合到LEO卫星
def aggregate_user_traffic_to_satellites(user_traffic: np.ndarray, user_cover_leo_df: pd.DataFrame, sat_traffic_path):
    """
    将用户流量聚合到对应的 LEO 卫星上，返回按时间和卫星组织的字典。

    参数:
    - user_traffic: np.ndarray, 形如 [T, N, 4]，每一行为 [user_id, business_type, down_MB, up_MB]
    - user_cover_leo_df: pd.DataFrame，形如 [T, N]，每个元素是用户在每个时刻分配到的卫星名

    返回:
    - sat_traffic_dict: dict[time_id][sat_id] = {"T_down_MB": float, "T_up_MB": float}
    """


    if os.path.exists(sat_traffic_path):
        print(f"检测到已有文件：{sat_traffic_path}，正在读取...")
        data = np.load(sat_traffic_path, allow_pickle=True).item()
        return data
    else:
        print(f"未检测到缓存文件，正在将用户流量聚合至卫星...")

        T, N, _ = user_traffic.shape
        sat_traffic_dict = {}

        for t in range(T):
            sat_traffic_dict[t] = {}

            for i in range(N):
                user_id = user_traffic[t][i][0]
                user_id = str(user_id)  # 确保匹配 DataFrame 的列名

                if user_id not in user_cover_leo_df.columns:
                    continue  # 忽略无对应列的用户

                sat_name = user_cover_leo_df.loc[t, user_id]
                if not isinstance(sat_name, str) or not sat_name.startswith("LEO"):
                    continue  # 忽略无效卫星

                T_down = float(user_traffic[t][i][2])
                T_up = float(user_traffic[t][i][3])

                if sat_name not in sat_traffic_dict[t]:
                    sat_traffic_dict[t][sat_name] = {"T_down_MB": 0.0, "T_up_MB": 0.0}

                sat_traffic_dict[t][sat_name]["T_down_MB"] += T_down
                sat_traffic_dict[t][sat_name]["T_up_MB"] += T_up

        """保存字典为 .npy 文件"""
        np.save(sat_traffic_path, sat_traffic_dict)
        print(f"✅ 卫星流量数据已保存至 {sat_traffic_path}")

        return sat_traffic_dict





if __name__ == '__main__':

    #加载卫星信息
    satellites_data, sat_id_map = load_satellites()
    # print("satellites_data:\n",satellites_data)
    # # print(f"Time step count: {len(satellites_data)}")
    # print(sat_id_map)
    #
    # # 查看第一个时间步的部分卫星
    # for sat in satellites_data[0][:6]:
    #     print(sat)
    #     # print(sat.name)
    #
    # last_sat_name,last_sat_id = list(sat_id_map.items())[-1]
    last_sat_id, last_sat_name = list(sat_id_map.items())[-1]
    # print(f"last_sat_id = {last_sat_id}")



    # #加载地面站
    ground_stations, gs_id_map = load_ground_stations(last_sat_id) # 把地面站的id：name加到sat_id_map中，连续排
    # print(gs_id_map)
    print("ground_stations:\n",ground_stations)


    sat_id_map.update(gs_id_map)

    print("sat_id_map:", sat_id_map)





    # #生成可视矩阵
    # vis_matrix = load_access(sat_id_map)
    # np.save("./data/vis_matrix_all.npy", vis_matrix)
    # print("✅ 可视性矩阵已保存至 ./data/vis_matrix_all.npy")
    # print(vis_matrix.shape)
    # print(vis_matrix)


    # #加载可视矩阵
    access_data = np.load("./data/vis_matrix_all_1.npy")
    print("✅ 已加载 vis_matrix_all.npy，形状：", access_data.shape)
    print(access_data[50][3][696:720])
    # print("vis_matrix:\n", access_data)

    # # #加载用户和流量
    # user_positions_df, user_traffic_matrix = load_users_traffic()
    # print(f"user_positions:\n  {user_positions_df}")
    # print(f"user_traffic: \n {user_traffic_matrix[:3]}")# 三维矩阵user_traffic[time_step]
    #



    '''# 生成/读取用户覆盖matrix，cover_matrix[time_id][user_id]
    cover_matrix = assign_nearest_leo_satellites(
        user_positions_df=user_positions_df,
        satellites_data=satellites_data,
        output_csv_path=USER_COVERAGE_LEO_FILE,
        output_npy_path=USER_COVERAGE_LEO_NPY
    )'''
    # # 示例：访问 time_id=0 时 user_id='U000002' 的匹配卫星名称
    # print(cover_matrix[0]['U000002'])  # 输出如：LEO0102，不能切片


    '''# 加载用户覆盖dict # 该格式用不到
    cover_dict = np.load(USER_COVERAGE_LEO_NPY, allow_pickle=True).item()
    print(cover_dict[0]['U000002'])  # 输出如：LEO0102'''
    # print(cover_matrix==cover_dict)#True



    # # 加载用户覆盖卫星DataFrame
    # user_cover_leo_df = load_user_cover_leo()
    #
    # print(user_cover_leo_df.loc[0:4, 'U000939'])
    # # print(user_cover_leo_df.loc[0:4, :])#取数据的0-3行
    # print(f"user_cover_leo_df.iloc[0:10, :]: \n {user_cover_leo_df.iloc[0:10, :]}")#取数据的0-9行，0-4列（打印的第一列为行号=time_id
    #


    '''# TODO: 将用户流量聚集到卫星
    sat_traffic = aggregate_user_traffic_to_satellites(user_traffic=user_traffic_matrix,
                                                       user_cover_leo_df=user_cover_leo_df,
                                                       sat_traffic_path=SAT_TRAFFIC_FILE
                                                       )

    print(sat_traffic[0])'''