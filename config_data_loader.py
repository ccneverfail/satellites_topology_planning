# config_data_loader.py


# 数据路径配置
# SATELLITE_POSITION_FILE = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\sat_position_data_v1.json"
SATELLITE_POSITION_FILE = './data/sat_position_data_v2.json'

# ACCESS_FILE = {
#     'LEO-LEO': r'D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\access_leo_leo.json',
#     'MEO-MEO': r'D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\access_meo_meo.json',
#     'MEO-LEO': r'D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\access_meo_leo.json',
#     'GEO-ALL': r'D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\access_geo_other_v1.json',
#     'GS-ALL': r'D:\phd\projectWork\pythonProject\topology_optimization\sagin\stk12_based_stk11_demo\data\demo_sagin\access_facility_all_v1.json'
# }
ACCESS_FILE = {
    'LEO-LEO': './data/access_data/access_leo_leo.json',
    'MEO-MEO': './data/access_data/access_meo_meo.json',
    'MEO-LEO': './data/access_data/access_meo_leo.json',
    'GEO-ALL': './data/access_data/access_geo_other_v1.json',
    'GS-ALL': './data/access_data/access_facility_all_v1.json'
}

# ACCESS_MATRIX_NPY = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin_topology_optimization\data\vis_matrix_all.npy"
ACCESS_MATRIX_NPY = './data/vis_matrix_all.npy'

# 地面站配置
# GROUND_STATIONS = {
#     '三亚': (18.25, 109.51),
#     '西安': (34.34, 108.94),
#     '乌鲁木齐': (43.83, 87.60),
#     '漠河': (52.97, 122.54),
#     '雄安': (38.99, 116.10),
#     '重庆': (29.56, 106.55),
#     '深圳': (22.55, 114.05)
# }
GROUND_STATIONS = {
    "Sanya": (18.25, 109.51),
    "Xian": (34.34, 108.94),
    "Urumqi": (43.83, 87.60),
    "Mohe": (52.97, 122.54),
    "XiongAn": (38.99, 116.10),
    "Chongqing": (29.56, 106.55),
    "Shenzhen": (22.55, 114.05)
}

ELEVATION_LIMIT_DEG = 20


#场景仿真开始时间
START_TIME = "22 May 2025 00:00:00.000000"
END_TIME = "23 May 2025 00:00:00.000000"

TIME_STEPS = 1441

SAT_NUM = 723
GS_NUM = len(GROUND_STATIONS) #7

NODE_NUM = SAT_NUM + GS_NUM



# USER_POSITION_FILE = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin\user_traffic\results\user_1000_lonlat.csv"
USER_POSITION_FILE = './data/user_traffic/results/user_1000_lonlat.csv'

# USER_TRAFFIC_FILE = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin\user_traffic\results\user_1000_traffic.npy"
USER_TRAFFIC_FILE = './data/user_traffic/results/user_1000_traffic.npy'


# USER_COVERAGE_LEO_FILE = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin\coverage\results\user_nearest_LEO_satellite.csv"# 信息更全
# USER_COVERAGE_LEO_NPY = r"D:\phd\projectWork\pythonProject\topology_optimization\sagin\coverage\results\user_cover_leo_matrix.npy"
USER_COVERAGE_LEO_CSV = './data/user_cover_leo.csv'
# 仅仅连接关系


