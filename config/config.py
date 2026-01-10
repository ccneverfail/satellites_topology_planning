
# -*- coding: utf-8 -*-
"""
集中放置所有常量/超参数，其他模块只 from config import cfg
"""
# config/config.py
from dataclasses import dataclass, field
from typing import Optional,List

@dataclass
class Config:
    # ===== 设备 / 随机性 =====
    USE_AUTO_DEVICE: bool = True
    AMP: bool = True                 #Automatic Mixed Precision（自动混合精度）
    SEED: int = 42

    USE_NUM_WORKERS: bool = False
    NUM_WORKERS: int = 10  # 多进程 rollout 的 worker 个数

    # ===== 训练范围 / 轮次 =====
    DATA_T_START: int = 0            # 用第 0 分钟开始

    DATA_T_END: Optional[int] = 120  # 到第 20 分钟（不含），固定前 20 分钟
    EPOCHS: int = 35                 # 在这 20 分钟上循环 20 轮（约 20*19≈380 步）
    NUM_STEPS: int = 100000           # 全局步数上限，设大些，由上面共同限制

    # ===== DQN / 采样 =====
    BATCH_SIZE: int = 16             #4

    N_STEP: int = 1   #=3,多步回报DQN

    GAMMA: float = 0.96
    HIDDEN_DIM: int = 256

    # ===== 优化器 / 更新 =====
    LR: float = 1e-4                 # B=8 建议 8e-5~1e-4；若不稳可降到 6e-5
    WEIGHT_DECAY: float = 1e-4
    UPDATE_EVERY: int = 1            # 每步都可更新（由经验池门槛把关）
    WARMUP_STEPS: int = 32          # 需 ≥ BATCH_SIZE*(N_STEP+1)=32，取 64 更稳
    TARGET_TAU: float = 0.01         # 目标网络软更新系数

    # ===== ε-greedy 探索 =====
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY_STEPS: int = 100      # 约训练全过程缓慢降温；更长训练可调到 5000+

    # ===== 滑窗前瞻（动作层重排序） =====
    USE_LOOKAHEAD: bool = False
    WINDOW_W: int = 1
    NUM_CANDIDATES: int = 2          # 目前用“q/keep”两种候选，保留参数以备扩展

    # ===== 度约束（按星座实际可改）（train_window_gs.py） 回退使用=====
    MAX_LINK: int = 5
    MAX_DEG_LEO: int = 5
    MAX_DEG_MEO: int = 6
    MAX_DEG_GEO: int = 8
    MAX_DEG_GS: int = 3              #保证 MAX_DEG_GS ≥ 3（因为 GS 要分别连到 LEO/MEO/GEO）

    #  ===== 星间星地度约束分开算 （train_window_gs_isl.py）=====
    #  星间度约束，ISL
    MAX_DEG_ISL_LEO: int = 5
    MAX_DEG_ISL_MEO: int = 6
    MAX_DEG_ISL_GEO: int = 8
    MAX_DEG_ISL_GS: int = 0  # GS 没有 ISL（可置 0）

    #  星地度约束，SGL 在 卫星端占用 GSL 预算
    MAX_DEG_SGL_LEO: int = 3
    MAX_DEG_SGL_MEO: int = 3
    MAX_DEG_SGL_GEO: int = 3
    #  星地度约束，SGL 在 GS 端受限
    MAX_DEG_SGL_GS: int = 3  # 每个地面站最多并发 SGL 条数
    # 每个地面站在星地链路(SGL)上的“分层保底”配额
    # 整个地面站在星地链路(SGL)上的“分层保底”配额，修改于20250930
    REQ_SGL_LEO: int = 1
    REQ_SGL_MEO: int = 1
    REQ_SGL_GEO: int = 1


    # ===== 边特征对MLP打分的（x_e各项）权重 =====
    # 每维边特征的固定权重（与 edge_features 输出维度对应）
    # [inv_delay, rem_vis, conn_gain, is_new, layer_u_enc, layer_v_enc]
    USE_EDGE_FEAT_W: bool = False # 开关
    EDGE_FEAT_W: List[float] = field(
        default_factory=lambda: [1.0,    ## inv_delay，时延倒数，距离越短，时延越小，打分越高。
                                 1.0,    # rem_vis,剩余可见时长（分钟数
                                 10.0,    # conn_gain, 结构性增益
                                 0.02,    # is_new,切换惩罚
                                 1.0,   # layer_u_enc：u的层编码（0、1、2、3）
                                 1.0    # layer_v_enc：v的层编码（0、1、2、3）
                                 ])



    # ===== 连接增益（conn_gain_A）权重，见 topo/conn_gain.py =====
    CG_B1: float = 1.0               # 逆度项 invdeg 系数：1.0  # 偏好把低度/边缘节点拉进主网，提升连通、降直径。
    CG_B2: float = 0.6               # 跨分量连接收益 cross_comp 系数： 0.6   # 若 i,j 属于不同分量，连上会立刻合并两块图，给大额奖励
    CG_B3: float = 0.3               # 跨层指示 cross_layer 系数: 0.3 #
    CG_B4: float = 0.2               # 跨轨道面指示 cross_plane 系数: 0.2
    CG_EPS: float = 1e-6

    # ===== 奖励权重（可按目标微调） =====
    R_A1_CONN: float = 2000.0           # 连通性 2.0
    R_A2_DELAY: float = 1000.0          # 时延惩罚 1.0
    R_A3_HOP: float = 0.3            # 跳数惩罚 0.3
    R_A4_SWITCH: float = 0.08         # 切换惩罚（配合前瞻能明显降切换） 0.8

    # ===== 日志 / 保存 =====
    LOG_DIR: str = "./runs"
    FLUSH_EVERY_STEPS: int = 100     #多少个 step 就把日志文件强制写盘一次（flush+fsync）。值越小，崩溃时丢数据越少，但磁盘 I/O 越频繁。
    SAVE_EVERY_STEPS: int = 10000    #多少个 step 保存一次模型 checkpoint（文件名类似 edge_q_step2000.pt）
    RUN_NAME: Optional[str] = None  #设为 None：自动用时间戳（如 20250819_153012）

    # ===== 存checkpoint（训练快照） 的开关与频率 =====
    SAVE_EVERY_EPOCH: bool = True  # 是否在每个 epoch 末保存
    EPOCH_SAVE_INTERVAL: int = 5  # 每隔多少个 epoch 存一次cheakpoints，方便断点续训（1=每个 epoch 都存）

    # ===== 每步拓扑落盘（NDJSON） =====
    # TODO:eval时保存拓扑用
    SAVE_TOPOLOGY: bool = True
    TOPOLOGY_FILENAME: str = "topology_steps.ndjson"
    TOPOLOGY_FLUSH_EVERY: int = 1

    # ========== 数据路径（使用现有 data_loader） ==========
    ACCESS_MATRIX_NPY: str = "./data/vis_matrix_all_1.npy"  # 可视矩阵


    # ========== 更换train_step的loss ==========
    # --- GL loss ---
    GL_USE: bool = True  # 打开/关闭 GL
    GL_ALPHA: float = 1.2  # 形状参数 α，1.2~2 常用；2 接近 MSE
    GL_BETA_MODE: str = "median"  # "fixed" | "median" | "ema"
    GL_BETA_FIXED: float = 1.0  # 当 GL_BETA_MODE="fixed" 时使用
    GL_BETA_SCALE: float = 2.0  # beta ≈ scale * median(|δ|)
    GL_BETA_EPS: float = 1e-6  # 数值稳定项
    GL_BETA_EMA: float = 0.0  # 若 "ema"，初始化 beta 的 EMA 值（0 表示未启用）
    GL_BETA_EMA_DECAY: float = 0.95


cfg = Config()
