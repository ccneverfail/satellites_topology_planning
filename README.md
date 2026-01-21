
# satellites_topology_planning：Generalized Multi-kernel Correntropy-based Deep Reinforcement Learning for Dynamic Topology Planning in Multi-Layered Satellite Network

<video src=https://github.com/user-attachments/assets/0f92ca61-23d6-422a-b5f7-966f74158fef" controls></video>


https://github.com/user-attachments/assets/0f92ca61-23d6-422a-b5f7-966f74158fef


- 自适应 CUDA/AMP
- Dueling Double DQN（边级 Q） + b-匹配
- n-step训练目标
- 滑窗前瞻用于**动作层**（可在 `planner/rollout.py` 扩展）
- 指标记录 CSV，批量写入并硬刷新

## 目录
- `config/config.py`：常量/超参数（例如 MAX_LINK、窗口 W 等）
- `data/loader_adapter.py`：数据加载（优先尝试引用 `data_loader.py`，否则用 npy）
- `topo/*`：拓扑相关工具（conn_gain、bmatch、图指标等）
- `models/edge_q.py`：Q 网络
- `rl/replay.py`：序列回放（n-step）
- `planner/rollout.py`：滑窗前瞻评分（仅动作层使用）
- `utils/*`：日志/随机种子
- `train.py`：训练入口（模式 A）

## 依赖
```
pip install torch numpy networkx
```

## 运行
```
python train.py
```

> 如需使用你的 `data_loader.py` 与“可视时间窗口 JSON”，请将其放在 PYTHONPATH 下，并在 `data/loader_adapter.py` 中按你的字段适配。
