
# satellites_topology_planning：Generalized Multi-kernel Correntropy-based Deep Reinforcement Learning for Dynamic Topology Planning in Multi-Layered Satellite Network


https://github.com/user-attachments/assets/0f92ca61-23d6-422a-b5f7-966f74158fef


- Dueling Double DQN (edge-level Q) + b-matching
- Generalized Multi-kernel Correntropy (GMKC) loss 
- n-step training targets  
- Sliding-window lookahead for the **action layer** (can be extended in `planner/rollout.py`)  
- Metrics logged to CSV, written in batches with forced flush  

## Directory Structure

- `config/config.py`: constants / hyperparameters (e.g., MAX_LINK, window size W, etc.)
- `data/loader_adapter.py`: data loading (tries to import `data_loader.py` first, otherwise falls back to `.npy`)
- `topo/*`: topology-related utilities (conn_gain, bmatch, graph metrics, etc.)
- `models/edge_q.py`: Q-network
- `rl/replay.py`: sequence replay buffer (n-step)
- `planner/rollout.py`: sliding-window lookahead scoring (used only at the action layer)
- `utils/*`: logging / random seeds
- `train.py`: training entry point (Mode A)

## Dependencies

```bash
pip install torch numpy networkx
```
## Run

```bash
python train.py
```

> If you want to use your own `data_loader.py` and the “visible time window JSON”, place them in your `PYTHONPATH` and adapt the field mappings in `data/loader_adapter.py`.
