
# -*- coding: utf-8 -*-
"""
CSV 指标记录器：
- 支持 step 级与 epoch 级两个 CSV
- 每写入一定数量（cfg.FLUSH_EVERY_STEPS）批量落盘并 flush（硬刷新）
"""
import os, csv, time
from config.config import cfg

class MetricLogger:
    def __init__(self, run_dir:str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.step_csv = os.path.join(run_dir, "results_step_log.csv")
        self.epoch_csv = os.path.join(run_dir, "results_log.csv")

        with open(self.step_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step','conn','avg_delay','avg_hop','switch','loss','reward','notes'])

        with open(self.epoch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(['epoch','conn','avg_delay','avg_hop','switch','total_reward'])
            writer.writerow(['epoch','total_reward'])

        self._buffer = []
        self._last_flush = time.time()

    def log_step(self, step:int, conn:float, avg_delay:float, avg_hop:float,
                 switch:int, loss:float, reward:float, notes:str=""):
        self._buffer.append([step, conn, avg_delay, avg_hop, switch, loss, reward, notes])
        if len(self._buffer) >= cfg.FLUSH_EVERY_STEPS:
            self.flush()

    def flush(self):
        if not self._buffer: return
        with open(self.step_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self._buffer)
            f.flush()         # 硬刷新
        self._buffer.clear()

    def log_epoch(self, epoch:int, total_reward:float):
        with open(self.epoch_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow([epoch, conn, avg_delay, avg_hop, switch, total_reward])
            writer.writerow([epoch, total_reward])
            f.flush()
