# src/analytics/collector.py
import os

import numpy as np
import pandas as pd

import config


class MetricsCollector:
    def __init__(self):
        self.step_data = []
        self.delivery_data = []
        self.heatmap_grid = np.zeros((config.GRID_WIDTH, config.GRID_HEIGHT))

    def log_step(self, step_num, agents):
        for agent in agents:
            if "Transporter" in agent.agent_id:
                self.step_data.append(
                    {
                        "step": step_num,
                        "agent_id": agent.agent_id,
                        "total_reward": agent.total_reward,
                        "epsilon": agent.epsilon,
                        "is_busy": int(agent.is_busy),
                    }
                )
                x, y = agent.position
                if 0 <= x < 10 and 0 <= y < 10:
                    self.heatmap_grid[y][x] += 1

    def log_delivery(self, agent_id, order_id, steps_taken, promised_cost):
        print(
            f"Agent {agent_id} zrealizował zamówienie {order_id} w {steps_taken} krokach za koszt {promised_cost}"
        )
        self.delivery_data.append(
            {
                "agent_id": agent_id,
                "order_id": order_id,
                "steps_taken": steps_taken,
                "promised_cost": promised_cost,
            }
        )

    def save_data(self):
        output_dir = os.path.join("src", "analytics", "data")

        df_steps = pd.DataFrame(self.step_data)
        df_steps.to_csv(os.path.join(output_dir, "analytics_steps.csv"), index=False)

        df_deliveries = pd.DataFrame(self.delivery_data)
        df_deliveries.to_csv(
            os.path.join(output_dir, "analytics_deliveries.csv"), index=False
        )

        np.save(
            os.path.join(output_dir, "analytics_heatmap.npy"),
            self.heatmap_grid,
        )

        print(f"Dane zapisane w: {output_dir}")
