import numpy as np

import config
from src.agents.base_agent import BaseAgent
from src.communication.message import (
    FIPA_ACCEPT_PROPOSAL,
    FIPA_CFP,
    FIPA_PROPOSE,
    FIPA_REFUSE,
    FIPA_REJECT_PROPOSAL,
    Message,
)
from src.learning.policy import Policy
from src.learning.q_learning import QLearningTable


class TransporterAgent(BaseAgent):
    def __init__(
        self,
        agent_id,
        start_pos,
        label,
        warehouse_location,
        environment_dynamics,
        post_office,
        metrics_collector,
    ):
        super().__init__(agent_id, start_pos, label)

        self.brain = QLearningTable(
            actions=list(range(config.NUM_ACTIONS)),
            learning_rate=config.ALPHA,
            reward_decay=config.GAMMA,
        )
        self.epsilon = config.EPSILON_START

        self.is_busy = False
        self.has_cargo = False

        self.current_target = None
        self.final_destination = None  # Sklep
        self.warehouse_location = warehouse_location  # Magazyn
        self.current_order_id = None

        self.reached_goal_count = 0

        self.total_reward = 0

        self.job_start_step = 0
        self.agreed_cost = 0

        self.environment_dynamics = environment_dynamics
        self.post_office = post_office
        self.metrics_collector = metrics_collector

    def get_state(self):
        return (self.position, self.has_cargo)

    def step(self, current_step):
        self.handle_messages(current_step)

        if self.is_busy and self.current_target:
            self.move_towards_target(current_step)
        else:
            pass

    def handle_messages(self, current_step):
        messages = self.post_office.get_messages(self.agent_id)

        for msg in messages:
            if msg.performative == FIPA_CFP:
                if not self.is_busy:
                    dest = msg.content["destination"]

                    dist_to_warehouse = abs(
                        self.position[0] - self.warehouse_location[0]
                    ) + abs(self.position[1] - self.warehouse_location[1])

                    dist_to_target = abs(self.warehouse_location[0] - dest[0]) + abs(
                        self.warehouse_location[1] - dest[1]
                    )

                    total_dist = dist_to_warehouse + dist_to_target

                    # Narzut za "niewiedzę" (explorację).
                    # Jeśli epsilon=1.0 (start), narzut jest duży (np. 10x).
                    # Jeśli epsilon=0.01 (koniec), narzut jest minimalny.
                    bast_cost = total_dist * 1.5
                    uncertainty_factor = 1 + (self.epsilon * 20)

                    estimated_steps = bast_cost * uncertainty_factor

                    cost = estimated_steps

                    reply = Message(
                        sender_id=self.agent_id,
                        receiver_id=msg.sender_id,
                        performative=FIPA_PROPOSE,
                        content={"order_id": msg.content["order_id"], "cost": cost},
                    )
                    self.post_office.send_message(reply)
                else:
                    reply = Message(
                        sender_id=self.agent_id,
                        receiver_id=msg.sender_id,
                        performative=FIPA_REFUSE,
                        content={"order_id": msg.content["order_id"]},
                    )
                    self.post_office.send_message(reply)

            elif msg.performative == FIPA_ACCEPT_PROPOSAL:
                self.is_busy = True
                self.has_cargo = False
                self.current_order_id = msg.content["order_id"]
                self.final_destination = msg.content["destination"]

                self.current_target = self.warehouse_location

                self.agreed_cost = msg.content["cost"]
                self.job_start_step = current_step

            elif msg.performative == FIPA_REJECT_PROPOSAL:
                pass

    def move_towards_target(self, current_step):
        state = self.get_state()

        q_values = self.brain.get_q_values(state)
        action_idx = Policy.epsilon_greedy(q_values, self.epsilon, config.NUM_ACTIONS)

        next_pos, env_reward, hit_wall = self.environment_dynamics.step(
            self.position, action_idx
        )

        reward = config.REWARD_STEP
        done = False

        if next_pos == self.current_target:
            # Agent dotarl do magazynu i odbiera towar
            if not self.has_cargo:
                self.has_cargo = True
                self.current_target = self.final_destination
                reward = 20  # zhardkodowane -> do zmiany

            # Agent dotarl do sklepu
            else:
                reward = config.REWARD_GOAL
                self.reached_goal_count += 1
                done = True

        elif hit_wall:
            if next_pos == self.position:
                reward = config.REWARD_COLLISION
            else:
                reward = config.REWARD_STEP
            done = False
        else:
            reward = config.REWARD_STEP
            done = False

        next_state = (next_pos, self.has_cargo)

        self.brain.learn(state, action_idx, reward, next_state, done)
        self.position = next_pos
        self.total_reward += reward

        if done:
            self.finish_job(current_step)

    def finish_job(self, current_step):
        duration = current_step - self.job_start_step

        self.metrics_collector.log_delivery(
            self.agent_id, self.current_order_id, duration, self.agreed_cost
        )

        self.is_busy = False
        self.has_cargo = False
        self.current_target = None
        self.current_order_id = None
        self.final_destination = None

        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)
