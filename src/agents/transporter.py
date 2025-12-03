import numpy as np
import config
from src.agents.base_agent import BaseAgent
from src.learning.q_learning import QLearningTable
from src.learning.policy import Policy
from src.communication.message import (
    Message, FIPA_CFP, FIPA_PROPOSE, 
    FIPA_ACCEPT_PROPOSAL, FIPA_REJECT_PROPOSAL, FIPA_REFUSE
)

class TransporterAgent(BaseAgent):
    def __init__(self, agent_id, start_pos, label):
        super().__init__(agent_id, start_pos, label)
        
        self.brain = QLearningTable(
            actions=list(range(config.NUM_ACTIONS)),
            learning_rate=config.ALPHA,
            reward_decay=config.GAMMA
        )
        self.epsilon = config.EPSILON_START
        
        self.is_busy = False
        self.has_cargo = False;

        self.current_target = None
        self.final_destination = None;  # Sklep
        self.warehouse_location = None; # Magazyn
        self.current_order_id = None

        self.reached_goal_count = 0
        
        self.total_reward = 0

    def get_state(self):
        return (self.position, self.has_cargo)

    def step(self, environment_dynamics, post_office):
        self.handle_messages(post_office)
        
        if self.is_busy and self.current_target:
            self.move_towards_target(environment_dynamics, post_office)
        else:
            pass

    def handle_messages(self, post_office):
        messages = post_office.get_messages(self.agent_id)
        
        for msg in messages:
            if msg.performative == FIPA_CFP:
                if not self.is_busy:
                    dest = msg.content['destination']
                    dist = abs(self.position[0] - dest[0]) + abs(self.position[1] - dest[1])
                    cost = dist * 1.5
                    
                    reply = Message(
                        sender_id=self.agent_id,
                        receiver_id=msg.sender_id,
                        performative=FIPA_PROPOSE,
                        content={'order_id': msg.content['order_id'], 'cost': cost}
                    )
                    post_office.send_message(reply)
                else:
                    reply = Message(
                        sender_id=self.agent_id,
                        receiver_id=msg.sender_id,
                        performative=FIPA_REFUSE,
                        content={'order_id': msg.content['order_id']}
                    )
                    post_office.send_message(reply)

            elif msg.performative == FIPA_ACCEPT_PROPOSAL:
                self.is_busy = True
                self.has_cargo = False
                self.current_order_id = msg.content['order_id']
                self.final_destination = msg.content['destination']

                self.warehouse_location = (5, 5) # zhardcodowane, do zmiany

                self.current_target = self.warehouse_location
                
            elif msg.performative == FIPA_REJECT_PROPOSAL:
                pass

    def move_towards_target(self, environment_dynamics, post_office):
        state = self.get_state()
        
        q_values = self.brain.get_q_values(state)
        action_idx = Policy.epsilon_greedy(q_values, self.epsilon, config.NUM_ACTIONS)
        
        next_pos, env_reward, hit_wall = environment_dynamics.step(self.position, action_idx)
        
        reward = config.REWARD_STEP
        done = False

        if next_pos == self.current_target:
            
            # Agent dotarl do magazynu i odbiera towar
            if not self.has_cargo:
                self.has_cargo = True
                self.current_target = self.final_destination
                reward = 20; # zhardkodowane -> do zmiany
            
            # Agent dotarl do sklepu
            else:
                reward = config.REWARD_GOAL
                self.reached_goal_count += 1
                done = True

        elif hit_wall:
            if next_pos == state: 
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
            self.finish_job(post_office)

    def finish_job(self, post_office):
        self.is_busy = False
        self.has_cargo = False
        self.current_target = None
        self.current_order_id = None
        self.final_destination = None
        
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)