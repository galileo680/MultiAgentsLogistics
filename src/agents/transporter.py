import numpy as np
import config
from src.agents.base_agent import BaseAgent
from src.learning.q_learning import QLearningTable
from src.learning.policy import Policy
from src.communication.message import (
    Message, FIPA_CFP, FIPA_PROPOSE, 
    FIPA_ACCEPT_PROPOSAL, FIPA_REJECT_PROPOSAL, FIPA_INFORM
)

class TransporterAgent(BaseAgent):
    def __init__(self, agent_id, start_pos):
        super().__init__(agent_id, start_pos)
        
        self.brain = QLearningTable(
            actions=list(range(config.NUM_ACTIONS)),
            learning_rate=config.ALPHA,
            reward_decay=config.GAMMA
        )
        self.epsilon = config.EPSILON_START
        
        self.is_busy = False
        self.current_target = None
        self.current_order_id = None
        
        self.total_reward = 0

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
            
            elif msg.performative == FIPA_ACCEPT_PROPOSAL:
                self.is_busy = True
                self.current_target = msg.content['destination']
                self.current_order_id = msg.content['order_id']

            elif msg.performative == FIPA_REJECT_PROPOSAL:
                pass

    def move_towards_target(self, environment_dynamics, post_office):
        state = self.position
        
        q_values = self.brain.get_q_values(state)
        action_idx = Policy.epsilon_greedy(q_values, self.epsilon, config.NUM_ACTIONS)
        
        next_pos, env_reward, hit_wall = environment_dynamics.step(state, action_idx)
        
        if next_pos == self.current_target:
            reward = config.REWARD_GOAL
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

        self.brain.learn(state, action_idx, reward, next_pos, done)
        
        self.position = next_pos
        self.total_reward += reward
        
        if done:
            self.finish_job(post_office)

    def finish_job(self, post_office):
        self.is_busy = False
        self.current_target = None
        self.current_order_id = None
        
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)