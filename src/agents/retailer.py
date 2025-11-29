import random
from src.agents.base_agent import BaseAgent
from src.communication.message import Message, FIPA_REQUEST

class RetailerAgent(BaseAgent):
    def __init__(self, agent_id, start_pos, warehouse_id):
        super().__init__(agent_id, start_pos)
        self.warehouse_id = warehouse_id
        self.orders_placed = 0

    def step(self, environment_dynamics, post_office):
        if random.random() < 0.05:
            self.generate_order(post_office)

        messages = post_office.get_messages(self.agent_id)
        for msg in messages:
            pass

    def generate_order(self, post_office):
        content = {
            "destination": self.position,
            "order_id": f"{self.agent_id}_ORD_{self.orders_placed}"
        }
        
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=self.warehouse_id,
            performative=FIPA_REQUEST,
            content=content
        )
        
        post_office.send_message(msg)
        self.orders_placed += 1