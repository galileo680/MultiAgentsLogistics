import random

from src.agents.base_agent import BaseAgent
from src.communication.message import FIPA_REQUEST, Message


class RetailerAgent(BaseAgent):
    def __init__(
        self,
        agent_id,
        start_pos,
        label,
        warehouse_id,
        post_office,
    ):
        super().__init__(agent_id, start_pos, label)
        self.warehouse_id = warehouse_id
        self.orders_placed = 0

        self.post_office = post_office

    def step(self):
        if random.random() < 0.05:  # prev: 0.05
            self.generate_order()
        messages = self.post_office.get_messages(self.agent_id)
        for msg in messages:
            pass

    def generate_order(self):
        content = {
            "destination": self.position,
            "order_id": f"{self.agent_id}_ORD_{self.orders_placed}",
        }

        msg = Message(
            sender_id=self.agent_id,
            receiver_id=self.warehouse_id,
            performative=FIPA_REQUEST,
            content=content,
        )

        self.post_office.send_message(msg)
        self.orders_placed += 1
