from src.agents.base_agent import BaseAgent
from src.communication.message import (
    Message, FIPA_REQUEST, FIPA_CFP, 
    FIPA_PROPOSE, FIPA_REFUSE, 
    FIPA_ACCEPT_PROPOSAL, FIPA_REJECT_PROPOSAL
)

class WarehouseAgent(BaseAgent):
    def __init__(self, agent_id, start_pos, transporter_ids):
        super().__init__(agent_id, start_pos)
        self.transporter_ids = transporter_ids
        
        self.active_auctions = {}

    def step(self, environment_dynamics, post_office):
        messages = post_office.get_messages(self.agent_id)
        
        for msg in messages:
            if msg.performative == FIPA_REQUEST:
                self.handle_request(msg, post_office)
            
            elif msg.performative == FIPA_PROPOSE:
                self.handle_proposal(msg)
                
            elif msg.performative == FIPA_REFUSE:
                pass

        self.finalize_auctions(post_office)

    def handle_request(self, msg, post_office):
        order_details = msg.content
        order_id = order_details['order_id']
        
        self.active_auctions[order_id] = {
            'content': order_details,
            'proposals': [],
            'pending_responses': len(self.transporter_ids)
        }
        
        cfp_msg = Message(
            sender_id=self.agent_id,
            receiver_id=None, # Broadcast
            performative=FIPA_CFP,
            content=order_details
        )
        post_office.broadcast(cfp_msg, self.transporter_ids)

    def handle_proposal(self, msg):
        order_id = msg.content['order_id']
        cost = msg.content['cost']
        
        if order_id in self.active_auctions:
            auction = self.active_auctions[order_id]
            auction['proposals'].append({
                'agent_id': msg.sender_id,
                'cost': cost
            })
            auction['pending_responses'] -= 1

    def finalize_auctions(self, post_office):
        for order_id in list(self.active_auctions.keys()):
            auction = self.active_auctions[order_id]
            
            if auction['pending_responses'] <= 0 and auction['proposals']:
                
                best_offer = min(auction['proposals'], key=lambda x: x['cost'])
                winner_id = best_offer['agent_id']
                
                accept_msg = Message(
                    sender_id=self.agent_id,
                    receiver_id=winner_id,
                    performative=FIPA_ACCEPT_PROPOSAL,
                    content=auction['content']
                )
                post_office.send_message(accept_msg)
                
                for prop in auction['proposals']:
                    if prop['agent_id'] != winner_id:
                        reject_msg = Message(
                            sender_id=self.agent_id,
                            receiver_id=prop['agent_id'],
                            performative=FIPA_REJECT_PROPOSAL,
                            content={'order_id': order_id}
                        )
                        post_office.send_message(reject_msg)
                
                del self.active_auctions[order_id]