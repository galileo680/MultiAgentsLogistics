from dataclasses import dataclass
from typing import Any

FIPA_REQUEST = "REQUEST"          
FIPA_CFP = "CFP"                   
FIPA_PROPOSE = "PROPOSE"           
FIPA_REFUSE = "REFUSE"             
FIPA_ACCEPT_PROPOSAL = "ACCEPT"    
FIPA_REJECT_PROPOSAL = "REJECT"    
FIPA_INFORM = "INFORM"       

@dataclass
class Message:
    sender_id: str          
    receiver_id: str      
    performative: str       
    content: Any          
    
    def __str__(self):
        return f"[{self.performative}] {self.sender_id} -> {self.receiver_id}: {self.content}"