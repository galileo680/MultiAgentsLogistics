from collections import defaultdict
from src.communication.message import Message

class PostOffice:
    def __init__(self):
        self._mailboxes = defaultdict(list)
        self.history = []

    def send_message(self, message: Message):
        self._mailboxes[message.receiver_id].append(message)
        
        self.history.append(message)
        # print(f"LOG KOMUNIKACJI: {message}")

    def broadcast(self, message, recipients):
        for recipient_id in recipients:
            new_msg = Message(
                sender_id=message.sender_id,
                receiver_id=recipient_id,
                performative=message.performative,
                content=message.content
            )
            self.send_message(new_msg)

    def get_messages(self, agent_id):
        if agent_id in self._mailboxes:
            messages = self._mailboxes[agent_id]
            self._mailboxes[agent_id] = []
            return messages
        return []

    def clear_history(self):
        self._mailboxes.clear()
        self.history = []