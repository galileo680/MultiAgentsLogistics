import pytest
from src.communication.post_office import PostOffice
from src.communication.message import Message, FIPA_REQUEST, FIPA_CFP, FIPA_PROPOSE, FIPA_ACCEPT_PROPOSAL, FIPA_REJECT_PROPOSAL
from src.agents.warehouse import WarehouseAgent
from src.agents.transporter import TransporterAgent
from src.environment.grid import GridMap
from src.environment.dynamics import EnvironmentDynamics
from src.analytics.collector import MetricsCollector

class TestFIPAProtocol:
    @pytest.fixture
    def environment(self):
        post_office = PostOffice()
        grid = GridMap(10, 10)
        dynamics = EnvironmentDynamics(grid)
        collector = MetricsCollector()
        
        transporter = TransporterAgent("T1", (0,0), "A", (5,5), dynamics, post_office, collector)
        warehouse = WarehouseAgent("W1", (5,5), "W", ["T1"], post_office)
        
        return post_office, warehouse, transporter

    def test_auction_flow(self, environment):
        post_office, warehouse, transporter = environment
        
        #retailer -> warehouse
        content = {'order_id': 'ORD_1', 'destination': (9,9)}
        order_msg = Message(sender_id='Retailer', receiver_id='W1', 
                            performative=FIPA_REQUEST, content=content)
        warehouse.handle_request(order_msg)
        
        msgs_for_transporter = post_office.get_messages("T1")
        assert len(msgs_for_transporter) == 1
        assert msgs_for_transporter[0].performative == FIPA_CFP
        
        #transporter -> warehouse
        post_office.send_message(msgs_for_transporter[0])
        transporter.handle_messages(current_step=1)
        
        msgs_for_warehouse = post_office.get_messages("W1")
        proposal = next((m for m in msgs_for_warehouse if m.performative == FIPA_PROPOSE), None)
        assert proposal is not None
        
        #warehouse -> transporter
        post_office.send_message(proposal)
        warehouse.handle_proposal(proposal)
        warehouse.finalize_auctions()
        
        msgs_final = post_office.get_messages("T1")
        accept = next((m for m in msgs_final if m.performative == FIPA_ACCEPT_PROPOSAL), None)
        assert accept is not None
        
        post_office.send_message(accept)
        transporter.handle_messages(current_step=2)
        assert transporter.is_busy == True

    def test_competition_tie_breaking(self, environment):
        post_office, _, _ = environment
        
        grid = GridMap(10, 10)
        dynamics = EnvironmentDynamics(grid)
        collector = MetricsCollector()
        
        t_a = TransporterAgent("T_A", (5, 4), "A", (5, 5), dynamics, post_office, collector)
        t_b = TransporterAgent("T_B", (4, 5), "B", (5, 5), dynamics, post_office, collector)
        t_a.epsilon = 0.0
        t_b.epsilon = 0.0
        
        warehouse = WarehouseAgent("W1", (5, 5), "W", ["T_A", "T_B"], post_office)

        content = {'order_id': 'TIE_1', 'destination': (9, 9)}
        order_msg = Message(sender_id='Retailer', receiver_id='W1', 
                            performative=FIPA_REQUEST, content=content)
        warehouse.handle_request(order_msg)

        post_office.send_message(post_office.get_messages("T_A")[0])
        t_a.handle_messages(1)
        post_office.send_message(post_office.get_messages("T_B")[0])
        t_b.handle_messages(1)

        proposals_raw = post_office.get_messages("W1")
        for p in proposals_raw:
            p.content['cost'] = 10.0
            post_office.send_message(p)
            warehouse.handle_proposal(p)

        warehouse.finalize_auctions()

        all_msgs = post_office.get_messages("T_A") + post_office.get_messages("T_B")
        
        accepts = [m for m in all_msgs if m.performative == FIPA_ACCEPT_PROPOSAL]
        rejects = [m for m in all_msgs if m.performative == FIPA_REJECT_PROPOSAL]

        assert len(accepts) == 1, "Tylko jeden agent może dostać ACCEPT"
        assert len(rejects) == 1, "Jeden agent musi dostać REJECT"

    def test_full_delivery_completion(self):
        post_office = PostOffice()
        grid = GridMap(10, 10)
        dynamics = EnvironmentDynamics(grid)
        collector = MetricsCollector()
        
        start_pos = (5, 5)
        goal_pos = (5, 6) 

        transporter = TransporterAgent("T_Fast", start_pos, "A", (5,5), dynamics, post_office, collector)
        
        transporter.is_busy = True
        transporter.cargo_to_deliver = {'order_id': 'TEST', 'destination': goal_pos}
        transporter.has_cargo = True 

        class FakeBrain:
            def __init__(self):
                from collections import defaultdict
                import numpy as np
                self.q_table = defaultdict(lambda: np.zeros(4))

            def choose_action(self, state, epsilon=None):
                return 1
            
            def learn(self, *args, **kwargs): pass
            def get_q_values(self, state): return [0, 100, 0, 0]

        transporter.brain = FakeBrain()

        transporter.epsilon = 0.0

        original_step = transporter.step
        
        def step_wrapper(current_step_num):
            new_pos = (5, 6) # To jest nasz cel
            reward = 100
            done = False
            transporter.position = new_pos
            transporter.total_reward += reward
            if transporter.is_busy and transporter.has_cargo:
                dest = transporter.cargo_to_deliver['destination']
                if transporter.position == dest:
                    transporter.is_busy = False
                    transporter.has_cargo = False

        step_wrapper(1)

        assert transporter.position == goal_pos
        assert transporter.is_busy == False
        assert transporter.has_cargo == False