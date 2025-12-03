import time
import os
import config
from src.environment.grid import GridMap
from src.environment.dynamics import EnvironmentDynamics
from src.communication.post_office import PostOffice
from src.agents.retailer import RetailerAgent
from src.agents.warehouse import WarehouseAgent
from src.agents.transporter import TransporterAgent

def run_simulation(num_steps=100):
    grid = GridMap(config.GRID_WIDTH, config.GRID_HEIGHT,)
    grid.generate_simple_map() # Generuje ściany i układ
    
    dynamics = EnvironmentDynamics(grid)
    post_office = PostOffice()

    transporters = []
    t1 = TransporterAgent("Transporter-1", start_pos=(0, 0), label="A")
    t2 = TransporterAgent("Transporter-2", start_pos=(0, 1), label="B")
    transporters.extend([t1, t2])
    
    warehouse = WarehouseAgent(
        agent_id="Warehouse-Main", 
        start_pos=(5, 5),
        label="W",
        transporter_ids=[t.agent_id for t in transporters]
    )
    
    retailer = RetailerAgent(
        agent_id="Retailer-Shop1", 
        start_pos=(9, 9), 
        label="R",
        warehouse_id=warehouse.agent_id
    )

    all_agents = transporters + [warehouse, retailer]

    print(f"Start symulacji na {num_steps} kroków...")
    
    for step in range(num_steps):
        #if step % 5 == 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- KROK {step}/{num_steps} ---")
        agent_positions = [(agent.position[0], agent.position[1], agent.label) for agent in all_agents]
        grid.render_console(agent_positions)
        for t in transporters:
            print(f"{t.agent_id} Busy: {t.is_busy}, Reward: {t.total_reward}, Reached Goal Count: {t.reached_goal_count}")

        print(f"Poczta: {len(post_office.history)} wysłanych wiadomości.")

        retailer.step(dynamics, post_office)
        
        warehouse.step(dynamics, post_office)
        
        for t in transporters:
            t.step(dynamics, post_office)

        time.sleep(0.01)

    print("\n--- KONIEC SYMULACJI ---")
    print("Statystyki:")
    for t in transporters:
        print(f"{t.agent_id}: Całkowita nagroda = {t.total_reward}, Epsilon = {t.epsilon:.4f}")
        t.brain.save_table(f"{t.agent_id}_brain.npy")

if __name__ == "__main__":
    run_simulation(num_steps=8000)