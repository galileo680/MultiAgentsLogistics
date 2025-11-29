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
    grid = GridMap(config.GRID_WIDTH, config.GRID_HEIGHT)
    grid.generate_simple_map() # Generuje ściany i układ
    
    dynamics = EnvironmentDynamics(grid)
    post_office = PostOffice()

    transporters = []
    t1 = TransporterAgent("Transporter-1", start_pos=(0, 0))
    t2 = TransporterAgent("Transporter-2", start_pos=(0, 1))
    transporters.extend([t1, t2])
    
    warehouse = WarehouseAgent(
        agent_id="Warehouse-Main", 
        start_pos=(5, 5), 
        transporter_ids=[t.agent_id for t in transporters]
    )
    
    retailer = RetailerAgent(
        agent_id="Retailer-Shop1", 
        start_pos=(9, 9), 
        warehouse_id=warehouse.agent_id
    )

    all_agents = transporters + [warehouse, retailer]

    print(f"Start symulacji na {num_steps} kroków...")
    
    for step in range(num_steps):
        if step % 5 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- KROK {step}/{num_steps} ---")
            grid.render_console(agent_pos=t1.position)
            print(f"Transporter-1 Busy: {t1.is_busy}, Reward: {t1.total_reward}")
            print(f"Transporter-2 Busy: {t2.is_busy}")
            print(f"Poczta: {len(post_office.history)} wysłanych wiadomości.")

        retailer.step(dynamics, post_office)
        
        warehouse.step(dynamics, post_office)
        
        for t in transporters:
            t.step(dynamics, post_office)

        time.sleep(0.1)

    print("\n--- KONIEC SYMULACJI ---")
    print("Statystyki:")
    for t in transporters:
        print(f"{t.agent_id}: Całkowita nagroda = {t.total_reward}, Epsilon = {t.epsilon:.4f}")
        t.brain.save_table(f"{t.agent_id}_brain.npy")

if __name__ == "__main__":
    run_simulation(num_steps=3000)