import matplotlib.pyplot as plt
from typing import List, Dict
from state_management_module import VehicleStateManager, VehicleState
from initialization_module import VehicleType, Obstacle

class SimulationVisualizer:
    """Visualizes simulation results"""
    
    def __init__(self, state_manager: VehicleStateManager):
        self.state_manager = state_manager
        
    def plot_vehicle_trajectories(self, obstacles: List[Obstacle] = None):
        """Plot vehicle trajectories over time"""
        plt.figure(figsize=(14, 8))
        
        # Plot each vehicle's trajectory
        for vehicle_id, vehicle in self.state_manager.vehicles.items():
            history = self.state_manager.state_history.get_vehicle_history(vehicle_id)
            
            if not history:
                continue
                
            times = [state.timestamp for state in history]
            positions = [state.position for state in history]
            lanes = [state.lane for state in history]
            
            # Color by vehicle type
            color = 'blue' if vehicle.type == VehicleType.CAV else 'red'
            label = f"{vehicle.type.name} {vehicle_id}"
            
            plt.plot(positions, lanes, marker='o', label=label, color=color, alpha=0.7)
        
        # Plot obstacles
        if obstacles:
            for obs in obstacles:
                plt.scatter(obs.position, obs.lane, marker='X', s=200, 
                          color='black', label=f'Obstacle {obs.id}', zorder=5)
        
        plt.xlabel('Position (m)', fontsize=12)
        plt.ylabel('Lane Number', fontsize=12)
        plt.title('Vehicle Trajectories', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yticks(range(1, self.state_manager.road_config.num_lanes + 1))
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_profiles(self):
        """Plot velocity over time for each vehicle"""
        plt.figure(figsize=(14, 6))
        
        for vehicle_id, vehicle in self.state_manager.vehicles.items():
            history = self.state_manager.state_history.get_vehicle_history(vehicle_id)
            
            if not history:
                continue
                
            times = [state.timestamp for state in history]
            velocities = [state.velocity for state in history]
            
            color = 'blue' if vehicle.type == VehicleType.CAV else 'red'
            label = f"{vehicle.type.name} {vehicle_id}"
            
            plt.plot(times, velocities, label=label, color=color, alpha=0.7)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Velocity (m/s)', fontsize=12)
        plt.title('Vehicle Velocity Profiles', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, filename: str = 'simulation_results.csv'):
        """Export simulation data to CSV"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Vehicle_ID', 'Type', 'Position', 'Lane', 'Velocity', 'Acceleration'])
            
            for vehicle_id in self.state_manager.vehicles:
                history = self.state_manager.state_history.get_vehicle_history(vehicle_id)
                
                for state in history:
                    writer.writerow([
                        f"{state.timestamp:.2f}",
                        state.vehicle_id,
                        state.vehicle_type.name,
                        f"{state.position:.2f}",
                        state.lane,
                        f"{state.velocity:.2f}",
                        f"{state.acceleration:.2f}"
                    ])
        
        print(f"Simulation data exported to {filename}")
