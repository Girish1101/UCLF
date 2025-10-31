import numpy as np
from typing import List, Dict
from initialization_module import ScenarioInitializer
from state_management_module import VehicleStateManager, VehicleState
from hdv_behaviour_module import HDVBehaviorPredictor
from priority_manager_module import PriorityManager
from collision_detection_module import CollisionDetector

class SimulationEngine:
    """Main simulation engine that coordinates all modules"""
    
    def __init__(self, scenario: ScenarioInitializer):
        self.scenario = scenario
        
        # Initialize all managers
        self.state_manager = VehicleStateManager(scenario.road_config)
        self.state_manager.add_vehicles(scenario.vehicles)
        
        self.hdv_predictor = HDVBehaviorPredictor(self.state_manager)
        self.priority_manager = PriorityManager(self.state_manager)
        self.collision_detector = CollisionDetector(self.state_manager)
        
        self.simulation_time = 0.0
        self.time_step = 0.1  # seconds
        self.total_steps = 0
        
    def run_single_step(self):
        """Execute one simulation step"""
        # 1. Predict HDV behaviors
        hdv_predictions = self.hdv_predictor.predict_all_hdvs(self.scenario.obstacles)
        
        # 2. Calculate CAV priorities
        self.priority_manager.calculate_all_priorities(self.scenario.obstacles)
        
        # 3. Check for collisions
        collisions = self.collision_detector.check_all_collisions(self.scenario.obstacles)
        
        # 4. Update vehicle positions (simple forward simulation)
        self.state_manager.step_simulation(self.time_step)
        
        self.simulation_time += self.time_step
        self.total_steps += 1
        
        return {
            'time': self.simulation_time,
            'hdv_predictions': hdv_predictions,
            'priorities': self.priority_manager.get_priority_order(),
            'collisions': collisions
        }
    
    def run_simulation(self, duration: float = 10.0, verbose: bool = True):
        """Run simulation for specified duration"""
        num_steps = int(duration / self.time_step)
        
        if verbose:
            print("\n" + "="*80)
            print(f"STARTING SIMULATION - Duration: {duration}s, Steps: {num_steps}")
            print("="*80 + "\n")
            
            # Initial state
            print("INITIAL STATE:")
            self.state_manager.print_current_state()
        
        collision_count = 0
        
        for step in range(num_steps):
            result = self.run_single_step()
            
            # Count collisions
            if result['collisions']:
                collision_count += len(result['collisions'])
                if verbose:
                    self.collision_detector.print_collision_report(result['collisions'])
            
            # Print periodic updates
            if verbose and (step + 1) % 50 == 0:
                print(f"\n--- Time: {result['time']:.1f}s ---")
                self.state_manager.print_current_state()
        
        if verbose:
            print("\n" + "="*80)
            print("SIMULATION COMPLETE")
            print("="*80)
            print(f"Total Time: {self.simulation_time:.2f}s")
            print(f"Total Steps: {self.total_steps}")
            print(f"Collision Events: {collision_count}")
            print("="*80 + "\n")
            
            # Final state
            print("FINAL STATE:")
            self.state_manager.print_current_state()
            
    def get_summary(self) -> Dict:
        """Get simulation summary statistics"""
        return {
            'total_time': self.simulation_time,
            'total_steps': self.total_steps,
            'total_collisions': self.collision_detector.get_collision_count(),
            'num_cavs': len(self.state_manager.get_vehicles_by_type(self.scenario.vehicles[0].type.__class__.CAV)),
            'num_hdvs': len(self.state_manager.get_vehicles_by_type(self.scenario.vehicles[0].type.__class__.HDV))
        }
