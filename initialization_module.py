import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

# ============================================
# ENUMERATIONS AND DATA CLASSES
# ============================================

class VehicleType(Enum):
    CAV = "Connected Autonomous Vehicle"
    HDV = "Human Driven Vehicle"

class LaneAction(Enum):
    KEEP_LANE = "K"
    CHANGE_LEFT = "L"
    CHANGE_RIGHT = "R"

@dataclass
class Vehicle:
    """Represents a vehicle in the simulation"""
    id: int
    type: VehicleType
    position: float          # Longitudinal position (meters)
    lane: int               # Current lane number (1, 2, 3, ...)
    velocity: float         # Speed (m/s)
    acceleration: float     # Current acceleration (m/s²)
    length: float = 4.5     # Vehicle length (meters)
    width: float = 2.0      # Vehicle width (meters)
    
    # HDV-specific attributes
    intended_action: LaneAction = LaneAction.KEEP_LANE
    action_probabilities: Dict[LaneAction, float] = None
    
    def __post_init__(self):
        if self.action_probabilities is None:
            self.action_probabilities = {
                LaneAction.KEEP_LANE: 0.7,
                LaneAction.CHANGE_LEFT: 0.2,
                LaneAction.CHANGE_RIGHT: 0.1
            }

@dataclass
class Obstacle:
    """Represents a static obstacle on the road"""
    id: int
    position: float         # Longitudinal position (meters)
    lane: int              # Lane number
    length: float          # Length of obstacle (meters)
    width: float           # Width of obstacle (meters)
    type: str = "blockage" # Type: blockage, lane_drop, barrier
    passable: bool = False # Can vehicles pass through?

@dataclass
class RoadConfig:
    """Represents road configuration"""
    num_lanes: int
    lane_width: float = 3.5  # meters
    road_length: float = 500.0  # meters

# ============================================
# INPUT AND INITIALIZATION MODULE
# ============================================

class ScenarioInitializer:
    """Handles initialization of simulation scenario"""
    
    def __init__(self):
        self.vehicles: List[Vehicle] = []
        self.obstacles: List[Obstacle] = []
        self.road_config: RoadConfig = None
        
    def initialize_road(self, num_lanes: int = 3, road_length: float = 500.0):
        """Initialize road configuration"""
        self.road_config = RoadConfig(
            num_lanes=num_lanes,
            road_length=road_length
        )
        print(f"✓ Road initialized: {num_lanes} lanes, {road_length}m length")
        
    def add_cav(self, vehicle_id: int, position: float, lane: int, 
                velocity: float, acceleration: float = 0.0):
        """Add a Connected Autonomous Vehicle"""
        cav = Vehicle(
            id=vehicle_id,
            type=VehicleType.CAV,
            position=position,
            lane=lane,
            velocity=velocity,
            acceleration=acceleration
        )
        self.vehicles.append(cav)
        print(f"✓ CAV {vehicle_id} added: Lane {lane}, Position {position}m, Speed {velocity}m/s")
        
    def add_hdv(self, vehicle_id: int, position: float, lane: int, 
                velocity: float, acceleration: float = 0.0,
                behavior_probs: Dict[LaneAction, float] = None):
        """Add a Human Driven Vehicle with behavior probabilities"""
        hdv = Vehicle(
            id=vehicle_id,
            type=VehicleType.HDV,
            position=position,
            lane=lane,
            velocity=velocity,
            acceleration=acceleration,
            action_probabilities=behavior_probs
        )
        self.vehicles.append(hdv)
        print(f"✓ HDV {vehicle_id} added: Lane {lane}, Position {position}m, Speed {velocity}m/s")
        
    def add_obstacle(self, obs_id: int, position: float, lane: int,
                    length: float = 10.0, obs_type: str = "blockage"):
        """Add an obstacle to the scenario"""
        obstacle = Obstacle(
            id=obs_id,
            position=position,
            lane=lane,
            length=length,
            width=self.road_config.lane_width,
            type=obs_type
        )
        self.obstacles.append(obstacle)
        print(f"✓ Obstacle {obs_id} added: Lane {lane}, Position {position}m")
        
    def get_vehicles_by_type(self, vehicle_type: VehicleType) -> List[Vehicle]:
        """Get all vehicles of a specific type"""
        return [v for v in self.vehicles if v.type == vehicle_type]
    
    def print_scenario_summary(self):
        """Print summary of the initialized scenario"""
        print("\n" + "="*60)
        print("SCENARIO SUMMARY")
        print("="*60)
        print(f"Road: {self.road_config.num_lanes} lanes × {self.road_config.road_length}m")
        print(f"Total Vehicles: {len(self.vehicles)}")
        print(f"  - CAVs: {len(self.get_vehicles_by_type(VehicleType.CAV))}")
        print(f"  - HDVs: {len(self.get_vehicles_by_type(VehicleType.HDV))}")
        print(f"Obstacles: {len(self.obstacles)}")
        print("="*60)
        
        print("\nVEHICLE DETAILS:")
        print("-"*60)
        for v in self.vehicles:
            print(f"{v.type.name} {v.id}: Lane {v.lane} | Pos: {v.position}m | "
                  f"Speed: {v.velocity}m/s | Accel: {v.acceleration}m/s²")
            if v.type == VehicleType.HDV:
                print(f"  Behavior Probs: K={v.action_probabilities[LaneAction.KEEP_LANE]:.0%}, "
                      f"L={v.action_probabilities[LaneAction.CHANGE_LEFT]:.0%}, "
                      f"R={v.action_probabilities[LaneAction.CHANGE_RIGHT]:.0%}")
        
        if self.obstacles:
            print("\nOBSTACLE DETAILS:")
            print("-"*60)
            for obs in self.obstacles:
                print(f"Obstacle {obs.id}: Lane {obs.lane} | Position {obs.position}m | "
                      f"Type: {obs.type}")
        print("="*60 + "\n")

# ============================================
# PREDEFINED SCENARIO EXAMPLES
# ============================================

def create_scenario_1():
    """Example Scenario 1: Lane blockage with mixed traffic"""
    scenario = ScenarioInitializer()
    
    # Initialize road
    scenario.initialize_road(num_lanes=3, road_length=500.0)
    
    # Add CAVs
    scenario.add_cav(vehicle_id=1, position=50.0, lane=3, velocity=20.0)
    scenario.add_cav(vehicle_id=2, position=80.0, lane=2, velocity=18.0)
    scenario.add_cav(vehicle_id=3, position=100.0, lane=1, velocity=22.0)
    
    # Add HDVs with different behavior probabilities
    scenario.add_hdv(
        vehicle_id=4, position=120.0, lane=2, velocity=19.0,
        behavior_probs={
            LaneAction.KEEP_LANE: 0.7,
            LaneAction.CHANGE_LEFT: 0.2,
            LaneAction.CHANGE_RIGHT: 0.1
        }
    )
    
    scenario.add_hdv(
        vehicle_id=5, position=150.0, lane=1, velocity=21.0,
        behavior_probs={
            LaneAction.KEEP_LANE: 0.6,
            LaneAction.CHANGE_LEFT: 0.0,
            LaneAction.CHANGE_RIGHT: 0.4
        }
    )
    
    # Add obstacle in right lane
    scenario.add_obstacle(obs_id=1, position=200.0, lane=3, length=15.0)
    
    return scenario

def create_scenario_2():
    """Example Scenario 2: Lane drop scenario"""
    scenario = ScenarioInitializer()
    
    # Initialize road
    scenario.initialize_road(num_lanes=3, road_length=400.0)
    
    # Add CAVs
    scenario.add_cav(vehicle_id=1, position=30.0, lane=3, velocity=25.0)
    scenario.add_cav(vehicle_id=2, position=60.0, lane=3, velocity=24.0)
    scenario.add_cav(vehicle_id=3, position=90.0, lane=2, velocity=23.0)
    
    # Add HDVs
    scenario.add_hdv(vehicle_id=4, position=120.0, lane=1, velocity=22.0)
    scenario.add_hdv(vehicle_id=5, position=140.0, lane=2, velocity=20.0)
    
    # Add lane drop obstacle (lane 3 ends)
    scenario.add_obstacle(obs_id=1, position=250.0, lane=3, 
                         length=50.0, obs_type="lane_drop")
    
    return scenario

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UCLF SCENARIO INITIALIZATION MODULE")
    print("="*60 + "\n")
    
    # Create and display scenario 1
    print("Creating Scenario 1: Lane Blockage")
    print("-"*60)
    scenario1 = create_scenario_1()
    scenario1.print_scenario_summary()
