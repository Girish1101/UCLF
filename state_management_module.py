import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from initialization_module import Vehicle, VehicleType, LaneAction, RoadConfig

@dataclass
class VehicleState:
    """Complete state information for a vehicle at a given timestep"""
    timestamp: float
    vehicle_id: int
    vehicle_type: VehicleType
    position: float
    lane: int
    velocity: float
    acceleration: float
    intended_action: LaneAction = LaneAction.KEEP_LANE

class StateHistory:
    """Maintains history of vehicle states over time"""
    def __init__(self):
        self.history: Dict[int, List[VehicleState]] = {}
        
    def add_state(self, vehicle_id: int, state: VehicleState):
        """Add a state snapshot for a vehicle"""
        if vehicle_id not in self.history:
            self.history[vehicle_id] = []
        self.history[vehicle_id].append(state)
        
    def get_vehicle_history(self, vehicle_id: int) -> List[VehicleState]:
        """Get all historical states for a specific vehicle"""
        return self.history.get(vehicle_id, [])
    
    def get_latest_state(self, vehicle_id: int) -> VehicleState:
        """Get most recent state for a vehicle"""
        if vehicle_id in self.history and self.history[vehicle_id]:
            return self.history[vehicle_id][-1]
        return None

class VehicleStateManager:
    """Manages all vehicle states and updates"""
    
    def __init__(self, road_config: RoadConfig):
        self.road_config = road_config
        self.vehicles: Dict[int, Vehicle] = {}
        self.current_time = 0.0
        self.dt = 0.1  # Time step (seconds)
        self.state_history = StateHistory()
        
    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to be managed"""
        self.vehicles[vehicle.id] = vehicle
        self._record_state(vehicle)
        
    def add_vehicles(self, vehicles: List[Vehicle]):
        """Add multiple vehicles"""
        for vehicle in vehicles:
            self.add_vehicle(vehicle)
            
    def _record_state(self, vehicle: Vehicle):
        """Record current state of a vehicle"""
        state = VehicleState(
            timestamp=self.current_time,
            vehicle_id=vehicle.id,
            vehicle_type=vehicle.type,
            position=vehicle.position,
            lane=vehicle.lane,
            velocity=vehicle.velocity,
            acceleration=vehicle.acceleration,
            intended_action=vehicle.intended_action
        )
        self.state_history.add_state(vehicle.id, state)
        
    def update_vehicle_position(self, vehicle_id: int, delta_time: float = None):
        """Update vehicle position based on current velocity and acceleration"""
        if delta_time is None:
            delta_time = self.dt
            
        vehicle = self.vehicles[vehicle_id]
        
        # Kinematic update
        vehicle.position += vehicle.velocity * delta_time + 0.5 * vehicle.acceleration * delta_time**2
        vehicle.velocity += vehicle.acceleration * delta_time
        vehicle.velocity = max(0.0, vehicle.velocity)
        
    def step_simulation(self, delta_time: float = None):
        """Advance simulation by one timestep"""
        if delta_time is None:
            delta_time = self.dt
            
        for vehicle_id in self.vehicles:
            self.update_vehicle_position(vehicle_id, delta_time)
            self._record_state(self.vehicles[vehicle_id])
            
        self.current_time += delta_time
        
    def get_vehicle(self, vehicle_id: int) -> Vehicle:
        """Get vehicle by ID"""
        return self.vehicles.get(vehicle_id)
    
    def get_all_vehicles(self) -> List[Vehicle]:
        """Get all vehicles"""
        return list(self.vehicles.values())
    
    def get_vehicles_by_type(self, vehicle_type: VehicleType) -> List[Vehicle]:
        """Get all vehicles of a specific type"""
        return [v for v in self.vehicles.values() if v.type == vehicle_type]
    
    def get_vehicles_in_lane(self, lane: int) -> List[Vehicle]:
        """Get all vehicles in a specific lane"""
        return [v for v in self.vehicles.values() if v.lane == lane]
    
    def get_vehicle_ahead(self, vehicle_id: int, same_lane_only: bool = True) -> Tuple[Vehicle, float]:
        """Get the vehicle directly ahead and distance to it"""
        vehicle = self.vehicles[vehicle_id]
        min_distance = float('inf')
        ahead_vehicle = None
        
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
                
            if same_lane_only and other_vehicle.lane != vehicle.lane:
                continue
                
            if other_vehicle.position > vehicle.position:
                distance = other_vehicle.position - vehicle.position
                if distance < min_distance:
                    min_distance = distance
                    ahead_vehicle = other_vehicle
                    
        return ahead_vehicle, min_distance
    
    def get_vehicle_behind(self, vehicle_id: int, same_lane_only: bool = True) -> Tuple[Vehicle, float]:
        """Get the vehicle directly behind and distance to it"""
        vehicle = self.vehicles[vehicle_id]
        min_distance = float('inf')
        behind_vehicle = None
        
        for other_id, other_vehicle in self.vehicles.items():
            if other_id == vehicle_id:
                continue
                
            if same_lane_only and other_vehicle.lane != vehicle.lane:
                continue
                
            if other_vehicle.position < vehicle.position:
                distance = vehicle.position - other_vehicle.position
                if distance < min_distance:
                    min_distance = distance
                    behind_vehicle = other_vehicle
                    
        return behind_vehicle, min_distance
    
    def print_current_state(self):
        """Print current state of all vehicles"""
        print("\n" + "="*80)
        print(f"VEHICLE STATES AT TIME: {self.current_time:.2f}s")
        print("="*80)
        print(f"{'ID':<5} {'Type':<6} {'Lane':<6} {'Position':<12} {'Velocity':<12} {'Accel':<10}")
        print("-"*80)
        
        for vehicle in sorted(self.vehicles.values(), key=lambda v: v.id):
            print(f"{vehicle.id:<5} {vehicle.type.name:<6} {vehicle.lane:<6} "
                  f"{vehicle.position:>10.2f}m  {vehicle.velocity:>10.2f}m/s  "
                  f"{vehicle.acceleration:>8.2f}m/s²")
        print("="*80 + "\n")
