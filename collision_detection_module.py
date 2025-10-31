import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from initialization_module import Vehicle, Obstacle
from state_management_module import VehicleStateManager

class CollisionType(Enum):
    """Types of collisions"""
    NO_COLLISION = 0
    REAR_END = 1
    SIDE_COLLISION = 2
    OBSTACLE_COLLISION = 3

@dataclass
class CollisionEvent:
    """Represents a collision event"""
    timestamp: float
    collision_type: CollisionType
    vehicle1_id: int
    vehicle2_id: int = None
    obstacle_id: int = None
    distance: float = 0.0
    severity: str = "WARNING"

class CollisionDetector:
    """Detects and reports collisions"""
    
    def __init__(self, state_manager: VehicleStateManager):
        self.state_manager = state_manager
        self.collision_history: List[CollisionEvent] = []
        self.safety_margin = 5.0  # Minimum safe distance (meters)
        
    def check_rear_end_collision(self, vehicle1: Vehicle, vehicle2: Vehicle) -> Tuple[bool, float]:
        """
        Check if two vehicles in same lane are too close (rear-end collision risk)
        Returns: (collision_detected, distance)
        """
        if vehicle1.lane != vehicle2.lane:
            return False, float('inf')
            
        # Determine which is ahead
        if vehicle2.position > vehicle1.position:
            ahead = vehicle2
            behind = vehicle1
        else:
            ahead = vehicle1
            behind = vehicle2
            
        distance = ahead.position - behind.position - ahead.length
        
        if distance < self.safety_margin:
            return True, distance
            
        return False, distance
    
    def check_side_collision(self, vehicle1: Vehicle, vehicle2: Vehicle) -> Tuple[bool, float]:
        """
        Check if two vehicles in adjacent lanes are at risk during lane change
        Returns: (collision_detected, lateral_distance)
        """
        # Check if lanes are adjacent
        if abs(vehicle1.lane - vehicle2.lane) != 1:
            return False, float('inf')
            
        # Check longitudinal overlap
        longitudinal_overlap = abs(vehicle1.position - vehicle2.position) < (vehicle1.length + vehicle2.length) / 2 + self.safety_margin
        
        if longitudinal_overlap:
            lateral_distance = abs(vehicle1.lane - vehicle2.lane)
            return True, lateral_distance
            
        return False, float('inf')
    
    def check_obstacle_collision(self, vehicle: Vehicle, obstacles: List[Obstacle]) -> Tuple[bool, int, float]:
        """
        Check if vehicle is at risk of colliding with obstacle
        Returns: (collision_detected, obstacle_id, distance)
        """
        for obs in obstacles:
            if obs.lane == vehicle.lane:
                distance = obs.position - vehicle.position
                
                if 0 < distance < self.safety_margin:
                    return True, obs.id, distance
                    
        return False, None, float('inf')
    
    def check_all_collisions(self, obstacles: List[Obstacle] = None) -> List[CollisionEvent]:
        """Check for all types of collisions in current state"""
        current_collisions = []
        vehicles = self.state_manager.get_all_vehicles()
        
        # Check vehicle-to-vehicle collisions
        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i+1:]:
                # Rear-end collision
                rear_end, distance = self.check_rear_end_collision(v1, v2)
                if rear_end:
                    severity = "CRITICAL" if distance < 2.0 else "WARNING"
                    event = CollisionEvent(
                        timestamp=self.state_manager.current_time,
                        collision_type=CollisionType.REAR_END,
                        vehicle1_id=v1.id,
                        vehicle2_id=v2.id,
                        distance=distance,
                        severity=severity
                    )
                    current_collisions.append(event)
                    
                # Side collision
                side_collision, lat_dist = self.check_side_collision(v1, v2)
                if side_collision:
                    event = CollisionEvent(
                        timestamp=self.state_manager.current_time,
                        collision_type=CollisionType.SIDE_COLLISION,
                        vehicle1_id=v1.id,
                        vehicle2_id=v2.id,
                        distance=lat_dist,
                        severity="WARNING"
                    )
                    current_collisions.append(event)
        
        # Check vehicle-obstacle collisions
        if obstacles:
            for vehicle in vehicles:
                collision, obs_id, distance = self.check_obstacle_collision(vehicle, obstacles)
                if collision:
                    severity = "CRITICAL" if distance < 2.0 else "WARNING"
                    event = CollisionEvent(
                        timestamp=self.state_manager.current_time,
                        collision_type=CollisionType.OBSTACLE_COLLISION,
                        vehicle1_id=vehicle.id,
                        obstacle_id=obs_id,
                        distance=distance,
                        severity=severity
                    )
                    current_collisions.append(event)
        
        # Add to history
        self.collision_history.extend(current_collisions)
        
        return current_collisions
    
    def is_lane_change_safe(self, vehicle_id: int, target_lane: int) -> bool:
        """Check if lane change to target lane is safe"""
        vehicle = self.state_manager.get_vehicle(vehicle_id)
        
        # Get vehicles in target lane
        target_lane_vehicles = self.state_manager.get_vehicles_in_lane(target_lane)
        
        for other in target_lane_vehicles:
            if other.id == vehicle_id:
                continue
                
            # Check distance
            distance = abs(other.position - vehicle.position)
            if distance < 3 * self.safety_margin:  # Need extra space for lane change
                return False
                
        return True
    
    def print_collision_report(self, collisions: List[CollisionEvent]):
        """Print collision events"""
        if not collisions:
            print("\n✓ No collisions detected")
            return
            
        print("\n" + "="*80)
        print("⚠️  COLLISION WARNINGS")
        print("="*80)
        
        for event in collisions:
            if event.collision_type == CollisionType.REAR_END:
                print(f"[{event.severity}] REAR-END: Vehicle {event.vehicle1_id} and {event.vehicle2_id} "
                      f"too close ({event.distance:.2f}m)")
            elif event.collision_type == CollisionType.SIDE_COLLISION:
                print(f"[{event.severity}] SIDE: Vehicle {event.vehicle1_id} and {event.vehicle2_id} "
                      f"risk during lane change")
            elif event.collision_type == CollisionType.OBSTACLE_COLLISION:
                print(f"[{event.severity}] OBSTACLE: Vehicle {event.vehicle1_id} approaching "
                      f"obstacle {event.obstacle_id} ({event.distance:.2f}m)")
                      
        print("="*80 + "\n")
    
    def get_collision_count(self) -> int:
        """Get total number of collisions detected"""
        return len(self.collision_history)
