import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from initialization_module import Vehicle, VehicleType, Obstacle
from state_management_module import VehicleStateManager

class UrgencyLevel(Enum):
    """Urgency levels for lane changing"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class CAVPriority:
    """Priority information for a CAV"""
    vehicle_id: int
    urgency_level: UrgencyLevel
    urgency_score: float
    reason: str
    distance_to_obstacle: float = float('inf')
    time_to_obstacle: float = float('inf')
    requires_lane_change: bool = False
    target_lane: int = None

class PriorityManager:
    """Manages prioritization of CAVs for cooperative lane changing"""
    
    def __init__(self, state_manager: VehicleStateManager):
        self.state_manager = state_manager
        self.priorities: Dict[int, CAVPriority] = {}
        self.priority_order: List[int] = []
        
    def calculate_urgency_score(self, vehicle: Vehicle, obstacles: List[Obstacle]) -> Tuple[float, str]:
        """Calculate urgency score for a CAV"""
        if vehicle.type != VehicleType.CAV:
            return 0.0, "Not a CAV"
            
        urgency_score = 0.0
        reasons = []
        
        # Distance to obstacle
        min_obstacle_distance = float('inf')
        obstacle_in_lane = False
        
        for obs in obstacles:
            if obs.lane == vehicle.lane:
                distance = obs.position - vehicle.position
                if distance > 0 and distance < min_obstacle_distance:
                    min_obstacle_distance = distance
                    obstacle_in_lane = True
                    
        if obstacle_in_lane and min_obstacle_distance < 100:
            obstacle_urgency = 100.0 / (min_obstacle_distance + 1)
            urgency_score += obstacle_urgency
            reasons.append(f"Obstacle ahead at {min_obstacle_distance:.1f}m")
                
        # Lane drop
        for obs in obstacles:
            if obs.type == "lane_drop" and obs.lane == vehicle.lane:
                distance = obs.position - vehicle.position
                if distance > 0 and distance < 200:
                    lane_drop_urgency = 50.0 / (distance + 1)
                    urgency_score += lane_drop_urgency
                    reasons.append(f"Lane ending at {distance:.1f}m")
                    
        # Vehicle ahead too close
        ahead_vehicle, distance_ahead = self.state_manager.get_vehicle_ahead(vehicle.id)
        if ahead_vehicle and distance_ahead < 20.0:
            proximity_urgency = 10.0 / (distance_ahead + 1)
            urgency_score += proximity_urgency
            reasons.append(f"Vehicle ahead at {distance_ahead:.1f}m")
            
        reason_str = "; ".join(reasons) if reasons else "No urgency"
        return urgency_score, reason_str
    
    def classify_urgency_level(self, urgency_score: float) -> UrgencyLevel:
        """Convert numeric urgency score to urgency level"""
        if urgency_score >= 50.0:
            return UrgencyLevel.CRITICAL
        elif urgency_score >= 20.0:
            return UrgencyLevel.HIGH
        elif urgency_score >= 10.0:
            return UrgencyLevel.MEDIUM
        elif urgency_score >= 5.0:
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.NONE
    
    def calculate_all_priorities(self, obstacles: List[Obstacle]):
        """Calculate priorities for all CAVs"""
        self.priorities.clear()
        
        cavs = self.state_manager.get_vehicles_by_type(VehicleType.CAV)
        
        for cav in cavs:
            urgency_score, reason = self.calculate_urgency_score(cav, obstacles)
            urgency_level = self.classify_urgency_level(urgency_score)
            
            # Calculate min distance to obstacle
            min_dist = float('inf')
            for obs in obstacles:
                if obs.lane == cav.lane:
                    dist = obs.position - cav.position
                    if dist > 0:
                        min_dist = min(min_dist, dist)
                            
            priority = CAVPriority(
                vehicle_id=cav.id,
                urgency_level=urgency_level,
                urgency_score=urgency_score,
                reason=reason,
                distance_to_obstacle=min_dist,
                requires_lane_change=(urgency_score > 5.0)
            )
            
            self.priorities[cav.id] = priority
            
        self.priority_order = sorted(
            self.priorities.keys(),
            key=lambda vid: self.priorities[vid].urgency_score,
            reverse=True
        )
    
    def get_priority_order(self) -> List[int]:
        """Get CAVs ordered by priority"""
        return self.priority_order.copy()
    
    def print_priorities(self):
        """Print priority information"""
        print("\n" + "="*100)
        print("CAV PRIORITY RANKING")
        print("="*100)
        print(f"{'Rank':<6} {'CAV ID':<8} {'Urgency':<12} {'Score':<8} {'Reason':<35}")
        print("-"*100)
        
        for rank, vehicle_id in enumerate(self.priority_order, 1):
            priority = self.priorities[vehicle_id]
            
            print(f"{rank:<6} {vehicle_id:<8} {priority.urgency_level.name:<12} "
                  f"{priority.urgency_score:>6.1f}  {priority.reason[:35]:<35}")
                  
        print("="*100 + "\n")
