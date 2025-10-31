import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from initialization_module import Vehicle, VehicleType, LaneAction
from state_management_module import VehicleStateManager

@dataclass
class BehaviorPrediction:
    """Predicted behavior for an HDV"""
    vehicle_id: int
    predicted_action: LaneAction
    confidence: float
    all_probabilities: Dict[LaneAction, float]
    timestamp: float

class HDVBehaviorPredictor:
    """Predicts HDV behavior based on context and probabilistic models"""
    
    def __init__(self, state_manager: VehicleStateManager):
        self.state_manager = state_manager
        self.prediction_history: Dict[int, List[BehaviorPrediction]] = {}
        
    def estimate_lane_change_probability(self, vehicle: Vehicle, obstacles: List = None) -> Dict[LaneAction, float]:
        """Estimate probability of different lane actions based on context"""
        base_probs = vehicle.action_probabilities.copy()
        
        ahead_vehicle, distance_ahead = self.state_manager.get_vehicle_ahead(vehicle.id)
        road_config = self.state_manager.road_config
        
        # Factor 1: Distance to vehicle ahead
        if ahead_vehicle and distance_ahead < 30.0:
            base_probs[LaneAction.KEEP_LANE] *= 0.7
            base_probs[LaneAction.CHANGE_LEFT] *= 1.3
            base_probs[LaneAction.CHANGE_RIGHT] *= 1.3
            
        # Factor 2: Speed difference
        if ahead_vehicle and vehicle.velocity > ahead_vehicle.velocity + 5.0:
            base_probs[LaneAction.KEEP_LANE] *= 0.6
            base_probs[LaneAction.CHANGE_LEFT] *= 1.5
            base_probs[LaneAction.CHANGE_RIGHT] *= 1.5
            
        # Factor 3: Lane constraints
        if vehicle.lane == 1:
            base_probs[LaneAction.CHANGE_LEFT] = 0.0
        if vehicle.lane == road_config.num_lanes:
            base_probs[LaneAction.CHANGE_RIGHT] = 0.0
            
        # Factor 4: Obstacles
        if obstacles:
            for obs in obstacles:
                if obs.lane == vehicle.lane:
                    distance_to_obs = obs.position - vehicle.position
                    if 0 < distance_to_obs < 50.0:
                        base_probs[LaneAction.KEEP_LANE] *= 0.2
                        if vehicle.lane > 1:
                            base_probs[LaneAction.CHANGE_LEFT] *= 2.0
                        if vehicle.lane < road_config.num_lanes:
                            base_probs[LaneAction.CHANGE_RIGHT] *= 2.0
                            
        # Normalize
        total = sum(base_probs.values())
        if total > 0:
            base_probs = {k: v/total for k, v in base_probs.items()}
        else:
            base_probs = {
                LaneAction.KEEP_LANE: 1.0,
                LaneAction.CHANGE_LEFT: 0.0,
                LaneAction.CHANGE_RIGHT: 0.0
            }
            
        return base_probs
    
    def predict_hdv_action(self, vehicle_id: int, obstacles: List = None) -> BehaviorPrediction:
        """Predict the most likely action for an HDV"""
        vehicle = self.state_manager.get_vehicle(vehicle_id)
        
        if vehicle.type != VehicleType.HDV:
            raise ValueError(f"Vehicle {vehicle_id} is not an HDV")
            
        probs = self.estimate_lane_change_probability(vehicle, obstacles)
        predicted_action = max(probs, key=probs.get)
        confidence = probs[predicted_action]
        
        prediction = BehaviorPrediction(
            vehicle_id=vehicle_id,
            predicted_action=predicted_action,
            confidence=confidence,
            all_probabilities=probs,
            timestamp=self.state_manager.current_time
        )
        
        if vehicle_id not in self.prediction_history:
            self.prediction_history[vehicle_id] = []
        self.prediction_history[vehicle_id].append(prediction)
        
        return prediction
    
    def sample_hdv_behavior(self, vehicle_id: int, obstacles: List = None, num_samples: int = 1) -> List[LaneAction]:
        """Sample possible HDV behaviors based on probability distribution"""
        vehicle = self.state_manager.get_vehicle(vehicle_id)
        
        if vehicle.type != VehicleType.HDV:
            raise ValueError(f"Vehicle {vehicle_id} is not an HDV")
            
        probs = self.estimate_lane_change_probability(vehicle, obstacles)
        
        actions = list(probs.keys())
        probabilities = list(probs.values())
        
        samples = random.choices(actions, weights=probabilities, k=num_samples)
        
        return samples
    
    def predict_all_hdvs(self, obstacles: List = None) -> Dict[int, BehaviorPrediction]:
        """Predict behavior for all HDVs in the scenario"""
        predictions = {}
        
        hdvs = self.state_manager.get_vehicles_by_type(VehicleType.HDV)
        
        for hdv in hdvs:
            prediction = self.predict_hdv_action(hdv.id, obstacles)
            predictions[hdv.id] = prediction
            
        return predictions
    
    def print_predictions(self, predictions: Dict[int, BehaviorPrediction]):
        """Print HDV behavior predictions"""
        print("\n" + "="*80)
        print("HDV BEHAVIOR PREDICTIONS")
        print("="*80)
        print(f"{'Vehicle':<10} {'Predicted Action':<20} {'Confidence':<12} {'Probabilities':<30}")
        print("-"*80)
        
        for vehicle_id, pred in predictions.items():
            probs_str = f"K:{pred.all_probabilities[LaneAction.KEEP_LANE]:.0%} " \
                       f"L:{pred.all_probabilities[LaneAction.CHANGE_LEFT]:.0%} " \
                       f"R:{pred.all_probabilities[LaneAction.CHANGE_RIGHT]:.0%}"
            
            print(f"HDV {vehicle_id:<5} {pred.predicted_action.name:<20} "
                  f"{pred.confidence:>10.0%}  {probs_str:<30}")
        print("="*80 + "\n")
