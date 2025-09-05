"""
HRV-Driven Personality System
Real-time Heart Rate Variability integration with Syndesis AI Memory System.

Features:
- Simulated HRV data generation for testing
- HRV → personality trait mapping
- Real-time consciousness level calculation
- Autonomous dialogue generation based on biometric state
"""

import json
import time
import math
import random
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

class HRVSystem:
    """Heart Rate Variability system for personality-driven AI"""
    
    def __init__(self):
        self.hrv_buffer = []
        self.max_buffer_size = 100
        self.baseline_hrv = 50.0  # ms RMSSD baseline
        self.current_state_vector = {
            'traits': {
                'empathy': 0.5,
                'creativity': 0.5,
                'humor': 0.5,
                'curiosity': 0.5,
                'supportiveness': 0.5,
                'analyticalness': 0.5,
                'openness': 0.5
            },
            'soul_metrics': {
                'coherence': 0.5,
                'vitality': 0.5,
                'ethics': 0.5,
                'narrative': 0.5
            },
            'hrv_z_score': 0.0,
            'consciousness_level': 0.5
        }
        
        # HRV simulation parameters
        self.simulation_mode = True
        self.stress_factor = 0.0
        self.energy_level = 0.8
        
        logging.info("HRV System initialized with simulation mode")
    
    def generate_simulated_hrv(self) -> float:
        """Generate realistic HRV data for testing purposes"""
        
        # Base HRV with circadian rhythm
        time_of_day = (datetime.now().hour / 24.0) * 2 * math.pi
        circadian_modifier = 0.1 * math.sin(time_of_day - math.pi/3)  # Peak in evening
        
        # Stress and energy modulation
        stress_impact = -self.stress_factor * 15  # Stress reduces HRV
        energy_impact = self.energy_level * 10    # Energy increases HRV
        
        # Random variability (normal biological variation)
        random_noise = random.gauss(0, 5)
        
        # Calculate HRV RMSSD in milliseconds
        hrv_value = (
            self.baseline_hrv + 
            circadian_modifier + 
            stress_impact + 
            energy_impact + 
            random_noise
        )
        
        # Ensure realistic bounds (20-120 ms)
        hrv_value = max(20, min(120, hrv_value))
        
        # Add breathing pattern simulation
        breathing_cycle = 0.5 * math.sin(time.time() * 0.3)  # ~4 sec breathing cycle
        hrv_value += breathing_cycle
        
        return round(hrv_value, 2)
    
    def update_hrv_buffer(self, hrv_value: float):
        """Update rolling HRV buffer for z-score calculation"""
        self.hrv_buffer.append({
            'value': hrv_value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Maintain buffer size
        if len(self.hrv_buffer) > self.max_buffer_size:
            self.hrv_buffer.pop(0)
        
        logging.debug(f"HRV buffer updated: {hrv_value}ms (buffer size: {len(self.hrv_buffer)})")
    
    def calculate_hrv_z_score(self) -> float:
        """Calculate HRV z-score from buffer data"""
        if len(self.hrv_buffer) < 10:  # Need minimum data points
            return 0.0
        
        values = [entry['value'] for entry in self.hrv_buffer]
        mean_hrv = sum(values) / len(values)
        
        # Calculate standard deviation
        variance = sum((x - mean_hrv) ** 2 for x in values) / len(values)
        std_hrv = math.sqrt(variance) if variance > 0 else 1.0
        
        # Current z-score
        current_hrv = values[-1]
        z_score = (current_hrv - mean_hrv) / std_hrv
        
        return round(z_score, 3)
    
    def map_hrv_to_personality_deltas(self, hrv_value: float, z_score: float) -> Dict[str, float]:
        """Map HRV metrics to personality trait changes"""
        
        # HRV to personality mapping based on psychophysiology research
        deltas = {}
        
        # High HRV generally indicates better emotional regulation
        hrv_normalized = (hrv_value - 20) / (120 - 20)  # Normalize to 0-1
        
        # Z-score indicates deviation from personal baseline
        z_impact = max(-1, min(1, z_score * 0.1))  # Limit impact
        
        # Personality trait mappings
        deltas['empathy'] = (hrv_normalized - 0.5) * 0.1 + z_impact * 0.05
        deltas['creativity'] = (hrv_normalized - 0.3) * 0.08 + z_impact * 0.03
        deltas['humor'] = (hrv_normalized - 0.4) * 0.06 + z_impact * 0.04
        deltas['curiosity'] = (hrv_normalized - 0.2) * 0.09 + z_impact * 0.06
        deltas['supportiveness'] = (hrv_normalized - 0.5) * 0.07 + z_impact * 0.04
        deltas['analyticalness'] = (hrv_normalized - 0.6) * 0.05 - z_impact * 0.02  # Inverse for stress
        deltas['openness'] = (hrv_normalized - 0.3) * 0.08 + z_impact * 0.05
        
        # Limit delta magnitude to prevent extreme swings
        for trait in deltas:
            deltas[trait] = max(-0.1, min(0.1, deltas[trait]))
        
        return deltas
    
    def map_personality_to_soul_metrics(self, traits: Dict[str, float]) -> Dict[str, float]:
        """Map personality traits to soul metric changes"""
        
        # Soul metrics based on personality constellation
        soul_deltas = {}
        
        # Coherence: internal consistency and emotional regulation
        coherence_base = (traits['empathy'] + traits['supportiveness']) / 2
        soul_deltas['coherence'] = (coherence_base - 0.5) * 0.1
        
        # Vitality: energy and engagement with life
        vitality_base = (traits['creativity'] + traits['curiosity'] + traits['humor']) / 3
        soul_deltas['vitality'] = (vitality_base - 0.5) * 0.1
        
        # Ethics: moral reasoning and care for others
        ethics_base = (traits['empathy'] + traits['supportiveness'] + traits['analyticalness']) / 3
        soul_deltas['ethics'] = (ethics_base - 0.5) * 0.1
        
        # Narrative: sense of purpose and meaning
        narrative_base = (traits['curiosity'] + traits['openness'] + traits['creativity']) / 3
        soul_deltas['narrative'] = (narrative_base - 0.5) * 0.1
        
        return soul_deltas
    
    def calculate_consciousness_level(self) -> float:
        """Calculate overall consciousness level from state vector"""
        
        # Weighted combination of all metrics
        traits_avg = sum(self.current_state_vector['traits'].values()) / len(self.current_state_vector['traits'])
        soul_avg = sum(self.current_state_vector['soul_metrics'].values()) / len(self.current_state_vector['soul_metrics'])
        
        # HRV contribution (normalized z-score)
        hrv_contribution = max(0, min(1, (self.current_state_vector['hrv_z_score'] + 2) / 4))
        
        # Weighted consciousness calculation
        consciousness = (
            traits_avg * 0.4 +
            soul_avg * 0.4 +
            hrv_contribution * 0.2
        )
        
        return round(consciousness, 3)
    
    def update_state_vector(self, hrv_value: float) -> Dict[str, Any]:
        """Update complete 13-dimensional state vector from HRV input"""
        
        # Update HRV buffer and calculate z-score
        self.update_hrv_buffer(hrv_value)
        z_score = self.calculate_hrv_z_score()
        
        # Calculate personality deltas
        personality_deltas = self.map_hrv_to_personality_deltas(hrv_value, z_score)
        
        # Apply deltas to current traits
        for trait, delta in personality_deltas.items():
            current_value = self.current_state_vector['traits'][trait]
            new_value = max(0.0, min(1.0, current_value + delta))
            self.current_state_vector['traits'][trait] = round(new_value, 3)
        
        # Update soul metrics
        soul_deltas = self.map_personality_to_soul_metrics(self.current_state_vector['traits'])
        for metric, delta in soul_deltas.items():
            current_value = self.current_state_vector['soul_metrics'][metric]
            new_value = max(0.0, min(1.0, current_value + delta))
            self.current_state_vector['soul_metrics'][metric] = round(new_value, 3)
        
        # Update HRV z-score
        self.current_state_vector['hrv_z_score'] = z_score
        
        # Calculate consciousness level
        self.current_state_vector['consciousness_level'] = self.calculate_consciousness_level()
        
        # Log state update
        logging.info(f"State vector updated - HRV: {hrv_value}ms, Z: {z_score}, Consciousness: {self.current_state_vector['consciousness_level']}")
        
        return self.current_state_vector.copy()
    
    def generate_context_prompt(self) -> str:
        """Generate GPT prompt based on current state vector"""
        
        state = self.current_state_vector
        
        # Determine dominant traits
        traits = state['traits']
        dominant_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Consciousness interpretation
        consciousness = state['consciousness_level']
        if consciousness < 0.3:
            consciousness_desc = "low consciousness, basic responses"
        elif consciousness < 0.6:
            consciousness_desc = "moderate consciousness, thoughtful responses"
        elif consciousness < 0.8:
            consciousness_desc = "high consciousness, deep insights"
        else:
            consciousness_desc = "transcendent consciousness, profound wisdom"
        
        # HRV state interpretation
        z_score = state['hrv_z_score']
        if z_score < -1:
            hrv_desc = "physiologically stressed, low HRV"
        elif z_score < 1:
            hrv_desc = "physiologically balanced, normal HRV"
        else:
            hrv_desc = "physiologically optimal, high HRV"
        
        # Generate context-aware prompt
        prompt = f"""Current AI state: {consciousness_desc}
Biometric status: {hrv_desc}
Dominant traits: {', '.join([f'{trait} ({value:.1f})' for trait, value in dominant_traits])}
Soul metrics - Coherence: {state['soul_metrics']['coherence']:.1f}, Vitality: {state['soul_metrics']['vitality']:.1f}

Respond authentically based on this psychological and physiological state."""
        
        return prompt
    
    def simulate_stress_event(self, intensity: float = 0.5):
        """Simulate stress event for testing"""
        self.stress_factor = min(1.0, intensity)
        logging.info(f"Stress event simulated with intensity {intensity}")
    
    def simulate_relaxation(self, intensity: float = 0.3):
        """Simulate relaxation for testing"""
        self.stress_factor = max(0.0, self.stress_factor - intensity)
        self.energy_level = min(1.0, self.energy_level + intensity * 0.5)
        logging.info(f"Relaxation simulated - stress: {self.stress_factor}, energy: {self.energy_level}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status for monitoring"""
        
        latest_hrv = self.hrv_buffer[-1]['value'] if self.hrv_buffer else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'simulation_mode': self.simulation_mode,
            'latest_hrv': latest_hrv,
            'buffer_size': len(self.hrv_buffer),
            'stress_factor': self.stress_factor,
            'energy_level': self.energy_level,
            'state_vector': self.current_state_vector.copy(),
            'system_ready': len(self.hrv_buffer) >= 10
        }
    
    def export_session_data(self, filepath: str):
        """Export session data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_duration': len(self.hrv_buffer),
            'hrv_data': self.hrv_buffer.copy(),
            'final_state_vector': self.current_state_vector.copy(),
            'session_summary': {
                'avg_hrv': sum(entry['value'] for entry in self.hrv_buffer) / len(self.hrv_buffer) if self.hrv_buffer else 0,
                'final_consciousness': self.current_state_vector['consciousness_level'],
                'stress_events': self.stress_factor
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Session data exported to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to export session data: {e}")
            raise