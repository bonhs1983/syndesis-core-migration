# Enhanced TICKR Live Agent - Full Integration
import logging
import threading
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import random
import numpy as np

class TICKRLiveAgent:
    """Enhanced TICKR Live Agent με full integration capabilities"""
    
    def __init__(self):
        self.agent_id = "tickr_live_agent"
        self.role = "biometric_data_specialist"
        self.active = True
        
        # PC Connection Settings
        self.pc_server_url = None  # Will be set by user
        self.pc_connection_active = False
        self.last_pc_data = None
        self.pc_data_timeout = 10  # seconds
        
        # Simulation Settings για cloud mode
        self.simulation_mode = True
        self.base_hr = 75
        self.hr_variability = 5
        
        # Data Processing
        self.hrv_buffer = []
        self.buffer_size = 60
        
        logging.info("🎯 Enhanced TICKR Live Agent initialized")
    
    def set_pc_connection(self, pc_ip_address: str, port: int = 8080):
        """Set PC connection για remote TICKR access"""
        self.pc_server_url = f"http://{pc_ip_address}:{port}/tickr-data"
        logging.info(f"🔗 PC Connection configured: {self.pc_server_url}")
    
    def test_pc_connection(self) -> bool:
        """Test PC TICKR server connection"""
        if not self.pc_server_url:
            return False
        
        try:
            response = requests.get(self.pc_server_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('connected') and data.get('heart_rate'):
                    self.pc_connection_active = True
                    logging.info(f"✅ PC TICKR connected: {data['heart_rate']} BPM")
                    return True
            
            self.pc_connection_active = False
            return False
            
        except Exception as e:
            self.pc_connection_active = False
            logging.warning(f"⚠️ PC connection failed: {e}")
            return False
    
    def get_pc_tickr_data(self) -> Optional[Dict[str, Any]]:
        """Get real TICKR data from PC"""
        if not self.pc_connection_active or not self.pc_server_url:
            return None
        
        try:
            response = requests.get(self.pc_server_url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                if data.get('connected') and data.get('heart_rate'):
                    self.last_pc_data = {
                        'timestamp': datetime.now().isoformat(),
                        'heart_rate': data['heart_rate'],
                        'device_name': data.get('device_name', 'TICKR A204'),
                        'total_readings': data.get('total_readings', 0),
                        'data_source': 'PC_TICKR_AUTHENTIC'
                    }
                    return self.last_pc_data
                    
            return None
            
        except Exception as e:
            logging.warning(f"⚠️ PC data fetch failed: {e}")
            return None
    
    def generate_enhanced_simulation(self) -> Dict[str, Any]:
        """Enhanced simulation για cloud demos"""
        current_time = datetime.now()
        
        # Realistic HRV simulation
        base_variation = np.sin(time.time() / 10) * 3  # Slow variation
        breath_variation = np.sin(time.time() / 2) * 2  # Breathing cycle
        random_noise = random.uniform(-1.5, 1.5)
        
        simulated_hr = int(self.base_hr + base_variation + breath_variation + random_noise)
        simulated_hr = max(60, min(100, simulated_hr))  # Keep realistic bounds
        
        # Calculate HRV metrics
        hrv_rmssd = self._calculate_hrv_from_hr(simulated_hr)
        coherence = self._calculate_coherence_from_hr(simulated_hr)
        
        return {
            'timestamp': current_time.isoformat(),
            'heart_rate': simulated_hr,
            'hrv_rmssd': hrv_rmssd,
            'coherence': coherence,
            'data_source': 'ENHANCED_SIMULATION',
            'device_connected': False,
            'simulation_quality': 'enterprise_grade'
        }
    
    def generate_authentic_hrv_data(self) -> Dict[str, Any]:
        """Main method για HRV data generation"""
        
        # Try PC connection first
        if self.pc_connection_active:
            pc_data = self.get_pc_tickr_data()
            if pc_data:
                hr = pc_data['heart_rate']
                
                # Calculate HRV metrics from real HR
                hrv_rmssd = self._calculate_hrv_from_hr(hr)
                coherence = self._calculate_coherence_from_hr(hr)
                
                result = {
                    'timestamp': pc_data['timestamp'],
                    'heart_rate': hr,
                    'hrv_rmssd': hrv_rmssd,
                    'coherence': coherence,
                    'data_source': 'PC_TICKR_AUTHENTIC',
                    'device_connected': True,
                    'device_name': pc_data.get('device_name', 'TICKR A204'),
                    'total_readings': pc_data.get('total_readings', 0)
                }
                
                logging.info(f"🎯 PC TICKR Data: {hr} BPM (Authentic)")
                return result
        
        # Fallback to enhanced simulation
        simulation_data = self.generate_enhanced_simulation()
        logging.debug(f"🎭 Enhanced Simulation: {simulation_data['heart_rate']} BPM")
        return simulation_data
    
    def _calculate_hrv_from_hr(self, heart_rate: int) -> float:
        """Calculate HRV RMSSD from heart rate"""
        # Add to buffer για temporal analysis
        self.hrv_buffer.append(heart_rate)
        if len(self.hrv_buffer) > self.buffer_size:
            self.hrv_buffer.pop(0)
        
        if len(self.hrv_buffer) < 3:
            return 30.0  # Default HRV
        
        # Calculate RR intervals (simplified)
        rr_intervals = [60000 / hr for hr in self.hrv_buffer[-10:]]  # Last 10 readings
        
        if len(rr_intervals) < 2:
            return 30.0
        
        # RMSSD calculation
        successive_diffs = [abs(rr_intervals[i] - rr_intervals[i-1]) for i in range(1, len(rr_intervals))]
        
        if not successive_diffs:
            return 30.0
        
        mean_square = sum(d**2 for d in successive_diffs) / len(successive_diffs)
        rmssd = np.sqrt(mean_square)
        
        return round(rmssd, 2)
    
    def _calculate_coherence_from_hr(self, heart_rate: int) -> float:
        """Calculate coherence score from heart rate"""
        # Optimal HR range για coherence (60-80 BPM)
        optimal_range = (65, 78)
        
        if optimal_range[0] <= heart_rate <= optimal_range[1]:
            base_coherence = 0.8
        else:
            distance_from_optimal = min(
                abs(heart_rate - optimal_range[0]),
                abs(heart_rate - optimal_range[1])
            )
            base_coherence = max(0.1, 0.8 - (distance_from_optimal * 0.05))
        
        # Add variability based on HRV buffer
        if len(self.hrv_buffer) >= 5:
            recent_stability = np.std(self.hrv_buffer[-5:])
            stability_factor = max(0.5, 1.0 - (recent_stability * 0.02))
            base_coherence *= stability_factor
        
        return round(min(1.0, max(0.0, base_coherence)), 3)
    
    def start_enhanced_monitoring(self):
        """Start enhanced monitoring με PC auto-detection"""
        def monitoring_loop():
            while self.active:
                # Auto-test PC connection περιοδικά
                if not self.pc_connection_active and self.pc_server_url:
                    self.test_pc_connection()
                
                # Generate current data
                current_data = self.generate_authentic_hrv_data()
                
                # Log current status
                source = current_data['data_source']
                hr = current_data['heart_rate']
                
                if 'AUTHENTIC' in source:
                    logging.info(f"❤️ LIVE TICKR: {hr} BPM (Real Device)")
                else:
                    logging.debug(f"🎭 SIMULATION: {hr} BPM (Enhanced)")
                
                time.sleep(1)  # 1Hz monitoring
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logging.info("🌊 Enhanced TICKR monitoring started")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'active': self.active,
            'pc_connection_configured': bool(self.pc_server_url),
            'pc_connection_active': self.pc_connection_active,
            'simulation_mode': self.simulation_mode,
            'last_pc_data_age': (
                (datetime.now() - datetime.fromisoformat(self.last_pc_data['timestamp'])).seconds
                if self.last_pc_data else None
            )
        }

# Global instance
enhanced_tickr_agent = TICKRLiveAgent()

def initialize_tickr_agent():
    """Initialize the enhanced TICKR agent"""
    enhanced_tickr_agent.start_enhanced_monitoring()
    logging.info("🎯 Enhanced TICKR Live Agent ready for dual-mode operation")
    return enhanced_tickr_agent

# Auto-initialize
if __name__ == "__main__":
    agent = initialize_tickr_agent()
    time.sleep(60)  # Run for 1 minute για testing