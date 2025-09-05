"""
Revolutionary Neural HRV System for Syndesis
Integrates real neural networks with existing personality and memory systems
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import the neural modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hrv_neural'))

# Define fallback constants
HRV_FEATURES = ['meanRR', 'SDNN', 'RMSSD', 'LF', 'HF', 'LF_HF']
CHAPTERS = ['PRESENCE', 'QUALITY', 'FORGIVENESS', 'AWARENESS', 'COMPASSION', 'HARMONY']

try:
    from evolution import GRUDynamics, NeuralODEDynamics
    from representation import VAE
    from manifold import UMAP
    from mapping import HRV_FEATURES as IMPORTED_HRV_FEATURES, CHAPTERS as IMPORTED_CHAPTERS
    # Use imported constants if available
    HRV_FEATURES = IMPORTED_HRV_FEATURES
    CHAPTERS = IMPORTED_CHAPTERS
    NEURAL_MODULES_AVAILABLE = True
except ImportError:
    logging.warning("Neural HRV modules not available, using fallback")
    NEURAL_MODULES_AVAILABLE = False

@dataclass
class NeuralHRVReading:
    """Enhanced HRV reading with neural analysis"""
    timestamp: datetime
    raw_features: Dict[str, float]  # meanRR, SDNN, RMSSD, LF, HF, LF_HF
    neural_embedding: np.ndarray    # VAE latent representation
    risk_score: float              # Manifold-based risk assessment
    state_trajectory: np.ndarray   # Neural ODE predicted trajectory
    chapter_activations: Dict[str, float]  # Book chapter mappings

class NeuralHRVSystem:
    """
    Revolutionary HRV system combining:
    - Real neural networks for pattern learning
    - VAE representations for personalized baselines
    - Neural ODEs for smooth state transitions
    - Risk detection via manifold learning
    - Integration with Soul Metrics and Memory Systems
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.hrv_buffer = []
        self.max_buffer_size = 1000
        
        # Initialize neural components
        self.vae_model = None
        self.gru_dynamics = None
        self.neural_ode = None
        self.manifold_analyzer = None
        
        # Feature mappings
        self.hrv_features = HRV_FEATURES
        self.book_chapters = CHAPTERS
        self.feature_to_chapter = dict(zip(self.hrv_features, self.book_chapters))
        
        # Personalization data
        self.personal_baseline = None
        self.risk_threshold = 2.0  # Z-score threshold for risk detection
        self.adaptation_rate = 0.1
        
        self._initialize_neural_models()
        
    def _initialize_neural_models(self):
        """Initialize all neural network components"""
        try:
            if NEURAL_MODULES_AVAILABLE:
                # VAE for learning personal HRV representations
                self.vae_model = VAE(in_dim=6, z_dim=8)  # 6 HRV features -> 8D latent
                
                # GRU for sequence modeling
                self.gru_dynamics = GRUDynamics(hrv_dim=6, latent_dim=32)
                
                # Neural ODE for continuous dynamics
                self.neural_ode = NeuralODEDynamics(hrv_dim=6, latent_dim=32)
                
                logging.info("Neural HRV models initialized successfully")
            else:
                self._use_fallback_models()
                
        except Exception as e:
            logging.warning(f"Could not initialize neural models: {e}")
            self._use_fallback_models()
    
    def _use_fallback_models(self):
        """Fallback to simple models if neural components fail"""
        self.vae_model = None
        self.gru_dynamics = None
        self.neural_ode = None
        logging.info("Using fallback HRV analysis")
    
    def process_hrv_reading(self, hrv_data: Dict[str, float]) -> NeuralHRVReading:
        """
        Process a new HRV reading through the neural pipeline
        
        Args:
            hrv_data: Dict with keys from HRV_FEATURES
            
        Returns:
            NeuralHRVReading with full neural analysis
        """
        timestamp = datetime.now()
        
        # Convert to tensor
        feature_vector = self._dict_to_tensor(hrv_data)
        
        # Neural analysis
        neural_embedding = self._get_neural_embedding(feature_vector)
        risk_score = self._calculate_risk_score(feature_vector, neural_embedding)
        state_trajectory = self._predict_trajectory(feature_vector)
        chapter_activations = self._map_to_chapters(hrv_data, neural_embedding)
        
        # Create reading object
        reading = NeuralHRVReading(
            timestamp=timestamp,
            raw_features=hrv_data,
            neural_embedding=neural_embedding,
            risk_score=risk_score,
            state_trajectory=state_trajectory,
            chapter_activations=chapter_activations
        )
        
        # Update buffer and adapt models
        self.hrv_buffer.append(reading)
        if len(self.hrv_buffer) > self.max_buffer_size:
            self.hrv_buffer.pop(0)
            
        self._adaptive_learning(reading)
        
        return reading
    
    def _dict_to_tensor(self, hrv_data: Dict[str, float]) -> torch.Tensor:
        """Convert HRV dict to ordered tensor"""
        values = [hrv_data.get(feature, 0.0) for feature in self.hrv_features]
        return torch.tensor(values, dtype=torch.float32)
    
    def _get_neural_embedding(self, feature_vector: torch.Tensor) -> np.ndarray:
        """Get VAE embedding of HRV features"""
        if self.vae_model is None:
            # Fallback: simple dimensionality reduction
            return feature_vector.numpy()[:4]  # First 4 features
            
        try:
            with torch.no_grad():
                mu, _ = self.vae_model.encode(feature_vector.unsqueeze(0))
                return mu.squeeze().numpy()
        except Exception as e:
            logging.warning(f"VAE encoding failed: {e}")
            return feature_vector.numpy()[:4]
    
    def _calculate_risk_score(self, features: torch.Tensor, embedding: np.ndarray) -> float:
        """Calculate risk score using manifold analysis"""
        if len(self.hrv_buffer) < 50:  # Not enough data for baseline
            return 0.0
            
        # Get recent embeddings for comparison
        recent_embeddings = np.array([r.neural_embedding for r in self.hrv_buffer[-50:]])
        
        # Calculate z-score distance from recent average
        mean_embedding = np.mean(recent_embeddings, axis=0)
        std_embedding = np.std(recent_embeddings, axis=0) + 1e-6  # Prevent division by zero
        
        z_scores = np.abs((embedding - mean_embedding) / std_embedding)
        risk_score = np.max(z_scores)  # Highest deviation
        
        return float(risk_score)
    
    def _predict_trajectory(self, features: torch.Tensor) -> np.ndarray:
        """Predict future state trajectory using Neural ODE"""
        if self.neural_ode is None or len(self.hrv_buffer) < 10:
            # Fallback: simple linear extrapolation
            if len(self.hrv_buffer) >= 2:
                last_two = [r.raw_features for r in self.hrv_buffer[-2:]]
                delta = np.array(list(last_two[1].values())) - np.array(list(last_two[0].values()))
                return delta * 5  # Project 5 steps ahead
            return np.zeros(6)
        
        try:
            # Use recent sequence for prediction
            recent_sequence = torch.stack([
                self._dict_to_tensor(r.raw_features) 
                for r in self.hrv_buffer[-10:]
            ]).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                if hasattr(self.neural_ode, 'forward'):
                    trajectory = self.neural_ode(recent_sequence, features.unsqueeze(0))
                    return trajectory.squeeze().numpy()
                else:
                    return np.zeros(32)  # Latent dimension
                    
        except Exception as e:
            logging.warning(f"Trajectory prediction failed: {e}")
            return np.zeros(6)
    
    def _map_to_chapters(self, hrv_data: Dict[str, float], embedding: np.ndarray) -> Dict[str, float]:
        """Map HRV state to Book Center chapter activations"""
        activations = {}
        
        # Direct mapping from HRV features to chapters
        for feature, chapter in self.feature_to_chapter.items():
            value = hrv_data.get(feature, 0.0)
            
            # Normalize to 0-1 range (assuming reasonable HRV ranges)
            if feature == 'meanRR':
                normalized = min(1.0, max(0.0, (value - 600) / 400))  # 600-1000ms range
            elif feature in ['SDNN', 'RMSSD']:
                normalized = min(1.0, max(0.0, value / 100))  # 0-100ms range
            elif feature in ['LF', 'HF']:
                normalized = min(1.0, max(0.0, value / 1000))  # 0-1000 units
            elif feature == 'LF_HF':
                normalized = min(1.0, max(0.0, value / 5))  # 0-5 ratio
            else:
                normalized = 0.5
                
            activations[chapter] = normalized
        
        # Enhance with neural embedding information
        if len(embedding) >= 6:
            for i, chapter in enumerate(self.book_chapters):
                # Use embedding dimensions to modulate activations
                embedding_influence = (embedding[i % len(embedding)] + 1) / 2  # Normalize to 0-1
                activations[chapter] = 0.7 * activations[chapter] + 0.3 * embedding_influence
        
        return activations
    
    def _adaptive_learning(self, reading: NeuralHRVReading):
        """Continuously adapt models based on new data"""
        if len(self.hrv_buffer) % 100 == 0 and self.vae_model is not None:
            # Periodic retraining of VAE with recent data
            try:
                recent_data = torch.stack([
                    self._dict_to_tensor(r.raw_features) 
                    for r in self.hrv_buffer[-500:]  # Last 500 readings
                ])
                
                # Simple online learning step
                self.vae_model.train()
                optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=1e-4)
                
                for _ in range(5):  # Few quick epochs
                    optimizer.zero_grad()
                    x_hat, mu, logvar = self.vae_model(recent_data)
                    recon_loss = nn.functional.mse_loss(x_hat, recent_data)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    loss.backward()
                    optimizer.step()
                
                self.vae_model.eval()
                logging.info(f"Adapted VAE model with {len(recent_data)} samples")
                
            except Exception as e:
                logging.warning(f"Adaptive learning failed: {e}")
    
    def get_current_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive current state summary"""
        if not self.hrv_buffer:
            return {"status": "no_data"}
            
        latest = self.hrv_buffer[-1]
        
        # Calculate trends from recent readings
        recent_count = min(10, len(self.hrv_buffer))
        recent_readings = self.hrv_buffer[-recent_count:]
        
        trends = {}
        for feature in self.hrv_features:
            values = [r.raw_features.get(feature, 0) for r in recent_readings]
            if len(values) >= 2:
                trends[feature] = (values[-1] - values[0]) / len(values)
            else:
                trends[feature] = 0.0
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "current_features": latest.raw_features,
            "neural_embedding": latest.neural_embedding.tolist(),
            "risk_score": latest.risk_score,
            "risk_level": "high" if latest.risk_score > self.risk_threshold else "normal",
            "chapter_activations": latest.chapter_activations,
            "trends": trends,
            "buffer_size": len(self.hrv_buffer),
            "adaptation_cycles": len(self.hrv_buffer) // 100
        }
    
    def generate_biofeedback_signals(self, reading: NeuralHRVReading) -> Dict[str, Any]:
        """Generate multimodal biofeedback signals"""
        
        # Sound generation based on HRV coherence
        coherence = reading.raw_features.get('RMSSD', 50) / 100  # Normalize
        frequency = 220 + coherence * 220  # 220-440 Hz range
        
        # Light generation based on chapter activations
        dominant_chapter = max(reading.chapter_activations.items(), key=lambda x: x[1])
        
        # Color mapping for chapters
        chapter_colors = {
            'Breathing': (0, 150, 255),    # Blue
            'Stress': (255, 100, 100),     # Red
            'Recovery': (100, 255, 100),   # Green
            'Energy': (255, 255, 0),       # Yellow
            'Focus': (150, 0, 255),        # Purple
            'Balance': (255, 150, 0)       # Orange
        }
        
        color = chapter_colors.get(dominant_chapter[0], (255, 255, 255))
        intensity = dominant_chapter[1]
        
        return {
            "sound": {
                "frequency": frequency,
                "volume": min(1.0, coherence),
                "type": "sine_wave"
            },
            "light": {
                "color_rgb": color,
                "intensity": intensity,
                "pattern": "breathing" if reading.risk_score < 1.0 else "alert"
            },
            "haptic": {
                "strength": reading.risk_score / 5.0,
                "pattern": "steady" if reading.risk_score < 1.0 else "pulse"
            }
        }

# Global instance
neural_hrv_system = NeuralHRVSystem()