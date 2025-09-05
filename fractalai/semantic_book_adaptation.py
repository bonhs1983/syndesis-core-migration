"""
Semantic Book Adaptation Engine for HRV Galaxy
Connects book chapters to Book Centers through semantic embedding and HRV adaptation
"""

import numpy as np
import openai
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI client
openai.api_key = os.environ.get("OPENAI_API_KEY")

@dataclass
class BookCenter:
    """Represents a Book Center with its semantic properties"""
    name: str
    description: str
    embedding: np.ndarray
    hrv_triggers: Dict[str, float]  # HRV thresholds for activation
    coach_prompts: List[str]
    target_metrics: Dict[str, float]  # Target personality metrics
    color: str
    position: Tuple[int, int]

@dataclass
class ChapterContext:
    """Context for current chapter being read"""
    book_id: str
    chapter_index: int
    title: str
    text_preview: str
    embedding: np.ndarray
    active_center: str
    similarity_scores: Dict[str, float]
    timestamp: datetime

class SemanticBookAdaptation:
    """
    AI Adaptation engine that κουμπώνει στο λογικό σχήμα κάθε κεφαλαίου/Book-Center
    """
    
    def __init__(self):
        self.book_centers = self._initialize_book_centers()
        self.current_context: Optional[ChapterContext] = None
        self.adaptation_gamma = 0.4  # Mixing coefficient for latent state
        self.hrv_history = []
        
        logging.info("🧠 Semantic Book Adaptation Engine initialized")
    
    def _initialize_book_centers(self) -> Dict[str, BookCenter]:
        """Initialize the 7 Book Centers with semantic embeddings"""
        centers_config = {
            "MINDFULNESS": {
                "description": "Present moment awareness, meditation, conscious breathing, mindful attention",
                "hrv_triggers": {"coherence": 0.6, "stress": 0.4},
                "coach_prompts": [
                    "Take 3 slow breaths for mindfulness",
                    "Focus on your breathing",
                    "Observe the moment without judgment"
                ],
                "target_metrics": {"focus": 0.8, "empathy": 0.7, "compassion": 0.8},
                "color": "#4facfe",
                "position": (25, 20)
            },
            "BALANCE": {
                "description": "Emotional equilibrium, work-life harmony, centered stability, inner peace",
                "hrv_triggers": {"lf_hf_ratio": 2.5, "rmssd": 30},
                "coach_prompts": [
                    "Try a mini body scan for balance",
                    "Adjust your posture",
                    "Find your center"
                ],
                "target_metrics": {"resilience": 0.8, "focus": 0.6, "compassion": 0.6},
                "color": "#43e97b",
                "position": (75, 15)
            },
            "AWARENESS": {
                "description": "Self-consciousness, recognition, understanding, insight, perception",
                "hrv_triggers": {"coherence": 0.5, "variability": 0.7},
                "coach_prompts": [
                    "Observe your thoughts",
                    "What are you feeling right now?",
                    "Explore your current experience"
                ],
                "target_metrics": {"curiosity": 0.9, "empathy": 0.7, "focus": 0.7},
                "color": "#fa709a",
                "position": (15, 60)
            },
            "COMPASSION": {
                "description": "Loving-kindness, empathy, care, understanding, connection with others",
                "hrv_triggers": {"hf_power": 200, "coherence": 0.7},
                "coach_prompts": [
                    "Send love to yourself",
                    "Connect with your heart",
                    "Care for someone else"
                ],
                "target_metrics": {"empathy": 0.9, "compassion": 0.9, "creativity": 0.6},
                "color": "#fee140",
                "position": (85, 65)
            },
            "PRESENCE": {
                "description": "Being fully here now, conscious existence, authentic self-expression",
                "hrv_triggers": {"coherence": 0.8, "stress": 0.3},
                "coach_prompts": [
                    "Become fully present",
                    "Feel your aliveness",
                    "Express yourself authentically"
                ],
                "target_metrics": {"focus": 0.9, "resilience": 0.7, "creativity": 0.8},
                "color": "#667eea",
                "position": (50, 80)
            },
            "QUALITY": {
                "description": "Excellence, mastery, attention to detail, craftsmanship, refinement",
                "hrv_triggers": {"focus_index": 0.7, "variability": 0.6},
                "coach_prompts": [
                    "Focus on quality",
                    "Do it better",
                    "Pay attention to details"
                ],
                "target_metrics": {"focus": 0.9, "curiosity": 0.7, "resilience": 0.8},
                "color": "#764ba2",
                "position": (20, 35)
            },
            "ENERGY": {
                "description": "Vitality, motivation, dynamic action, life force, enthusiasm",
                "hrv_triggers": {"lf_power": 400, "vitality": 0.8},
                "coach_prompts": [
                    "Breathe in energy",
                    "Activate your power",
                    "Move with vitality"
                ],
                "target_metrics": {"creativity": 0.8, "resilience": 0.9, "curiosity": 0.7},
                "color": "#f093fb",
                "position": (80, 40)
            }
        }
        
        centers = {}
        for name, config in centers_config.items():
            # Generate embedding for each center description
            embedding = self._get_text_embedding(config["description"])
            
            centers[name] = BookCenter(
                name=name,
                description=config["description"],
                embedding=embedding,
                hrv_triggers=config["hrv_triggers"],
                coach_prompts=config["coach_prompts"],
                target_metrics=config["target_metrics"],
                color=config["color"],
                position=config["position"]
            )
        
        return centers
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text"""
        try:
            # the newest OpenAI model is "text-embedding-3-small" which was released after your knowledge cutoff.
            # do not change this unless explicitly requested by the user
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text[:2000],  # Limit to 2000 chars as specified
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logging.warning(f"Embedding failed: {e}")
            return np.random.rand(1536)  # Fallback random embedding
    
    def process_chapter_reading(self, book_id: str, chapter_index: int, title: str, text_preview: str) -> ChapterContext:
        """
        Process new chapter being read - core semantic matching function
        """
        # 1. Semantic embedding του κεφαλαίου
        chapter_embedding = self._get_text_embedding(text_preview)
        
        # 2. Similarity → Center-tag
        similarity_scores = {}
        for center_name, center in self.book_centers.items():
            similarity = cosine_similarity(
                chapter_embedding.reshape(1, -1),
                center.embedding.reshape(1, -1)
            )[0][0]
            similarity_scores[center_name] = float(similarity)
        
        # 3. Find the center with highest similarity score
        active_center = max(similarity_scores.keys(), key=lambda x: similarity_scores[x])
        
        # Create context
        context = ChapterContext(
            book_id=book_id,
            chapter_index=chapter_index,
            title=title,
            text_preview=text_preview,
            embedding=chapter_embedding,
            active_center=active_center,
            similarity_scores=similarity_scores,
            timestamp=datetime.now()
        )
        
        self.current_context = context
        
        logging.info(f"📖 Chapter '{title}' mapped to {active_center} (similarity: {similarity_scores[active_center]:.3f})")
        
        return context
    
    def adapt_to_hrv_state(self, hrv_data: Dict[str, float]) -> Dict[str, any]:
        """
        Context vector στον AI Adaptation - adapts system based on HRV + active center
        """
        if not self.current_context:
            return {"error": "No active chapter context"}
        
        active_center = self.book_centers[self.current_context.active_center]
        self.hrv_history.append(hrv_data)
        
        # Keep only recent history
        if len(self.hrv_history) > 100:
            self.hrv_history = self.hrv_history[-50:]
        
        # 4. Context vector στον AI Adaptation
        center_vector = self._create_center_one_hot(self.current_context.active_center)
        
        # Mix with current state (γ≈0.4)
        # state.latent = γ·ctx + (1-γ)·state.latent
        adaptation_signal = {
            "active_center": self.current_context.active_center,
            "center_vector": center_vector.tolist(),
            "mixing_gamma": self.adaptation_gamma,
            "target_metrics": active_center.target_metrics
        }
        
        # 5. Κανόνες coach-prompt
        coach_prompt = self._generate_contextual_coach_prompt(hrv_data, active_center)
        
        # 6. Γραφική ανάδραση
        visual_feedback = self._generate_visual_feedback(hrv_data, active_center)
        
        # 7. Προσαρμογή metric weights
        metric_adjustments = self._calculate_metric_adjustments(hrv_data, active_center)
        
        return {
            "success": True,
            "adaptation_signal": adaptation_signal,
            "coach_prompt": coach_prompt,
            "visual_feedback": visual_feedback,
            "metric_adjustments": metric_adjustments,
            "hrv_state": hrv_data,
            "context": {
                "book_id": self.current_context.book_id,
                "chapter": self.current_context.title,
                "active_center": self.current_context.active_center,
                "similarity_scores": self.current_context.similarity_scores
            }
        }
    
    def _create_center_one_hot(self, active_center: str) -> np.ndarray:
        """Create one-hot encoding for active center"""
        center_names = list(self.book_centers.keys())
        one_hot = np.zeros(len(center_names))
        if active_center in center_names:
            one_hot[center_names.index(active_center)] = 1.0
        return one_hot
    
    def _generate_contextual_coach_prompt(self, hrv_data: Dict[str, float], center: BookCenter) -> Optional[str]:
        """Generate contextual coach prompts based on HRV + center rules"""
        
        # Example rules as specified:
        coherence = hrv_data.get('coherence', 0.5)
        lf_hf_ratio = hrv_data.get('LF_HF', 1.5)
        stress_level = hrv_data.get('stress', 0.5)
        
        # If HRV Coherence ↓ and activeCenter == Mindfulness → suggest slow breaths
        if center.name == "MINDFULNESS" and coherence < center.hrv_triggers.get('coherence', 0.6):
            return "Take 3 slow breaths for mindfulness"
        
        # If LF/HF ↑ and center == Balance → suggest mini body scan
        if center.name == "BALANCE" and lf_hf_ratio > center.hrv_triggers.get('lf_hf_ratio', 2.5):
            return "Try a mini body scan for balance"
        
        # High stress situations
        if stress_level > 0.7 and center.coach_prompts:
            return center.coach_prompts[0]  # Return first prompt for high stress
        
        # Default: random prompt from center's collection
        if center.coach_prompts and np.random.random() > 0.7:  # 30% chance to show prompt
            import random
            return random.choice(center.coach_prompts)
        
        return None
    
    def _generate_visual_feedback(self, hrv_data: Dict[str, float], center: BookCenter) -> Dict[str, any]:
        """Generate visual feedback for Galaxy interface"""
        coherence = hrv_data.get('coherence', 0.5)
        
        return {
            "center_highlight": {
                "name": center.name,
                "color": center.color,
                "intensity": min(1.0, coherence * 1.5),  # Scale highlight with coherence
                "position": center.position
            },
            "you_node_target": {
                "x": center.position[0] + np.random.normal(0, 5),  # Add slight variation
                "y": center.position[1] + np.random.normal(0, 5),
                "connection_thickness": 2 + coherence * 3  # Thicker line with higher coherence
            },
            "progress_ring": {
                "center": center.name,
                "progress": min(1.0, len(self.hrv_history) / 50),  # Progress based on time in chapter
                "color": center.color
            }
        }
    
    def _calculate_metric_adjustments(self, hrv_data: Dict[str, float], center: BookCenter) -> Dict[str, float]:
        """Calculate target metric adjustments for current center"""
        base_targets = {
            "empathy": 0.5,
            "creativity": 0.5,
            "resilience": 0.5,
            "focus": 0.5,
            "curiosity": 0.5,
            "compassion": 0.5
        }
        
        # Apply center-specific adjustments
        adjusted_targets = base_targets.copy()
        for metric, target_value in center.target_metrics.items():
            if metric in adjusted_targets:
                adjusted_targets[metric] = target_value
        
        # HRV-based modulation
        coherence = hrv_data.get('coherence', 0.5)
        if coherence > 0.7:  # High coherence boosts all targets
            for metric in adjusted_targets.keys():
                adjusted_targets[metric] = min(1.0, adjusted_targets[metric] * 1.1)
        
        return adjusted_targets
    
    def get_current_adaptation_state(self) -> Dict[str, any]:
        """Get current adaptation state for UI updates"""
        if not self.current_context:
            return {"active": False}
        
        return {
            "active": True,
            "context": {
                "book_id": self.current_context.book_id,
                "chapter_title": self.current_context.title,
                "active_center": self.current_context.active_center,
                "similarity_scores": self.current_context.similarity_scores,
                "timestamp": self.current_context.timestamp.isoformat()
            },
            "center_info": {
                "name": self.current_context.active_center,
                "description": self.book_centers[self.current_context.active_center].description,
                "color": self.book_centers[self.current_context.active_center].color,
                "position": self.book_centers[self.current_context.active_center].position
            }
        }

# Global instance
semantic_adaptation = SemanticBookAdaptation()