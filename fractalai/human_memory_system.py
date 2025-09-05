"""
Human-like Memory System for Syndesis AI
Implements encoding, consolidation, storage, retrieval, reconsolidation, forgetting and creative distortion
Based on computational neuroscience principles
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

@dataclass
class MemoryTrace:
    """Represents a single memory trace with context tags"""
    content: str
    encoding_time: datetime
    context_tags: Dict[str, Any]  # time, place, affect, intensity
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    consolidation_strength: float = 0.1
    semantic_weight: float = 0.5
    episodic_weight: float = 0.5
    decay_rate: float = 0.02

@dataclass
class ConsolidationEvent:
    """Tracks memory consolidation events"""
    memory_id: str
    consolidation_time: datetime
    strength_before: float
    strength_after: float
    trigger_type: str  # sleep, rehearsal, emotional_salience

class HumanLikeMemorySystem:
    """
    Computational system that emulates human-like memory processes:
    - Encoding: multimodal input → episodic & semantic traces
    - Consolidation: strengthen important memories over time
    - Storage: maintain context tags for time, place, affect
    - Retrieval: optimise for accuracy vs. resource use
    - Reconsolidation: memories become labile when retrieved
    - Forgetting: penalise uncontrolled catastrophic forgetting
    - Creative Distortion: introduce controlled noise for creativity
    """
    
    def __init__(self, max_memory_capacity: int = 10000):
        self.memories: Dict[str, MemoryTrace] = {}
        self.consolidation_history: List[ConsolidationEvent] = []
        self.max_capacity = max_memory_capacity
        self.forgetting_threshold = 0.05
        self.creative_distortion_rate = 0.1
        
    def encode_multimodal_input(self, 
                               content: str, 
                               context: Dict[str, Any],
                               input_type: str = "text") -> str:
        """
        Encode multimodal input into episodic & semantic traces
        
        Args:
            content: The input content to encode
            context: Context tags (time, place, affect, etc.)
            input_type: Type of input (text, image, audio, etc.)
            
        Returns:
            memory_id: Unique identifier for the encoded memory
        """
        memory_id = f"mem_{int(time.time() * 1000000)}"
        
        # Enhanced context tags
        enhanced_context = {
            "input_type": input_type,
            "encoding_timestamp": datetime.now().isoformat(),
            "emotional_valence": context.get("affect", 0.0),
            "spatial_context": context.get("place", "unknown"),
            "temporal_context": context.get("time", "present"),
            "attention_level": context.get("attention", 0.7),
            "novelty_score": self._calculate_novelty(content),
            **context
        }
        
        # Create memory trace with dual encoding (episodic + semantic)
        memory = MemoryTrace(
            content=content,
            encoding_time=datetime.now(),
            context_tags=enhanced_context,
            semantic_weight=self._calculate_semantic_weight(content),
            episodic_weight=self._calculate_episodic_weight(enhanced_context)
        )
        
        self.memories[memory_id] = memory
        
        # Trigger immediate consolidation for high-salience memories
        if enhanced_context.get("emotional_valence", 0) > 0.7:
            self._consolidate_memory(memory_id, "emotional_salience")
            
        # Check capacity and trigger forgetting if needed
        if len(self.memories) > self.max_capacity:
            self._controlled_forgetting()
            
        logging.info(f"Encoded memory {memory_id} with novelty {enhanced_context['novelty_score']:.2f}")
        return memory_id
    
    def retrieve_with_context(self, 
                            query: str, 
                            context_filter: Optional[Dict] = None,
                            max_results: int = 5) -> List[Tuple[str, MemoryTrace, float]]:
        """
        Retrieve memories optimized for accuracy vs. resource use
        
        Args:
            query: Search query
            context_filter: Filter by context tags
            max_results: Maximum number of results to return
            
        Returns:
            List of (memory_id, memory_trace, relevance_score) tuples
        """
        candidates = []
        
        for memory_id, memory in self.memories.items():
            # Calculate relevance score
            relevance = self._calculate_relevance(memory, query, context_filter)
            
            if relevance > 0.1:  # Threshold for relevance
                candidates.append((memory_id, memory, relevance))
                
                # Update access statistics (triggers reconsolidation)
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                
                # Reconsolidation: memories become labile when retrieved
                self._reconsolidate_memory(memory_id)
        
        # Sort by relevance and return top results
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_results]
    
    def consolidate_memories(self, trigger_type: str = "rehearsal"):
        """
        Strengthen important memories through consolidation
        
        Args:
            trigger_type: What triggered consolidation (sleep, rehearsal, etc.)
        """
        consolidated_count = 0
        
        for memory_id, memory in self.memories.items():
            # Consolidation criteria
            should_consolidate = (
                memory.access_count > 2 or
                memory.context_tags.get("emotional_valence", 0) > 0.5 or
                datetime.now() - memory.encoding_time > timedelta(hours=1)
            )
            
            if should_consolidate and memory.consolidation_strength < 0.9:
                self._consolidate_memory(memory_id, trigger_type)
                consolidated_count += 1
        
        logging.info(f"Consolidated {consolidated_count} memories via {trigger_type}")
    
    def apply_creative_distortion(self, memory_id: str, distortion_level: float = None) -> str:
        """
        Apply controlled creative distortion to a memory
        
        Args:
            memory_id: ID of memory to distort
            distortion_level: Level of distortion (0.0-1.0)
            
        Returns:
            Distorted memory content
        """
        if memory_id not in self.memories:
            return ""
            
        memory = self.memories[memory_id]
        distortion_level = distortion_level or self.creative_distortion_rate
        
        # Creative distortion algorithm
        distorted_content = self._distort_content(memory.content, distortion_level)
        
        # Create new memory trace for distorted version
        distorted_id = f"{memory_id}_distorted_{int(time.time())}"
        distorted_memory = MemoryTrace(
            content=distorted_content,
            encoding_time=datetime.now(),
            context_tags={**memory.context_tags, "distorted_from": memory_id},
            semantic_weight=memory.semantic_weight * 0.8,  # Reduce semantic weight
            episodic_weight=memory.episodic_weight * 1.2   # Increase episodic weight
        )
        
        self.memories[distorted_id] = distorted_memory
        
        logging.info(f"Applied creative distortion to {memory_id} → {distorted_id}")
        return distorted_content
    
    def _calculate_novelty(self, content: str) -> float:
        """Calculate novelty score based on existing memories"""
        if not self.memories:
            return 1.0
            
        # Simple novelty calculation (could be enhanced with embeddings)
        max_similarity = 0.0
        for memory in self.memories.values():
            similarity = len(set(content.lower().split()) & set(memory.content.lower().split()))
            similarity /= max(len(content.split()), len(memory.content.split()))
            max_similarity = max(max_similarity, similarity)
            
        return 1.0 - max_similarity
    
    def _calculate_semantic_weight(self, content: str) -> float:
        """Calculate semantic importance of content"""
        # Enhanced semantic analysis (could integrate with NLP models)
        semantic_indicators = [
            "concept", "principle", "theory", "definition", "meaning",
            "important", "significant", "key", "fundamental", "essential"
        ]
        
        weight = 0.3  # Base weight
        for indicator in semantic_indicators:
            if indicator in content.lower():
                weight += 0.1
                
        return min(weight, 1.0)
    
    def _calculate_episodic_weight(self, context: Dict) -> float:
        """Calculate episodic importance based on context"""
        weight = 0.3  # Base weight
        
        # High emotional valence increases episodic weight
        weight += context.get("emotional_valence", 0) * 0.3
        
        # Specific spatial/temporal context increases episodic weight
        if context.get("spatial_context") != "unknown":
            weight += 0.2
        if context.get("temporal_context") != "present":
            weight += 0.2
            
        return min(weight, 1.0)
    
    def _calculate_relevance(self, memory: MemoryTrace, query: str, context_filter: Optional[Dict]) -> float:
        """Calculate relevance score for memory retrieval"""
        # Content relevance (simple keyword matching - could be enhanced)
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        content_overlap = len(query_words & memory_words) / len(query_words) if query_words else 0
        
        # Context relevance
        context_relevance = 1.0
        if context_filter:
            for key, value in context_filter.items():
                if key in memory.context_tags:
                    if memory.context_tags[key] != value:
                        context_relevance *= 0.5
        
        # Recency bonus
        recency = 1.0 / (1 + (datetime.now() - memory.encoding_time).days)
        
        # Access frequency bonus
        access_bonus = min(memory.access_count * 0.1, 0.5)
        
        # Consolidation strength bonus
        consolidation_bonus = memory.consolidation_strength * 0.3
        
        total_relevance = (
            content_overlap * 0.4 +
            context_relevance * 0.3 +
            recency * 0.1 +
            access_bonus * 0.1 +
            consolidation_bonus * 0.1
        )
        
        return total_relevance
    
    def _consolidate_memory(self, memory_id: str, trigger_type: str):
        """Consolidate a specific memory"""
        if memory_id not in self.memories:
            return
            
        memory = self.memories[memory_id]
        old_strength = memory.consolidation_strength
        
        # Consolidation algorithm
        strength_increase = 0.1
        if trigger_type == "emotional_salience":
            strength_increase = 0.2
        elif trigger_type == "sleep":
            strength_increase = 0.15
            
        memory.consolidation_strength = min(1.0, memory.consolidation_strength + strength_increase)
        
        # Record consolidation event
        event = ConsolidationEvent(
            memory_id=memory_id,
            consolidation_time=datetime.now(),
            strength_before=old_strength,
            strength_after=memory.consolidation_strength,
            trigger_type=trigger_type
        )
        self.consolidation_history.append(event)
    
    def _reconsolidate_memory(self, memory_id: str):
        """Reconsolidate memory after retrieval (makes it labile)"""
        if memory_id not in self.memories:
            return
            
        memory = self.memories[memory_id]
        
        # Reconsolidation: slight reduction in strength, then strengthening
        memory.consolidation_strength *= 0.95  # Temporary lability
        
        # Schedule reconsolidation strengthening (simplified)
        if memory.consolidation_strength < 0.9:
            memory.consolidation_strength += 0.05
    
    def _controlled_forgetting(self):
        """Implement controlled forgetting to prevent catastrophic forgetting"""
        # Calculate forgetting scores
        forgetting_candidates = []
        
        for memory_id, memory in self.memories.items():
            # Forgetting score based on multiple factors
            time_decay = (datetime.now() - memory.encoding_time).days * memory.decay_rate
            access_protection = memory.access_count * 0.1
            consolidation_protection = memory.consolidation_strength
            
            forgetting_score = time_decay - access_protection - consolidation_protection
            
            if forgetting_score > self.forgetting_threshold:
                forgetting_candidates.append((memory_id, forgetting_score))
        
        # Sort by forgetting score and remove least important memories
        forgetting_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Remove 10% of capacity worth of memories
        to_remove = min(len(forgetting_candidates), self.max_capacity // 10)
        
        for i in range(to_remove):
            memory_id, _ = forgetting_candidates[i]
            del self.memories[memory_id]
            
        logging.info(f"Controlled forgetting: removed {to_remove} memories")
    
    def _distort_content(self, content: str, distortion_level: float) -> str:
        """Apply creative distortion to content"""
        words = content.split()
        distorted_words = []
        
        for word in words:
            if random.random() < distortion_level:
                # Creative distortion strategies
                if len(word) > 3:
                    # Character substitution
                    chars = list(word)
                    random_pos = random.randint(1, len(chars) - 2)
                    chars[random_pos] = random.choice("aeiou")
                    distorted_words.append("".join(chars))
                else:
                    # Synonym substitution (simplified)
                    synonyms = {
                        "good": "great", "bad": "poor", "big": "large", "small": "tiny",
                        "happy": "joyful", "sad": "melancholy", "fast": "quick", "slow": "gradual"
                    }
                    distorted_words.append(synonyms.get(word.lower(), word))
            else:
                distorted_words.append(word)
        
        return " ".join(distorted_words)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        if not self.memories:
            return {"total_memories": 0}
            
        consolidation_strengths = [m.consolidation_strength for m in self.memories.values()]
        access_counts = [m.access_count for m in self.memories.values()]
        
        return {
            "total_memories": len(self.memories),
            "avg_consolidation_strength": sum(consolidation_strengths) / len(consolidation_strengths),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "consolidation_events": len(self.consolidation_history),
            "memory_types": {
                "high_semantic": len([m for m in self.memories.values() if m.semantic_weight > 0.7]),
                "high_episodic": len([m for m in self.memories.values() if m.episodic_weight > 0.7]),
                "well_consolidated": len([m for m in self.memories.values() if m.consolidation_strength > 0.7])
            }
        }