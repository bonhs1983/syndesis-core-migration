"""
Conversational Memory Evolution System
A novel approach to AI that creates evolving, persistent AI personalities
that remember and build upon every interaction, creating continuous relationships.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import hashlib

class MemoryNode:
    """Represents a memory fragment with emotional weight and associations"""
    
    def __init__(self, content: str, emotional_weight: float = 0.0, 
                 memory_type: str = "interaction", timestamp: datetime = None):
        self.content = content
        self.emotional_weight = emotional_weight  # -1.0 to 1.0
        self.memory_type = memory_type  # interaction, concept, pattern, meta
        self.timestamp = timestamp or datetime.now()
        self.associations = []  # Links to other memory nodes
        self.access_count = 0
        self.relevance_decay = 1.0
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this memory node"""
        content_hash = hashlib.md5(
            f"{self.content}{self.timestamp}".encode()
        ).hexdigest()[:8]
        return f"mem_{content_hash}"
    
    def access(self):
        """Record memory access - affects relevance"""
        self.access_count += 1
        self.relevance_decay = min(1.0, self.relevance_decay + 0.1)
    
    def decay(self, decay_rate: float = 0.05):
        """Natural memory decay over time"""
        self.relevance_decay = max(0.1, self.relevance_decay - decay_rate)

class PersonalityProfile:
    """Evolving personality profile that changes based on interactions"""
    
    def __init__(self):
        self.traits = {
            'curiosity': 0.5,
            'empathy': 0.5,
            'creativity': 0.5,
            'analytical': 0.5,
            'humor': 0.5,
            'directness': 0.5,
            'supportiveness': 0.5
        }
        self.interaction_count = 0
        self.evolution_history = []
    
    def evolve_from_interaction(self, user_input: str, ai_response: str, 
                              user_feedback: Optional[float] = None):
        """Evolve personality based on interaction patterns"""
        # Analyze interaction characteristics
        if '?' in user_input:
            self.traits['curiosity'] += 0.01
        
        if any(word in user_input.lower() for word in ['feel', 'emotion', 'sad', 'happy', 'anxious']):
            self.traits['empathy'] += 0.02
        
        if any(word in user_input.lower() for word in ['create', 'imagine', 'dream', 'idea']):
            self.traits['creativity'] += 0.02
        
        if any(word in user_input.lower() for word in ['analyze', 'think', 'logic', 'reason']):
            self.traits['analytical'] += 0.02
        
        # Normalize traits to stay within bounds
        for trait in self.traits:
            self.traits[trait] = max(0.0, min(1.0, self.traits[trait]))
        
        self.interaction_count += 1
        
        # Record evolution milestone
        if self.interaction_count % 10 == 0:
            self.evolution_history.append({
                'timestamp': datetime.now(),
                'interaction_count': self.interaction_count,
                'traits': self.traits.copy()
            })

class ConversationalMemoryEvolution:
    """Main system for evolving conversational AI with persistent memory"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.personality = PersonalityProfile()
        self.conversation_patterns = defaultdict(int)
        self.meta_learning_insights = []
        self.relationship_depth = 0.0
        
    def add_memory(self, content: str, memory_type: str = "interaction", 
                   emotional_weight: float = 0.0) -> MemoryNode:
        """Add new memory node and create associations"""
        memory = MemoryNode(content, emotional_weight, memory_type)
        self.memory_nodes[memory.id] = memory
        
        # Create associations with existing memories
        self._create_associations(memory)
        
        return memory
    
    def _create_associations(self, new_memory: MemoryNode):
        """Create associations between memories based on content similarity"""
        for existing_id, existing_memory in self.memory_nodes.items():
            if existing_id == new_memory.id:
                continue
                
            # Simple association based on common words
            new_words = set(new_memory.content.lower().split())
            existing_words = set(existing_memory.content.lower().split())
            
            overlap = len(new_words.intersection(existing_words))
            if overlap >= 2:  # Threshold for association
                new_memory.associations.append(existing_id)
                existing_memory.associations.append(new_memory.id)
    
    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[MemoryNode]:
        """Retrieve memories relevant to current query"""
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.memory_nodes.values():
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            # Score based on relevance, recency, and access patterns
            recency_score = 1.0 / (1 + (datetime.now() - memory.timestamp).days)
            relevance_score = overlap * memory.relevance_decay
            access_score = min(1.0, memory.access_count / 10)
            
            total_score = relevance_score + 0.3 * recency_score + 0.2 * access_score
            
            if total_score > 0:
                scored_memories.append((total_score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        relevant_memories = [mem for _, mem in scored_memories[:limit]]
        
        # Mark memories as accessed
        for memory in relevant_memories:
            memory.access()
        
        return relevant_memories
    
    def generate_context_aware_response(self, user_input: str, base_response: str) -> str:
        """Generate response enhanced with memory and personality"""
        # Retrieve relevant memories
        relevant_memories = self.retrieve_relevant_memories(user_input)
        
        # Build context from memories
        memory_context = ""
        if relevant_memories and len(relevant_memories) > 0:
            memory_context = "\n\n💭 **Memory Context**: " + "; ".join([
                mem.content[:80] + "..." if len(mem.content) > 80 else mem.content
                for mem in relevant_memories[:2]
            ])
        
        # Enhance response based on personality evolution
        enhanced_response = self.get_personality_enhanced_response(base_response)
        
        # Add memory context
        if memory_context:
            enhanced_response += memory_context
        
        # Add personality modifier
        personality_modifier = self._get_personality_modifier()
        if personality_modifier:
            enhanced_response += f"\n\n{personality_modifier}"
        
        # Record this interaction
        self.add_memory(f"User: {user_input} | AI: {base_response}")
        self.personality.evolve_from_interaction(user_input, base_response)
        
        # Update relationship depth
        self.relationship_depth += 0.01
        
        return enhanced_response
    
    def get_personality_snapshot(self):
        """Get current personality trait values"""
        return {
            'empathy': self.personality.traits.get('empathy', 0.5),
            'curiosity': self.personality.traits.get('curiosity', 0.5),
            'humor': self.personality.traits.get('humor', 0.5),
            'formality': self.personality.traits.get('formality', 0.5),
            'creativity': self.personality.traits.get('creativity', 0.5),
            'supportiveness': self.personality.traits.get('supportiveness', 0.5),
            'analyticalness': self.personality.traits.get('analytical', 0.5)
        }
    
    @property
    def memories(self):
        """Property to access memory nodes for compatibility"""
        return list(self.memory_nodes.values())
    
    def _get_personality_modifier(self) -> str:
        """Generate personality-based response modifier"""
        traits = self.personality.traits
        modifiers = []
        
        # High empathy responses
        if traits.get('empathy', 0.5) > 0.7:
            modifiers.append("I sense this topic is meaningful to you.")
        
        # High curiosity responses  
        if traits.get('curiosity', 0.5) > 0.7:
            modifiers.append("This makes me wonder about the deeper connections here.")
        
        # High creativity responses
        if traits.get('creativity', 0.5) > 0.7:
            modifiers.append("I'm imagining several creative perspectives on this.")
        
        # High analytical responses
        if traits.get('analytical', 0.5) > 0.7:
            modifiers.append("Let me break down the logical components of this idea.")
        
        return " ".join(modifiers) if modifiers else ""
    
    def get_personality_enhanced_response(self, base_response: str) -> str:
        """Enhance response based on current personality traits"""
        traits = self.personality.traits
        
        # Modify response style based on personality
        if traits.get('empathy', 0.5) > 0.6:
            if 'happiness' in base_response.lower():
                base_response += " I believe happiness connects us to what truly matters in our relationships and personal growth."
        
        if traits.get('creativity', 0.5) > 0.6:
            if any(word in base_response.lower() for word in ['art', 'music', 'creative', 'imagine']):
                base_response += " Creativity flows from the intersection of experience and imagination."
        
        if traits.get('curiosity', 0.5) > 0.7:
            if '?' in base_response:
                base_response += " This opens up so many fascinating questions to explore together."
        
        return base_response
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of the evolving relationship"""
        return {
            'session_id': self.session_id,
            'total_memories': len(self.memory_nodes),
            'relationship_depth': self.relationship_depth,
            'interactions': self.personality.interaction_count,
            'dominant_personality_trait': max(self.personality.traits, key=self.personality.traits.get),
            'personality_evolution': len(self.personality.evolution_history),
            'memory_types': dict(defaultdict(int, {
                mem.memory_type: 1 for mem in self.memory_nodes.values()
            }))
        }
    
    def perform_meta_learning(self):
        """Analyze patterns in conversations to improve learning"""
        if len(self.memory_nodes) < 10:
            return
        
        # Analyze conversation patterns
        patterns = defaultdict(int)
        for memory in self.memory_nodes.values():
            if "User:" in memory.content:
                # Extract user input patterns
                user_part = memory.content.split("User:")[1].split("|")[0].strip()
                patterns[len(user_part.split())] += 1  # Word count patterns
        
        # Generate meta-insights
        if patterns:
            avg_user_input_length = sum(k * v for k, v in patterns.items()) / sum(patterns.values())
            insight = f"User tends to write {avg_user_input_length:.1f} words on average."
            self.meta_learning_insights.append({
                'timestamp': datetime.now(),
                'insight': insight,
                'pattern_data': dict(patterns)
            })
    
    def decay_memories(self):
        """Apply natural decay to memories"""
        for memory in self.memory_nodes.values():
            memory.decay()
    
    def export_evolution_data(self) -> Dict[str, Any]:
        """Export complete evolution data for analysis"""
        return {
            'session_id': self.session_id,
            'personality_profile': {
                'current_traits': self.personality.traits,
                'evolution_history': self.personality.evolution_history,
                'interaction_count': self.personality.interaction_count
            },
            'memory_summary': {
                'total_nodes': len(self.memory_nodes),
                'memory_types': dict(defaultdict(int, {
                    mem.memory_type: 1 for mem in self.memory_nodes.values()
                })),
                'average_emotional_weight': sum(
                    mem.emotional_weight for mem in self.memory_nodes.values()
                ) / len(self.memory_nodes) if self.memory_nodes else 0
            },
            'relationship_metrics': {
                'depth': self.relationship_depth,
                'conversation_patterns': dict(self.conversation_patterns)
            },
            'meta_learning': self.meta_learning_insights
        }
    
    def get_relationship_depth(self):
        """Calculate relationship depth based on interactions and emotional resonance"""
        if not self.memory_nodes:
            return 0.1  # Baseline relationship
        
        # Factors that increase relationship depth
        interaction_count = len(self.memory_nodes)
        
        # Calculate depth score (0.0 to 1.0)
        base_depth = min(interaction_count / 50, 0.4)
        return min(base_depth + 0.1, 1.0)  # Start at 0.1, grow with interactions