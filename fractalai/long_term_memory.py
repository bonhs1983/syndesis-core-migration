"""
Long-term memory system for persistent conversation context and emotional state tracking
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class LongTermMemory:
    """Manages persistent conversation memory and emotional context"""
    
    def __init__(self):
        self.conversation_threads = {}
        self.emotional_patterns = {}
        self.user_preferences = {}
        self.relationship_depth = {}
    
    def store_interaction(self, session_id: str, user_input: str, bot_response: str, 
                         personality: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Store interaction with full context for long-term recall"""
        try:
            from models import InteractionLog
            from app import db
            
            # Create comprehensive interaction record
            interaction_data = {
                'user_input': user_input,
                'bot_response': bot_response,
                'personality': personality,
                'timestamp': datetime.now().isoformat(),
                'context': context or {},
                'emotional_indicators': self._extract_emotional_indicators(user_input),
                'topics': self._extract_topics(user_input)
            }
            
            # Store in database (using existing model structure)
            record = InteractionLog()
            record.agent_input = user_input
            record.agent_output = bot_response
            record.context = json.dumps(interaction_data)
            record.session_id = session_id
            record.timestamp = datetime.now()
            
            db.session.add(record)
            db.session.commit()
            
            # Update in-memory caches
            if session_id not in self.conversation_threads:
                self.conversation_threads[session_id] = []
            
            self.conversation_threads[session_id].append(interaction_data)
            self._update_emotional_patterns(session_id, user_input, personality)
            self._update_relationship_depth(session_id)
            
        except Exception as e:
            logging.warning(f"Could not store long-term memory: {e}")
    
    def recall_conversation_history(self, session_id: str, 
                                  lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """Recall full conversation history within timeframe"""
        try:
            from models import InteractionLog
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            records = InteractionLog.query.filter(
                InteractionLog.session_id == session_id,
                InteractionLog.timestamp >= cutoff_time
            ).order_by(InteractionLog.timestamp.desc()).limit(50).all()
            
            history = []
            for record in reversed(records):  # Reverse to get chronological order
                try:
                    context_data = json.loads(record.context)
                    history.append(context_data)
                except json.JSONDecodeError:
                    # Fallback for simple context
                    history.append({
                        'user_input': record.agent_input,
                        'bot_response': record.agent_output,
                        'timestamp': record.timestamp.isoformat()
                    })
            
            return history
            
        except Exception as e:
            logging.warning(f"Could not recall conversation history: {e}")
            return []
    
    def get_emotional_context(self, session_id: str) -> Dict[str, Any]:
        """Get emotional patterns and context for this session"""
        if session_id not in self.emotional_patterns:
            return {'dominant_emotions': [], 'emotional_trend': 'neutral', 'support_needed': False}
        
        return self.emotional_patterns[session_id]
    
    def get_relevant_context(self, session_id: str, current_input: str) -> Dict[str, Any]:
        """Get contextually relevant information from conversation history"""
        history = self.recall_conversation_history(session_id)
        if not history:
            return {}
        
        # Find related topics and emotional context
        current_topics = self._extract_topics(current_input.lower())
        current_emotions = self._extract_emotional_indicators(current_input.lower())
        
        relevant_interactions = []
        for interaction in history[-10:]:  # Look at recent interactions
            interaction_topics = interaction.get('topics', [])
            interaction_emotions = interaction.get('emotional_indicators', [])
            
            # Check for topic overlap
            if any(topic in interaction_topics for topic in current_topics):
                relevant_interactions.append(interaction)
            # Check for emotional continuity
            elif any(emotion in interaction_emotions for emotion in current_emotions):
                relevant_interactions.append(interaction)
        
        return {
            'relevant_history': relevant_interactions[-3:],  # Most recent relevant interactions
            'conversation_length': len(history),
            'relationship_depth': self.relationship_depth.get(session_id, 0.1),
            'emotional_context': self.get_emotional_context(session_id)
        }
    
    def _extract_emotional_indicators(self, text: str) -> List[str]:
        """Extract emotional indicators from text"""
        emotion_keywords = {
            'sad': ['sad', 'depressed', 'down', 'crying', 'upset', 'hurt'],
            'happy': ['happy', 'excited', 'joy', 'glad', 'thrilled', 'wonderful'],
            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'afraid'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'confused': ['confused', 'lost', 'uncertain', 'unsure', 'puzzled'],
            'grateful': ['thank', 'grateful', 'appreciate', 'thankful']
        }
        
        indicators = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                indicators.append(emotion)
        
        return indicators
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        topic_keywords = {
            'relationships': ['relationship', 'friend', 'family', 'love', 'partner'],
            'work': ['work', 'job', 'career', 'boss', 'colleague', 'office'],
            'health': ['health', 'sick', 'doctor', 'medicine', 'pain', 'therapy'],
            'technology': ['ai', 'computer', 'software', 'technology', 'digital'],
            'philosophy': ['meaning', 'purpose', 'life', 'existence', 'consciousness'],
            'creativity': ['creative', 'art', 'music', 'writing', 'design', 'imagination']
        }
        
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _update_emotional_patterns(self, session_id: str, user_input: str, personality: Dict[str, float]):
        """Update emotional patterns for this session"""
        emotions = self._extract_emotional_indicators(user_input.lower())
        
        if session_id not in self.emotional_patterns:
            self.emotional_patterns[session_id] = {
                'dominant_emotions': [],
                'emotional_trend': 'neutral',
                'support_needed': False,
                'empathy_requests': 0
            }
        
        patterns = self.emotional_patterns[session_id]
        patterns['dominant_emotions'].extend(emotions)
        
        # Keep only recent emotions (last 10)
        patterns['dominant_emotions'] = patterns['dominant_emotions'][-10:]
        
        # Determine if support is needed
        support_emotions = ['sad', 'anxious', 'angry', 'confused']
        patterns['support_needed'] = any(emotion in patterns['dominant_emotions'] for emotion in support_emotions)
        
        # Track empathy requests
        if personality.get('empathy', 0) > 0.7:
            patterns['empathy_requests'] += 1
    
    def _update_relationship_depth(self, session_id: str):
        """Update relationship depth based on interaction count and emotional sharing"""
        if session_id not in self.relationship_depth:
            self.relationship_depth[session_id] = 0.1
        
        # Increase depth with each interaction
        self.relationship_depth[session_id] = min(1.0, self.relationship_depth[session_id] + 0.05)
        
        # Bonus for emotional sharing
        if session_id in self.emotional_patterns:
            emotions = self.emotional_patterns[session_id]['dominant_emotions']
            if len(emotions) > 3:  # User has shared emotions
                self.relationship_depth[session_id] = min(1.0, self.relationship_depth[session_id] + 0.1)