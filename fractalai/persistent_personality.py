"""
Persistent personality system with database storage and real-time synchronization
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class PersonalityState(Base):
    """Database model for persistent personality states"""
    __tablename__ = 'personality_states'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), nullable=False, index=True)
    user_id = Column(String(128), index=True)  # For cross-session persistence
    
    # Core personality traits
    empathy = Column(Float, default=0.5)
    curiosity = Column(Float, default=0.5)
    analyticalness = Column(Float, default=0.5)
    creativity = Column(Float, default=0.5)
    humor = Column(Float, default=0.4)
    supportiveness = Column(Float, default=0.5)
    assertiveness = Column(Float, default=0.5)
    
    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow)
    interaction_count = Column(Integer, default=0)
    emotional_context = Column(Text)  # JSON string
    dominant_topics = Column(Text)    # JSON string
    adaptation_history = Column(Text) # JSON string of trait changes
    
    # Relationship and context tracking
    relationship_depth = Column(Float, default=0.1)
    preferred_styles = Column(Text)   # JSON array of user preferences
    conflict_patterns = Column(Text)  # JSON record of trait conflicts

class EmotionalContext(Base):
    """Track emotional patterns and context per session"""
    __tablename__ = 'emotional_contexts'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user_emotion = Column(String(64))  # detected emotion
    bot_response_tone = Column(String(64))  # bot's emotional response
    empathy_level = Column(Float)
    support_provided = Column(Boolean, default=False)
    
    context_snapshot = Column(Text)  # JSON of conversation context

class PersistentPersonalityManager:
    """Manages persistent personality across sessions with real-time sync"""
    
    def __init__(self):
        try:
            # Use PostgreSQL for persistence
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                raise Exception("DATABASE_URL not found")
            
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
            
            # In-memory cache for performance
            self.personality_cache = {}
            self.sync_callbacks = []  # For UI synchronization
            
            logging.info("Persistent personality system initialized with PostgreSQL")
            
        except Exception as e:
            logging.error(f"Failed to initialize persistent personality: {e}")
            # Fallback to in-memory only
            self.engine = None
            self.db_session = None
            self.personality_cache = {}
            self.sync_callbacks = []
        
    def get_or_create_personality(self, session_id: str, user_id: str = None) -> Dict[str, float]:
        """Get personality state from database or create new one"""
        # Check cache first
        if session_id in self.personality_cache:
            return self.personality_cache[session_id]
        
        # Fallback if no database connection
        if not self.db_session:
            default_personality = {
                'empathy': 0.5, 'curiosity': 0.5, 'analyticalness': 0.5,
                'creativity': 0.5, 'humor': 0.4, 'supportiveness': 0.5,
                'assertiveness': 0.5
            }
            self.personality_cache[session_id] = default_personality
            return default_personality
        
        try:
            # Try to find existing personality
            personality_record = self.db_session.query(PersonalityState).filter(
                PersonalityState.session_id == session_id
            ).first()
            
            if not personality_record and user_id:
                # Try to find by user_id for cross-session continuity
                personality_record = self.db_session.query(PersonalityState).filter(
                    PersonalityState.user_id == user_id
                ).order_by(PersonalityState.last_updated.desc()).first()
                
                if personality_record:
                    # Create new session record based on user history
                    new_record = PersonalityState(
                        session_id=session_id,
                        user_id=user_id,
                        empathy=personality_record.empathy,
                        curiosity=personality_record.curiosity,
                        analyticalness=personality_record.analyticalness,
                        creativity=personality_record.creativity,
                        humor=personality_record.humor,
                        supportiveness=personality_record.supportiveness,
                        assertiveness=personality_record.assertiveness,
                        relationship_depth=min(1.0, personality_record.relationship_depth + 0.1)
                    )
                    self.db_session.add(new_record)
                    self.db_session.commit()
                    personality_record = new_record
            
            if not personality_record:
                # Create new personality with defaults
                personality_record = PersonalityState(
                    session_id=session_id,
                    user_id=user_id
                )
                self.db_session.add(personality_record)
                self.db_session.commit()
            
            # Convert to dictionary
            personality = {
                'empathy': personality_record.empathy,
                'curiosity': personality_record.curiosity,
                'analyticalness': personality_record.analyticalness,
                'creativity': personality_record.creativity,
                'humor': personality_record.humor,
                'supportiveness': personality_record.supportiveness,
                'assertiveness': personality_record.assertiveness
            }
            
            # Cache for performance
            self.personality_cache[session_id] = personality
            return personality
            
        except Exception as e:
            logging.error(f"Error getting personality state: {e}")
            # Fallback to defaults
            return {
                'empathy': 0.5, 'curiosity': 0.5, 'analyticalness': 0.5,
                'creativity': 0.5, 'humor': 0.4, 'supportiveness': 0.5,
                'assertiveness': 0.5
            }
    
    def update_personality(self, session_id: str, new_traits: Dict[str, float], 
                          adaptation_reason: str = "", user_input: str = "") -> Dict[str, float]:
        """Update personality traits with reason tracking and real-time sync"""
        # Get current personality for comparison
        current_personality = self.get_or_create_personality(session_id)
        
        # Update cache immediately
        updated_personality = current_personality.copy()
        updated_personality.update(new_traits)
        self.personality_cache[session_id] = updated_personality
        
        # If no database, return cached version
        if not self.db_session:
            logging.info(f"Updated personality (memory only) for {session_id}: {updated_personality}")
            return updated_personality
        
        try:
            personality_record = self.db_session.query(PersonalityState).filter(
                PersonalityState.session_id == session_id
            ).first()
            
            if not personality_record:
                return self.get_or_create_personality(session_id)
            
            # Track changes for transparency
            old_traits = {
                'empathy': personality_record.empathy,
                'curiosity': personality_record.curiosity,
                'analyticalness': personality_record.analyticalness,
                'creativity': personality_record.creativity,
                'humor': personality_record.humor,
                'supportiveness': personality_record.supportiveness,
                'assertiveness': personality_record.assertiveness
            }
            
            # Update traits
            for trait, value in new_traits.items():
                if hasattr(personality_record, trait):
                    setattr(personality_record, trait, max(0.0, min(1.0, value)))
            
            personality_record.last_updated = datetime.utcnow()
            personality_record.interaction_count += 1
            
            # Store adaptation history
            adaptation_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'reason': adaptation_reason,
                'user_input': user_input[:100],  # Truncate for storage
                'changes': {trait: {'from': old_traits[trait], 'to': new_traits.get(trait, old_traits[trait])} 
                           for trait in old_traits if abs(old_traits[trait] - new_traits.get(trait, old_traits[trait])) > 0.1}
            }
            
            history = json.loads(personality_record.adaptation_history or '[]')
            history.append(adaptation_entry)
            personality_record.adaptation_history = json.dumps(history[-20:])  # Keep last 20 changes
            
            self.db_session.commit()
            
            # Update cache
            updated_personality = {
                'empathy': personality_record.empathy,
                'curiosity': personality_record.curiosity,
                'analyticalness': personality_record.analyticalness,
                'creativity': personality_record.creativity,
                'humor': personality_record.humor,
                'supportiveness': personality_record.supportiveness,
                'assertiveness': personality_record.assertiveness
            }
            self.personality_cache[session_id] = updated_personality
            
            # Trigger UI sync callbacks
            self._trigger_sync_callbacks(session_id, updated_personality, adaptation_entry)
            
            return updated_personality
            
        except Exception as e:
            logging.error(f"Error updating personality: {e}")
            return self.personality_cache.get(session_id, {})
    
    def record_emotional_context(self, session_id: str, user_emotion: str, 
                                bot_tone: str, empathy_level: float, context: Dict[str, Any]):
        """Record emotional interaction for long-term pattern analysis"""
        try:
            emotional_record = EmotionalContext(
                session_id=session_id,
                user_emotion=user_emotion,
                bot_response_tone=bot_tone,
                empathy_level=empathy_level,
                context_snapshot=json.dumps(context)
            )
            
            self.db_session.add(emotional_record)
            self.db_session.commit()
            
        except Exception as e:
            logging.error(f"Error recording emotional context: {e}")
    
    def get_adaptation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the history of trait adaptations for transparency"""
        try:
            personality_record = self.db_session.query(PersonalityState).filter(
                PersonalityState.session_id == session_id
            ).first()
            
            if personality_record and personality_record.adaptation_history:
                return json.loads(personality_record.adaptation_history)
            return []
            
        except Exception as e:
            logging.error(f"Error getting adaptation history: {e}")
            return []
    
    def detect_trait_conflicts(self, traits: Dict[str, float]) -> List[str]:
        """Detect conflicting trait combinations and return explanations"""
        conflicts = []
        
        # High empathy + Low supportiveness conflict
        if traits.get('empathy', 0) > 0.8 and traits.get('supportiveness', 0) < 0.3:
            conflicts.append("High empathy with low supportiveness creates internal conflict - I understand your feelings but struggle to offer support")
        
        # High humor + High analytical (in serious contexts)
        if traits.get('humor', 0) > 0.8 and traits.get('analyticalness', 0) > 0.8:
            conflicts.append("Balancing high humor with analytical depth - I may alternate between witty observations and systematic analysis")
        
        # High assertiveness + High empathy
        if traits.get('assertiveness', 0) > 0.8 and traits.get('empathy', 0) > 0.8:
            conflicts.append("High assertiveness with high empathy - I'll be direct but compassionate, which can feel complex")
        
        return conflicts
    
    def _trigger_sync_callbacks(self, session_id: str, personality: Dict[str, float], change_info: Dict[str, Any]):
        """Trigger registered callbacks for UI synchronization"""
        for callback in self.sync_callbacks:
            try:
                callback(session_id, personality, change_info)
            except Exception as e:
                logging.error(f"Error in sync callback: {e}")
    
    def register_sync_callback(self, callback):
        """Register callback for real-time UI synchronization"""
        self.sync_callbacks.append(callback)
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db_session') and self.db_session is not None:
            try:
                self.db_session.close()
            except Exception:
                pass  # Ignore errors during cleanup