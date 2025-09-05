"""
Active chat memory system for conversation recall and meta-reasoning
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

class ActiveChatMemory:
    """Manages active conversation memory with recall and reasoning capabilities"""
    
    def __init__(self):
        self.conversation_history = {}  # session_id -> messages
        self.trait_change_log = {}      # session_id -> trait changes
        self.meta_reasoning_cache = {}  # session_id -> reasoning data
        self.mistake_log = {}           # session_id -> mistakes and learnings
        
    def store_message(self, session_id: str, user_message: str, bot_response: str, 
                     personality_before: Dict[str, float], personality_after: Dict[str, float],
                     reasoning: str = ""):
        """Store conversation message with full context"""
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
            self.trait_change_log[session_id] = []
            
        # Store message with metadata
        message_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'personality_before': personality_before,
            'personality_after': personality_after,
            'reasoning': reasoning,
            'significant_changes': self._detect_significant_changes(personality_before, personality_after)
        }
        
        self.conversation_history[session_id].append(message_entry)
        
        # Log trait changes if significant
        if message_entry['significant_changes']:
            change_entry = {
                'timestamp': datetime.now().isoformat(),
                'trigger': user_message[:100],
                'changes': message_entry['significant_changes'],
                'reasoning': reasoning
            }
            self.trait_change_log[session_id].append(change_entry)
        
        # Keep only last 20 messages for performance
        if len(self.conversation_history[session_id]) > 20:
            self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
            
        if len(self.trait_change_log[session_id]) > 10:
            self.trait_change_log[session_id] = self.trait_change_log[session_id][-10:]
    
    def recall_conversation(self, session_id: str, lookback_count: int = 5) -> List[Dict[str, Any]]:
        """Recall recent conversation messages with context"""
        if session_id not in self.conversation_history:
            return []
        
        recent_messages = self.conversation_history[session_id][-lookback_count:]
        return recent_messages
    
    def find_trait_changes(self, session_id: str, user_query: str) -> Dict[str, Any]:
        """Find specific trait changes that match user query"""
        if session_id not in self.trait_change_log:
            return {'found': False, 'message': 'No trait changes recorded for this session'}
        
        changes = self.trait_change_log[session_id]
        if not changes:
            return {'found': False, 'message': 'No significant trait changes detected in our conversation'}
        
        # Find the most recent significant change
        most_recent_change = changes[-1]
        
        # Analyze what triggered it
        trigger_analysis = self._analyze_trigger(user_query, most_recent_change)
        
        return {
            'found': True,
            'change_details': most_recent_change,
            'trigger_analysis': trigger_analysis,
            'explanation': self._generate_change_explanation(most_recent_change)
        }
    
    def generate_meta_reflection(self, session_id: str, reflection_type: str) -> str:
        """Generate meta-cognitive reflection on conversation patterns"""
        if session_id not in self.conversation_history:
            return "I don't have conversation history to reflect on yet."
        
        history = self.conversation_history[session_id]
        changes = self.trait_change_log.get(session_id, [])
        
        if reflection_type == "trait_evolution":
            return self._reflect_on_trait_evolution(history, changes)
        elif reflection_type == "conversation_pattern":
            return self._reflect_on_conversation_patterns(history)
        elif reflection_type == "mistake_analysis":
            return self._reflect_on_mistakes(session_id, history)
        else:
            return self._general_reflection(history, changes)
    
    def log_mistake(self, session_id: str, mistake_description: str, learning: str):
        """Log a mistake and what was learned from it"""
        if session_id not in self.mistake_log:
            self.mistake_log[session_id] = []
        
        mistake_entry = {
            'timestamp': datetime.now().isoformat(),
            'mistake': mistake_description,
            'learning': learning,
            'context': self.conversation_history.get(session_id, [])[-1] if session_id in self.conversation_history else None
        }
        
        self.mistake_log[session_id].append(mistake_entry)
        logging.info(f"Mistake logged for {session_id}: {mistake_description}")
    
    def generate_apology(self, session_id: str) -> str:
        """Generate contextual apology based on conversation analysis"""
        history = self.conversation_history.get(session_id, [])
        mistakes = self.mistake_log.get(session_id, [])
        
        if not history:
            return "I apologize, but I don't have enough conversation context to provide a specific apology. However, I'm always learning and trying to improve my responses."
        
        # Analyze recent interactions for potential issues
        recent_messages = history[-3:]
        potential_issues = []
        
        for msg in recent_messages:
            if msg.get('bot_response') == "Hello. I'm ready to help you work through whatever questions or challenges you have. What would you like to analyze or discuss?":
                potential_issues.append("giving generic responses instead of personalized ones")
            
            if not msg.get('significant_changes') and any(word in msg['user_message'].lower() for word in ['empathy', 'creative', 'funny', 'analytical']):
                potential_issues.append("not responding to your explicit trait requests")
        
        if mistakes:
            last_mistake = mistakes[-1]
            return f"I apologize for {last_mistake['mistake']}. I've learned that {last_mistake['learning']} and will apply this understanding in our future conversations."
        
        if potential_issues:
            issue = potential_issues[0]
            learning = f"I need to be more attentive to your specific requests and provide personalized responses that truly reflect the personality traits you're asking for"
            
            # Log this as a learning opportunity
            self.log_mistake(session_id, issue, learning)
            
            return f"I apologize for {issue}. I've realized that {learning}. I'll be more mindful of this going forward."
        
        return "Looking at our conversation, I notice I could have been more responsive to your specific requests. I apologize for any moments where my responses felt generic rather than truly adaptive. I'm learning to be more attentive to your needs and will strive to provide more personalized, trait-driven responses."
    
    def _detect_significant_changes(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Detect significant trait changes"""
        changes = {}
        for trait, after_value in after.items():
            before_value = before.get(trait, 0.5)
            if abs(after_value - before_value) > 0.15:
                changes[trait] = {
                    'from': before_value,
                    'to': after_value,
                    'change': after_value - before_value
                }
        return changes
    
    def _analyze_trigger(self, user_query: str, change_entry: Dict[str, Any]) -> str:
        """Analyze what triggered a trait change"""
        trigger_text = change_entry['trigger'].lower()
        changes = change_entry['changes']
        
        # Identify trigger patterns
        if 'empathy' in trigger_text or 'understanding' in trigger_text:
            return f"Your request for empathy/understanding triggered {list(changes.keys())} changes"
        elif 'analytical' in trigger_text or 'logical' in trigger_text:
            return f"Your request for analytical thinking activated {list(changes.keys())} traits"
        elif 'creative' in trigger_text or 'innovative' in trigger_text:
            return f"Your creative request boosted {list(changes.keys())} traits"
        elif any(emotion in trigger_text for emotion in ['sad', 'worried', 'anxious', 'upset']):
            return f"Your emotional context triggered empathy adjustment: {list(changes.keys())}"
        else:
            return f"Context-based adjustment triggered changes in: {list(changes.keys())}"
    
    def _generate_change_explanation(self, change_entry: Dict[str, Any]) -> str:
        """Generate human-readable explanation of trait changes"""
        changes = change_entry['changes']
        explanations = []
        
        for trait, change_data in changes.items():
            direction = "increased" if change_data['change'] > 0 else "decreased"
            explanations.append(f"{trait} {direction} from {change_data['from']:.1f} to {change_data['to']:.1f}")
        
        return f"Specifically: {', '.join(explanations)}. Reasoning: {change_entry.get('reasoning', 'Context-based adaptation')}"
    
    def _reflect_on_trait_evolution(self, history: List[Dict], changes: List[Dict]) -> str:
        """Reflect on how traits have evolved during conversation"""
        if not changes:
            return "My personality traits have remained relatively stable throughout our conversation, with only minor contextual adjustments."
        
        evolution_summary = []
        for change in changes[-3:]:  # Last 3 changes
            change_desc = ", ".join([f"{trait}: {data['from']:.1f}→{data['to']:.1f}" 
                                   for trait, data in change['changes'].items()])
            evolution_summary.append(f"- {change_desc} (triggered by: {change['trigger'][:50]}...)")
        
        return f"My trait evolution in our conversation:\n" + "\n".join(evolution_summary) + f"\n\nThis shows how I adapt to your communication style and needs in real-time."
    
    def _reflect_on_conversation_patterns(self, history: List[Dict]) -> str:
        """Reflect on conversation patterns and dynamics"""
        if len(history) < 2:
            return "Our conversation is just beginning, so I don't have enough pattern data to analyze yet."
        
        # Analyze user engagement patterns
        user_questions = sum(1 for msg in history if '?' in msg['user_message'])
        emotional_messages = sum(1 for msg in history if any(word in msg['user_message'].lower() 
                                for word in ['feel', 'emotion', 'sad', 'happy', 'worried']))
        
        return f"In our {len(history)} message exchange, I've noticed you've asked {user_questions} questions and shared {emotional_messages} emotionally-contextual messages. This suggests you're testing both my analytical and empathetic capabilities, which is why my traits have been adapting accordingly."
    
    def _reflect_on_mistakes(self, session_id: str, history: List[Dict]) -> str:
        """Reflect on mistakes and learnings"""
        mistakes = self.mistake_log.get(session_id, [])
        
        if not mistakes:
            # Auto-detect potential mistakes from conversation
            potential_mistakes = []
            for msg in history:
                if "Hello. I'm ready to help" in msg.get('bot_response', ''):
                    potential_mistakes.append("Giving generic responses when you asked for specific personality traits")
            
            if potential_mistakes:
                return f"Upon reflection, I realize I may have made some mistakes: {potential_mistakes[0]}. I should have been more responsive to your explicit requests for personality changes."
            else:
                return "I haven't logged any specific mistakes in our conversation, but I'm always open to feedback about how I can improve."
        
        return f"I've identified {len(mistakes)} areas for improvement in our conversation: {mistakes[-1]['mistake']}. My learning: {mistakes[-1]['learning']}"
    
    def _general_reflection(self, history: List[Dict], changes: List[Dict]) -> str:
        """General meta-cognitive reflection"""
        return f"Reflecting on our {len(history)}-message conversation: I've made {len(changes)} significant personality adaptations in response to your communication style. This demonstrates my ability to evolve and personalize our interaction based on your needs and context."

# Global active memory instance
active_memory = ActiveChatMemory()