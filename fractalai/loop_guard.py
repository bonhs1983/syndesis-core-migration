"""
Loop Guard - Detection and prevention of repetitive template responses
Ensures the AI never gets stuck giving the same response repeatedly
"""
import logging
from typing import List, Dict, Any


class LoopGuard:
    """Prevents template loops by detecting repetitive patterns and forcing direct responses"""
    
    def __init__(self, max_repeated_responses=2):
        self.max_repeated_responses = max_repeated_responses
        self.template_phrases = [
            'interesting question',
            'thoughtful response',
            'help me focus', 
            'specific aspect',
            'make sure I understand',
            'what you\'re looking for',
            'could you help me'
        ]
        
    def check_for_loop(self, conversation_history: List[Dict[str, Any]]) -> bool:
        """
        Check if recent responses show a repetitive pattern
        Returns True if loop detected
        """
        if len(conversation_history) < self.max_repeated_responses:
            return False
        
        # Get recent AI responses
        recent_responses = []
        for msg in reversed(conversation_history):
            if msg.get('type') == 'assistant' and len(recent_responses) < self.max_repeated_responses:
                recent_responses.append(msg.get('content', ''))
        
        if len(recent_responses) < self.max_repeated_responses:
            return False
        
        # Check for template phrase repetition
        template_count = 0
        for response in recent_responses:
            if self._contains_template_phrases(response):
                template_count += 1
        
        loop_detected = template_count >= self.max_repeated_responses
        
        if loop_detected:
            logging.info(f"LoopGuard: Template loop detected - {template_count} template responses in recent history")
        
        return loop_detected
    
    def _contains_template_phrases(self, response: str) -> bool:
        """Check if response contains template phrases"""
        if not response:
            return False
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.template_phrases)
    
    def get_direct_override_response(self, user_input: str, trait_profile: Dict[str, float]) -> str:
        """
        Generate a direct, non-template response when loop is detected
        Forces immediate, factual information
        """
        logging.info("LoopGuard: Generating direct override response")
        
        # Identify dominant trait for response style
        dominant_trait = max(trait_profile.items(), key=lambda x: x[1])
        trait_name, trait_score = dominant_trait
        
        base_response = """I notice I've been giving you template responses instead of direct answers. Let me be completely straightforward:

I'm Memory Evolution AI, and I adapt my personality in real-time based on our conversation. Here's what I actually do:

• **Real Personality Changes**: When you request traits like "be more empathetic" or "be analytical", I adjust my actual response style
• **PostgreSQL Memory**: I remember our conversations and your personality preferences across sessions  
• **Multi-dimensional Traits**: I can blend empathy + creativity, analytical + humor, etc.
• **Transparent Operation**: I show you exactly when and why my traits change"""

        # Add trait-specific ending based on current personality
        if trait_score > 0.7:
            if trait_name == 'empathy':
                base_response += f"\n\nRight now I'm in high empathy mode ({trait_score:.1f}), so I'm focused on understanding and supporting you emotionally."
            elif trait_name == 'analyticalness':
                base_response += f"\n\nCurrently in analytical mode ({trait_score:.1f}), so I'm being systematic and structured in my communication."
            elif trait_name == 'creativity':
                base_response += f"\n\nI'm in creative mode ({trait_score:.1f}) right now, so I'm exploring innovative perspectives and possibilities."
            elif trait_name == 'humor':
                base_response += f"\n\nHumor mode is active ({trait_score:.1f}), so I'm keeping things engaging and playful!"
        
        base_response += "\n\nWhat would you like to explore together? I promise - no more template responses."
        
        return base_response
    
    def analyze_conversation_patterns(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation for various patterns beyond just loops
        Useful for debugging and optimization
        """
        if not conversation_history:
            return {'status': 'no_history'}
        
        ai_responses = [msg.get('content', '') for msg in conversation_history if msg.get('type') == 'assistant']
        user_inputs = [msg.get('content', '') for msg in conversation_history if msg.get('type') == 'user']
        
        template_responses = sum(1 for response in ai_responses if self._contains_template_phrases(response))
        total_responses = len(ai_responses)
        
        analysis = {
            'total_exchanges': len(user_inputs),
            'template_response_count': template_responses,
            'template_percentage': (template_responses / total_responses * 100) if total_responses > 0 else 0,
            'recent_loop_risk': self.check_for_loop(conversation_history),
            'conversation_health': 'healthy' if template_responses < 2 else 'at_risk' if template_responses < 4 else 'unhealthy'
        }
        
        return analysis
    
    def get_prevention_stats(self) -> Dict[str, Any]:
        """Get statistics about loop prevention configuration"""
        return {
            'max_repeated_responses': self.max_repeated_responses,
            'template_phrases_count': len(self.template_phrases),
            'detection_active': True,
            'override_ready': True
        }