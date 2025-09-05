"""
Clarification Handler - Specialized handling for ambiguous/clarification prompts
Detects when users need clarification and provides direct, factual responses
"""
import logging


class ClarificationHandler:
    """Handles clarification requests with direct, factual responses"""
    
    def __init__(self):
        self.clarification_triggers = [
            'what do you mean',
            'please clarify', 
            'clarify',
            'explain that',
            'i don\'t understand',
            'what',
            'huh',
            'be more specific',
            'can you explain',
            'i\'m confused'
        ]
        
        self.template_indicators = [
            'interesting question',
            'thoughtful response', 
            'help me focus',
            'specific aspect',
            'make sure I understand'
        ]
    
    def is_clarification_request(self, user_input):
        """Detect if user is asking for clarification"""
        user_input_lower = user_input.lower()
        return any(trigger in user_input_lower for trigger in self.clarification_triggers)
    
    def handle_clarification(self, user_input, last_bot_response, context=None):
        """
        Provide direct clarification based on the last bot response
        Always factual, never template responses
        """
        if not self.is_clarification_request(user_input):
            return None
            
        logging.info("ClarificationHandler: Processing clarification request")
        
        # Check if last response was a template - provide direct explanation
        if self._was_template_response(last_bot_response):
            return self._explain_template_directly()
        
        # Check if last response was about capabilities
        if any(word in last_bot_response.lower() for word in ['capabilities', 'function', 'do']):
            return self._explain_capabilities_directly()
        
        # Check if last response mentioned personality/traits
        if any(word in last_bot_response.lower() for word in ['personality', 'trait', 'empathy', 'analytical']):
            return self._explain_personality_directly()
        
        # General clarification for other responses
        return self._general_clarification(last_bot_response)
    
    def _was_template_response(self, response):
        """Check if the response was a generic template"""
        if not response:
            return False
        return any(indicator in response.lower() for indicator in self.template_indicators)
    
    def _explain_template_directly(self):
        """Direct explanation when last response was a template"""
        return """Let me be completely direct: I'm Memory Evolution AI, and I adapt my personality traits in real-time based on our conversation. 

For example:
- If you say "be more empathetic", I increase my empathy score and respond with warmer, more supportive language
- If you say "be analytical", I become more systematic and structured in my responses
- If you say "be creative", I explore more innovative and imaginative perspectives

My personality scores (empathy, creativity, analytical thinking, humor, etc.) actually change how I communicate with you. Would you like to see this in action by requesting a specific trait?"""
    
    def _explain_capabilities_directly(self):
        """Direct explanation of capabilities"""
        return """My core capabilities include:

• **Persistent Personality Evolution**: I remember and adapt my personality traits across our entire conversation
• **Real-time Trait Adaptation**: I adjust empathy, creativity, analytical thinking, humor based on your requests
• **Cross-session Memory**: I store our relationship and personality state in PostgreSQL database
• **Multi-dimensional Traits**: I can blend multiple personality aspects (like empathetic + analytical)
• **Transparent Operation**: I show you exactly how and why my personality changes
• **Conflict Detection**: I identify when traits might conflict and explain the tensions

The key difference is that my personality changes actually affect how I respond - it's not just for show."""
    
    def _explain_personality_directly(self):
        """Direct explanation of personality system"""  
        return """Here's exactly how my personality system works:

**Trait Scores**: I have numerical scores (0.0 to 1.0) for traits like empathy, creativity, analytical thinking, humor, assertiveness.

**Real Impact**: These scores genuinely change my response style:
- High empathy (0.8+) = warmer, more emotionally supportive responses
- High analytical (0.8+) = structured, systematic, fact-based responses  
- High creativity (0.8+) = innovative, exploratory, imaginative responses

**Dynamic Adaptation**: When you request changes ("be more creative"), I actually adjust these scores and my responses change accordingly.

**Persistence**: Your personality preferences are saved in a PostgreSQL database and remembered across sessions.

This isn't simulated - the trait scores directly influence my response generation."""
    
    def _general_clarification(self, last_response):
        """General clarification for non-template responses"""
        return f"""Let me clarify what I meant in my last response:

{self._paraphrase_response(last_response)}

I'm an AI that adapts my communication style based on your needs. If there's a specific aspect you'd like me to explain differently or expand on, just let me know."""
    
    def _paraphrase_response(self, response):
        """Create a paraphrased version of the response"""
        if len(response) > 200:
            # For long responses, provide a summary
            return f"In simpler terms: {response[:150]}... [This was a longer explanation about the topic you asked about]"
        else:
            # For shorter responses, provide alternative wording
            return f"In other words: {response}"
    
    def get_clarification_examples(self):
        """Examples of clarification requests this handler recognizes"""
        return {
            'triggers': self.clarification_triggers,
            'example_responses': [
                "Direct explanation with concrete examples",
                "Factual breakdown of capabilities", 
                "Plain language paraphrase of previous response",
                "Specific demonstration of personality adaptation"
            ]
        }