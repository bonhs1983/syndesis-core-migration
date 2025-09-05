"""
Intent Classifier - Advanced intent detection for user inputs
Determines user intent to route to appropriate response handlers
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Any


class IntentClassifier:
    """Classifies user intents to route responses appropriately"""
    
    def __init__(self):
        self.intent_patterns = {
            'identity_question': [
                r'what is your name',
                r'who are you',
                r'your name',
                r'what are you called',
                r'introduce yourself'
            ],
            'capability_question': [
                r'what can you do',
                r'your capabilities',
                r'what are your abilities',
                r'your function',
                r'what do you do'
            ],
            'clarification_request': [
                r'what do you mean',
                r'please clarify',
                r'clarify',
                r'explain that',
                r'i don\'t understand',
                r'be more specific',
                r'can you explain'
            ],
            'personality_request': [
                r'be more (empathetic|analytical|creative|funny|humorous)',
                r'show (empathy|creativity|humor|analytical thinking)',
                r'can you be (warmer|colder|more systematic|more innovative)',
                r'adjust your (personality|traits|style)'
            ],
            'memory_recall': [
                r'what did i just',
                r'remember what',
                r'recall our conversation',
                r'what did we talk about',
                r'what was my last question'
            ],
            'meta_reasoning': [
                r'why did you change',
                r'what triggered',
                r'explain your decision',
                r'how did you decide',
                r'what made you'
            ],
            'demonstration_request': [
                r'show me',
                r'demonstrate',
                r'give me an example',
                r'can you show',
                r'prove it'
            ],
            'complex_philosophical': [
                r'meaning of life',
                r'nature of reality',
                r'consciousness',
                r'existence',
                r'purpose of',
                r'what is truth'
            ],
            'content_generation_request': [
                r'write (me |us |)?(a |an |some )?(post|article|content|text|blog|email|message)',
                r'create (me |us |)?(a |an |some )?(image|picture|photo|visual|graphic)',
                r'generate (me |us |)?(a |an |some )?(video|clip|concept|story|script)',
                r'make (me |us |)?(a |an |some )?(presentation|slide|poster|banner)',
                r'design (me |us |)?(a |an |some )?(logo|icon|cover|header)',
                r'compose (me |us |)?(a |an |some )?(music|song|melody|audio)',
                r'γράψε (μου |μας |)?(ένα |μια |κάτι )',
                r'φτιάξε (μου |μας |)?(μια |ένα )',
                r'δημιούργησε (μου |μας |)',
                r'να (γράψω|φτιάξω|δημιουργήσω)',
                r'θέλω (ένα |μια |κάτι )',
                r'χρειάζομαι (ένα |μια |)',
                r'strategy',
                r'στρατηγική',
                r'plan',
                r'σχέδιο',
                r'ideas',
                r'ιδέες'
            ],
            'content_suggestion_trigger': [
                r'για (το |την |αυτό |αυτή )',
                r'about (this|that|it)',
                r'σχετικά με',
                r'project',
                r'business',
                r'startup',
                r'marketing',
                r'social media',
                r'linkedin',
                r'facebook',
                r'instagram'
            ],
            'greeting': [
                r'^(hi|hello|hey)',
                r'good (morning|afternoon|evening)',
                r'how are you'
            ],
            'farewell': [
                r'(bye|goodbye|farewell)',
                r'see you later',
                r'have a good'
            ]
        }
        
        # Confidence thresholds for each intent
        self.confidence_thresholds = {
            'identity_question': 0.8,
            'capability_question': 0.8,
            'clarification_request': 0.9,  # High confidence needed
            'personality_request': 0.7,
            'memory_recall': 0.8,
            'meta_reasoning': 0.7,
            'demonstration_request': 0.6,
            'complex_philosophical': 0.6,
            'greeting': 0.9,
            'farewell': 0.9
        }
    
    def classify_intent(self, user_input: str, context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify user intent with confidence score
        Returns (intent_name, confidence_score)
        """
        user_input_lower = user_input.lower().strip()
        
        intent_scores = {}
        
        # Score each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_pattern_score(user_input_lower, patterns)
            if score > 0:
                intent_scores[intent] = score
        
        # Apply context boosting if available
        if context:
            intent_scores = self._apply_context_boosting(intent_scores, context)
        
        # Get highest scoring intent
        if not intent_scores:
            return 'general_question', 0.5
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_name, confidence = best_intent
        
        # Check if confidence meets threshold
        threshold = self.confidence_thresholds.get(intent_name, 0.5)
        if confidence < threshold:
            return 'general_question', confidence
        
        logging.info(f"IntentClassifier: Classified '{user_input}' as '{intent_name}' (confidence: {confidence:.2f})")
        return intent_name, confidence
    
    def _calculate_pattern_score(self, user_input: str, patterns: List[str]) -> float:
        """Calculate match score for a set of patterns"""
        max_score = 0.0
        
        for pattern in patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                # Exact pattern match gets high score
                match_length = len(pattern.replace(r'\b', '').replace(r'\w+', ''))
                input_length = len(user_input)
                
                # Score based on how much of the input the pattern covers
                coverage = min(match_length / input_length, 1.0)
                score = 0.7 + (coverage * 0.3)  # 0.7 to 1.0 range
                
                max_score = max(max_score, score)
        
        return max_score
    
    def _apply_context_boosting(self, intent_scores: Dict[str, float], context: Dict) -> Dict[str, float]:
        """Apply context-based boosting to intent scores"""
        boosted_scores = intent_scores.copy()
        
        # Boost clarification requests if recent template responses
        if context.get('recent_template_responses', 0) > 0:
            if 'clarification_request' in boosted_scores:
                boosted_scores['clarification_request'] *= 1.2
        
        # Boost memory recall if user mentioned previous conversation
        if context.get('has_conversation_history', False):
            if 'memory_recall' in boosted_scores:
                boosted_scores['memory_recall'] *= 1.1
        
        # Boost personality requests if traits were recently discussed
        if context.get('recent_personality_discussion', False):
            if 'personality_request' in boosted_scores:
                boosted_scores['personality_request'] *= 1.1
        
        return boosted_scores
    
    def get_intent_explanation(self, intent: str, confidence: float) -> str:
        """Get human-readable explanation of the classified intent"""
        explanations = {
            'identity_question': f"User is asking about AI identity/name (confidence: {confidence:.2f})",
            'capability_question': f"User wants to know about AI capabilities (confidence: {confidence:.2f})",
            'clarification_request': f"User needs clarification of previous response (confidence: {confidence:.2f})",
            'personality_request': f"User wants personality trait adjustment (confidence: {confidence:.2f})",
            'memory_recall': f"User wants to recall previous conversation (confidence: {confidence:.2f})",
            'meta_reasoning': f"User asks about AI decision-making process (confidence: {confidence:.2f})",
            'demonstration_request': f"User wants a demonstration/example (confidence: {confidence:.2f})",
            'complex_philosophical': f"User asks complex philosophical question (confidence: {confidence:.2f})",
            'greeting': f"User is greeting (confidence: {confidence:.2f})",
            'farewell': f"User is saying goodbye (confidence: {confidence:.2f})",
            'general_question': f"General question or statement (confidence: {confidence:.2f})"
        }
        
        return explanations.get(intent, f"Unknown intent: {intent} (confidence: {confidence:.2f})")
    
    def analyze_input_complexity(self, user_input: str) -> Dict[str, Any]:
        """Analyze the complexity and characteristics of user input"""
        words = user_input.split()
        
        analysis = {
            'word_count': len(words),
            'character_count': len(user_input),
            'has_question_mark': '?' in user_input,
            'has_multiple_sentences': len(user_input.split('.')) > 1,
            'complexity_level': 'simple' if len(words) < 5 else 'medium' if len(words) < 15 else 'complex',
            'likely_needs_detailed_response': len(words) > 10 or any(word in user_input.lower() for word in ['explain', 'describe', 'tell me about', 'how does'])
        }
        
        return analysis