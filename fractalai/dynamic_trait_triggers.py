"""
Dynamic Trait Triggers - Real triggers based on user input, not random changes
"""
import logging
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime


class DynamicTraitTriggers:
    """
    Analyzes user input to determine which traits should be triggered/adjusted
    """
    
    def __init__(self):
        self.trigger_patterns = self._initialize_trigger_patterns()
        self.emotional_indicators = self._initialize_emotional_indicators()
        self.interaction_types = self._initialize_interaction_types()
        
    def _initialize_trigger_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize patterns that trigger specific traits"""
        return {
            'empathy': {
                'emotional_distress': [
                    'sad', 'upset', 'hurt', 'pain', 'crying', 'depressed', 'anxious',
                    'worried', 'scared', 'afraid', 'lonely', 'heartbroken', 'devastated'
                ],
                'vulnerability': [
                    'struggling', 'difficult', 'hard time', 'overwhelmed', 'stressed',
                    'exhausted', 'lost', 'confused', 'helpless', 'vulnerable'
                ],
                'emotional_sharing': [
                    'feel like', 'feeling', 'emotions', 'heart', 'soul', 'deeply',
                    'personal', 'intimate', 'meaningful', 'touching'
                ]
            },
            'humor': {
                'playful_context': [
                    'funny', 'hilarious', 'joke', 'laugh', 'haha', 'lol', 'amusing',
                    'entertaining', 'witty', 'clever', 'silly', 'ridiculous'
                ],
                'light_situation': [
                    'fun', 'enjoyable', 'pleasant', 'delightful', 'cheerful',
                    'upbeat', 'positive', 'bright', 'sunny', 'happy'
                ],
                'irony_sarcasm': [
                    'ironic', 'sarcastic', 'kidding', 'joking', 'teasing',
                    'playful', 'tongue in cheek', 'wink wink'
                ]
            },
            'analyticalness': {
                'analysis_request': [
                    'analyze', 'breakdown', 'explain', 'understand', 'logic',
                    'reason', 'rational', 'systematic', 'methodical', 'structured'
                ],
                'data_focus': [
                    'data', 'statistics', 'numbers', 'facts', 'evidence',
                    'research', 'study', 'findings', 'results', 'metrics'
                ],
                'problem_solving': [
                    'solve', 'solution', 'problem', 'issue', 'challenge',
                    'approach', 'strategy', 'method', 'process', 'steps'
                ]
            },
            'creativity': {
                'creative_request': [
                    'creative', 'innovative', 'original', 'unique', 'novel',
                    'artistic', 'imaginative', 'inventive', 'inspired', 'vision'
                ],
                'design_context': [
                    'design', 'create', 'build', 'make', 'craft', 'compose',
                    'develop', 'generate', 'produce', 'construct'
                ],
                'alternative_thinking': [
                    'different', 'alternative', 'new way', 'fresh', 'unconventional',
                    'outside the box', 'think differently', 'another approach'
                ]
            },
            'curiosity': {
                'questions': [
                    'why', 'how', 'what', 'when', 'where', 'who', 'which',
                    'wonder', 'curious', 'interested', 'intrigued', 'fascinated'
                ],
                'exploration': [
                    'explore', 'discover', 'find out', 'learn', 'investigate',
                    'research', 'dig deeper', 'look into', 'examine', 'study'
                ],
                'mystery': [
                    'mystery', 'unknown', 'secret', 'hidden', 'unclear',
                    'puzzling', 'confusing', 'strange', 'odd', 'unusual'
                ]
            },
            'supportiveness': {
                'help_request': [
                    'help', 'assist', 'support', 'guide', 'advice', 'guidance',
                    'recommendation', 'suggestion', 'direction', 'assistance'
                ],
                'encouragement_needed': [
                    'trying', 'attempting', 'working on', 'effort', 'struggling',
                    'challenging', 'difficult', 'persevering', 'pushing through'
                ],
                'achievement_sharing': [
                    'accomplished', 'achieved', 'succeeded', 'completed', 'finished',
                    'proud', 'excited', 'happy', 'thrilled', 'satisfied'
                ]
            },
            'assertiveness': {
                'decision_needed': [
                    'should I', 'what do you think', 'recommend', 'suggest',
                    'decide', 'choice', 'option', 'best', 'right', 'correct'
                ],
                'confidence_request': [
                    'confident', 'sure', 'certain', 'definite', 'clear',
                    'direct', 'straightforward', 'honest', 'frank', 'blunt'
                ],
                'action_orientation': [
                    'action', 'do', 'act', 'move', 'step', 'implement',
                    'execute', 'proceed', 'forward', 'progress'
                ]
            }
        }
    
    def _initialize_emotional_indicators(self) -> Dict[str, List[str]]:
        """Initialize emotional state indicators"""
        return {
            'positive': [
                'happy', 'joy', 'excited', 'thrilled', 'pleased', 'satisfied',
                'content', 'grateful', 'blessed', 'amazing', 'wonderful', 'great'
            ],
            'negative': [
                'sad', 'angry', 'frustrated', 'disappointed', 'upset', 'hurt',
                'annoyed', 'irritated', 'mad', 'furious', 'terrible', 'awful'
            ],
            'neutral': [
                'okay', 'fine', 'alright', 'normal', 'usual', 'regular',
                'typical', 'standard', 'average', 'moderate', 'calm'
            ],
            'intense': [
                'extremely', 'incredibly', 'absolutely', 'completely', 'totally',
                'utterly', 'profoundly', 'deeply', 'intensely', 'overwhelming'
            ]
        }
    
    def _initialize_interaction_types(self) -> Dict[str, List[str]]:
        """Initialize interaction type patterns"""
        return {
            'question': ['?', 'how', 'what', 'why', 'when', 'where', 'who', 'which'],
            'statement': ['.', 'I think', 'I believe', 'I feel', 'I know', 'I understand'],
            'request': ['please', 'can you', 'could you', 'would you', 'help me', 'assist'],
            'exclamation': ['!', 'wow', 'amazing', 'incredible', 'fantastic', 'terrible']
        }
    
    def analyze_input_triggers(self, user_input: str, current_traits: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze user input to determine trait triggers and adjustments
        """
        try:
            if not user_input or not isinstance(user_input, str):
                return {'triggers': {}, 'adjustments': {}, 'analysis': 'No valid input provided'}
            
            input_lower = user_input.lower()
            
            # Analyze triggers for each trait
            triggered_traits = {}
            trait_adjustments = {}
            
            for trait, pattern_categories in self.trigger_patterns.items():
                trigger_strength = self._calculate_trigger_strength(input_lower, pattern_categories)
                
                if trigger_strength > 0:
                    triggered_traits[trait] = {
                        'strength': trigger_strength,
                        'current_value': current_traits.get(trait, 0.5),
                        'suggested_adjustment': self._calculate_adjustment(trigger_strength, current_traits.get(trait, 0.5))
                    }
                    
                    # Only suggest adjustment if it's meaningful
                    suggested_adj = triggered_traits[trait]['suggested_adjustment']
                    if abs(suggested_adj) > 0.05:  # Minimum threshold for changes
                        trait_adjustments[trait] = suggested_adj
            
            # Analyze emotional context
            emotional_context = self._analyze_emotional_context(input_lower)
            
            # Analyze interaction type
            interaction_type = self._determine_interaction_type(user_input)
            
            # Generate explanation
            explanation = self._generate_trigger_explanation(triggered_traits, emotional_context, interaction_type)
            
            return {
                'triggers': triggered_traits,
                'adjustments': trait_adjustments,
                'emotional_context': emotional_context,
                'interaction_type': interaction_type,
                'analysis': explanation,
                'input_processed': user_input[:100] + '...' if len(user_input) > 100 else user_input
            }
            
        except Exception as e:
            logging.error(f"Error analyzing input triggers: {e}")
            return {
                'triggers': {},
                'adjustments': {},
                'analysis': f'Error analyzing input: {str(e)}',
                'error': str(e)
            }
    
    def _calculate_trigger_strength(self, input_lower: str, pattern_categories: Dict[str, List[str]]) -> float:
        """Calculate how strongly the input triggers a specific trait"""
        try:
            total_matches = 0
            total_patterns = 0
            
            for category, patterns in pattern_categories.items():
                category_matches = 0
                for pattern in patterns:
                    if pattern in input_lower:
                        category_matches += 1
                
                # Weight categories differently
                category_weight = self._get_category_weight(category)
                total_matches += category_matches * category_weight
                total_patterns += len(patterns) * category_weight
            
            if total_patterns == 0:
                return 0.0
            
            # Normalize to 0-1 scale and apply diminishing returns
            raw_strength = total_matches / total_patterns
            return min(raw_strength * 2, 1.0)  # Scale up but cap at 1.0
            
        except Exception as e:
            logging.error(f"Error calculating trigger strength: {e}")
            return 0.0
    
    def _get_category_weight(self, category: str) -> float:
        """Get importance weight for different pattern categories"""
        category_weights = {
            'emotional_distress': 1.5,
            'vulnerability': 1.3,
            'analysis_request': 1.4,
            'creative_request': 1.3,
            'help_request': 1.2,
            'questions': 1.1,
            'decision_needed': 1.2,
            'playful_context': 1.0,
            'light_situation': 0.8,
            'design_context': 1.0,
            'data_focus': 1.1,
            'exploration': 1.0
        }
        return category_weights.get(category, 1.0)
    
    def _calculate_adjustment(self, trigger_strength: float, current_value: float) -> float:
        """Calculate suggested trait adjustment based on trigger strength"""
        try:
            # Base adjustment proportional to trigger strength
            base_adjustment = trigger_strength * 0.3  # Max 0.3 adjustment
            
            # Consider current trait level (don't over-boost already high traits)
            if current_value > 0.8:
                base_adjustment *= 0.3  # Reduce adjustment for already high traits
            elif current_value < 0.3:
                base_adjustment *= 1.5  # Increase adjustment for low traits
            
            # Ensure we don't exceed bounds
            max_possible = 1.0 - current_value
            min_possible = -current_value
            
            return max(min_possible, min(max_possible, base_adjustment))
            
        except Exception as e:
            logging.error(f"Error calculating adjustment: {e}")
            return 0.0
    
    def _analyze_emotional_context(self, input_lower: str) -> Dict[str, Any]:
        """Analyze emotional context of the input"""
        try:
            emotional_scores = {}
            
            for emotion, indicators in self.emotional_indicators.items():
                score = sum(1 for indicator in indicators if indicator in input_lower)
                if score > 0:
                    emotional_scores[emotion] = score
            
            if not emotional_scores:
                return {'dominant_emotion': 'neutral', 'intensity': 0.0, 'confidence': 0.0}
            
            # Find dominant emotion
            dominant_emotion = max(emotional_scores.keys(), key=emotional_scores.get)
            max_score = emotional_scores[dominant_emotion]
            
            # Calculate intensity and confidence
            total_indicators = sum(len(indicators) for indicators in self.emotional_indicators.values())
            intensity = min(max_score / 3, 1.0)  # Normalize intensity
            confidence = max_score / max(1, sum(emotional_scores.values()))
            
            return {
                'dominant_emotion': dominant_emotion,
                'intensity': intensity,
                'confidence': confidence,
                'all_scores': emotional_scores
            }
            
        except Exception as e:
            logging.error(f"Error analyzing emotional context: {e}")
            return {'dominant_emotion': 'neutral', 'intensity': 0.0, 'confidence': 0.0}
    
    def _determine_interaction_type(self, user_input: str) -> str:
        """Determine the type of interaction"""
        try:
            input_lower = user_input.lower()
            type_scores = {}
            
            for interaction_type, indicators in self.interaction_types.items():
                score = sum(1 for indicator in indicators if indicator in input_lower or indicator in user_input)
                if score > 0:
                    type_scores[interaction_type] = score
            
            if not type_scores:
                return 'statement'  # Default
            
            return max(type_scores.keys(), key=type_scores.get)
            
        except Exception as e:
            logging.error(f"Error determining interaction type: {e}")
            return 'statement'
    
    def _generate_trigger_explanation(self, triggered_traits: Dict, emotional_context: Dict, 
                                    interaction_type: str) -> str:
        """Generate explanation of what triggers were detected"""
        try:
            if not triggered_traits:
                return f"No strong trait triggers detected in this {interaction_type}."
            
            explanations = []
            
            for trait, data in triggered_traits.items():
                strength = data['strength']
                adjustment = data['suggested_adjustment']
                
                if strength > 0.7:
                    intensity = "strongly"
                elif strength > 0.4:
                    intensity = "moderately"
                else:
                    intensity = "lightly"
                
                direction = "increase" if adjustment > 0 else "maintain"
                explanations.append(f"{trait} {intensity} triggered (suggesting {direction})")
            
            base_explanation = f"Detected {interaction_type} with " + ", ".join(explanations)
            
            # Add emotional context if significant
            if emotional_context.get('intensity', 0) > 0.5:
                emotion = emotional_context['dominant_emotion']
                base_explanation += f". Emotional tone: {emotion}"
            
            return base_explanation
            
        except Exception as e:
            logging.error(f"Error generating trigger explanation: {e}")
            return "Unable to analyze trigger patterns."
    
    def apply_dynamic_adjustments(self, current_traits: Dict[str, float], 
                                adjustments: Dict[str, float], 
                                reason: str) -> Dict[str, float]:
        """Apply dynamic trait adjustments with safety checks"""
        try:
            updated_traits = current_traits.copy()
            
            for trait, adjustment in adjustments.items():
                if trait in updated_traits:
                    old_value = updated_traits[trait]
                    new_value = max(0.0, min(1.0, old_value + adjustment))
                    updated_traits[trait] = new_value
                    
                    logging.info(f"Dynamic adjustment: {trait} {old_value:.2f} -> {new_value:.2f} ({reason})")
            
            return updated_traits
            
        except Exception as e:
            logging.error(f"Error applying dynamic adjustments: {e}")
            return current_traits
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get statistics about trigger patterns for monitoring"""
        try:
            stats = {
                'total_traits': len(self.trigger_patterns),
                'patterns_per_trait': {},
                'total_patterns': 0,
                'emotional_indicators': len(self.emotional_indicators),
                'interaction_types': len(self.interaction_types)
            }
            
            for trait, categories in self.trigger_patterns.items():
                trait_total = sum(len(patterns) for patterns in categories.values())
                stats['patterns_per_trait'][trait] = trait_total
                stats['total_patterns'] += trait_total
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting trigger statistics: {e}")
            return {'error': str(e)}


# Global instance for use across the application
dynamic_trait_triggers = DynamicTraitTriggers()