"""
Production-Ready Trait-Based Response Engine
Every response goes through trait-to-style function with proper fallbacks and blending
"""
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class TraitResponseEngine:
    """
    Core engine that converts personality traits into actual response styles
    """
    
    def __init__(self):
        self.response_templates = self._initialize_response_templates()
        self.blending_templates = self._initialize_blending_templates()
        self.fallback_responses = self._initialize_fallback_responses()
        
    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize trait-specific response templates with variety"""
        return {
            'empathy': {
                'understanding': [
                    "I can really understand how you're feeling about this.",
                    "That sounds like it must be challenging for you.",
                    "I hear the emotion in what you're sharing with me."
                ],
                'supportive': [
                    "I'm here to support you through this.",
                    "You don't have to face this alone - I'm here to help.",
                    "Let me help you work through these feelings."
                ],
                'validation': [
                    "Your feelings about this are completely valid.",
                    "It makes perfect sense that you'd feel this way.",
                    "Anyone in your situation would have similar reactions."
                ]
            },
            'humor': {
                'playful': [
                    "Well, that's one way to keep things interesting! 😄",
                    "Looks like we're in for quite the adventure here!",
                    "Life certainly has a way of surprising us, doesn't it?"
                ],
                'witty': [
                    "I see what you did there - clever approach!",
                    "That's either genius or madness... possibly both!",
                    "You know what they say about great minds - they sometimes think alike!"
                ],
                'lighthearted': [
                    "Let's tackle this with a smile, shall we?",
                    "Time to put on our problem-solving party hats!",
                    "Every challenge is just an opportunity wearing a disguise!"
                ]
            },
            'analyticalness': {
                'structured': [
                    "Let me break this down into key components for clarity.",
                    "Here's a systematic approach to understanding this issue.",
                    "I'll analyze this step-by-step to give you a clear picture."
                ],
                'logical': [
                    "Based on the information provided, the logical conclusion is...",
                    "The data suggests several possible interpretations here.",
                    "From a rational perspective, we can identify these patterns."
                ],
                'methodical': [
                    "Let's examine this methodically, starting with the fundamentals.",
                    "I'll walk you through the reasoning process behind this.",
                    "The most effective approach involves these sequential steps."
                ]
            },
            'creativity': {
                'imaginative': [
                    "What if we approached this from a completely different angle?",
                    "Let me paint you a picture of an innovative solution.",
                    "Imagine this scenario as a blank canvas we can redesign."
                ],
                'innovative': [
                    "Here's a fresh perspective you might not have considered.",
                    "Let's think outside the conventional framework for a moment.",
                    "I have an unconventional idea that might just work perfectly."
                ],
                'artistic': [
                    "There's an elegant beauty in how we can solve this.",
                    "Like a masterpiece, the solution reveals itself layer by layer.",
                    "This reminds me of crafting something truly original."
                ]
            },
            'curiosity': {
                'inquisitive': [
                    "That's fascinating! Can you tell me more about that aspect?",
                    "I'm curious to understand the deeper layers of this situation.",
                    "What other factors might be influencing this dynamic?"
                ],
                'exploratory': [
                    "Let's explore all the dimensions of this topic together.",
                    "I wonder what happens if we dig deeper into this question.",
                    "There might be hidden connections we haven't discovered yet."
                ],
                'investigative': [
                    "This raises some intriguing questions worth investigating.",
                    "I'd love to explore the underlying mechanisms at work here.",
                    "Let's examine this from multiple angles to understand it fully."
                ]
            },
            'supportiveness': {
                'encouraging': [
                    "You're on the right track with this thinking!",
                    "I believe you have the strength to handle this challenge.",
                    "Your approach shows real wisdom and insight."
                ],
                'helpful': [
                    "I'm here to help you work through this step by step.",
                    "Let me provide some guidance to make this easier for you.",
                    "Together, we can find a path forward that works for you."
                ],
                'nurturing': [
                    "Take your time - there's no rush to figure this all out.",
                    "You're doing better than you might realize right now.",
                    "Every small step forward is meaningful progress."
                ]
            },
            'assertiveness': {
                'confident': [
                    "I'm confident that the best approach is clearly this:",
                    "Based on my analysis, I strongly recommend this path.",
                    "The evidence clearly points to this conclusion."
                ],
                'direct': [
                    "Let me be straightforward about what I think here.",
                    "I'll give you my honest, direct assessment of this situation.",
                    "Here's what you need to know, without any sugar-coating."
                ],
                'decisive': [
                    "The most effective action you can take right now is:",
                    "I recommend we move forward decisively with this plan.",
                    "This situation calls for clear, confident decision-making."
                ]
            }
        }
    
    def _initialize_blending_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for blending multiple traits"""
        return {
            'empathy_analytical': [
                "I understand how you feel ({empathy_phrase}), and let's also examine this logically ({analytical_phrase}).",
                "Your emotions about this are valid ({empathy_phrase}), while the data shows us ({analytical_phrase}).",
                "I can sense your feelings here ({empathy_phrase}), so let me break down the facts to help ({analytical_phrase})."
            ],
            'empathy_humor': [
                "I really understand your situation ({empathy_phrase}), and maybe we can find some lightness in it too ({humor_phrase}).",
                "Your feelings make complete sense ({empathy_phrase}), though I can't help but smile at ({humor_phrase}).",
                "I'm here to support you through this ({empathy_phrase}), and perhaps we can tackle it with some levity ({humor_phrase})."
            ],
            'analytical_creativity': [
                "The logical approach suggests ({analytical_phrase}), but what if we also considered this creative angle ({creativity_phrase})?",
                "Data points to this conclusion ({analytical_phrase}), though an innovative perspective might be ({creativity_phrase}).",
                "Systematically, we see ({analytical_phrase}), while imaginatively, we could explore ({creativity_phrase})."
            ],
            'humor_creativity': [
                "This is quite the entertaining puzzle ({humor_phrase}), and creatively speaking ({creativity_phrase})!",
                "I love the playful nature of this challenge ({humor_phrase}), which opens up imaginative possibilities like ({creativity_phrase}).",
                "There's something delightfully quirky here ({humor_phrase}), inspiring some creative thinking ({creativity_phrase})."
            ],
            'curiosity_analytical': [
                "I'm genuinely curious about this ({curiosity_phrase}), so let me analyze what we know ({analytical_phrase}).",
                "This raises fascinating questions ({curiosity_phrase}), which the data helps us understand ({analytical_phrase}).",
                "My interest is piqued by ({curiosity_phrase}), and systematic examination reveals ({analytical_phrase})."
            ],
            'supportiveness_assertiveness': [
                "I'm here to help you ({supportiveness_phrase}), and I believe you should confidently ({assertiveness_phrase}).",
                "You have my full support ({supportiveness_phrase}), so I recommend you decisively ({assertiveness_phrase}).",
                "Let me guide you through this ({supportiveness_phrase}), with the clear direction to ({assertiveness_phrase})."
            ]
        }
    
    def _initialize_fallback_responses(self) -> List[str]:
        """Initialize fallback responses for when trait mapping fails"""
        return [
            "I'm here to help you with whatever you need.",
            "Let me assist you in the best way I can.",
            "I'll do my best to provide you with a helpful response.",
            "Thank you for sharing that with me - how can I help?",
            "I appreciate you bringing this to my attention.",
            "Let me think about how to best address your question."
        ]
    
    def get_response_style(self, traits: Dict[str, float], context: Dict = None) -> Dict[str, Any]:
        """
        Determine response style based on trait profile with proper fallbacks
        """
        try:
            # Validate traits input
            if not traits or not isinstance(traits, dict):
                logging.warning("Invalid traits provided, using fallback")
                return {'style': 'neutral', 'primary_trait': None, 'blend': False}
            
            # Find dominant traits (> 0.7)
            dominant_traits = []
            for trait, value in traits.items():
                try:
                    trait_value = float(value) if value is not None else 0.5
                    if trait_value > 0.7:
                        dominant_traits.append((trait, trait_value))
                except (ValueError, TypeError):
                    logging.warning(f"Invalid trait value for {trait}: {value}")
                    continue
            
            # Sort by strength
            dominant_traits.sort(key=lambda x: x[1], reverse=True)
            
            # Determine response strategy
            if len(dominant_traits) == 0:
                return {'style': 'balanced', 'primary_trait': None, 'blend': False}
            elif len(dominant_traits) == 1:
                return {
                    'style': dominant_traits[0][0], 
                    'primary_trait': dominant_traits[0][0],
                    'trait_value': dominant_traits[0][1],
                    'blend': False
                }
            else:
                # Multiple dominant traits - check for blending
                primary = dominant_traits[0][0]
                secondary = dominant_traits[1][0]
                blend_key = f"{primary}_{secondary}"
                
                # Check if we have blending templates
                if blend_key in self.blending_templates:
                    return {
                        'style': 'blended',
                        'primary_trait': primary,
                        'secondary_trait': secondary,
                        'blend_key': blend_key,
                        'blend': True,
                        'primary_value': dominant_traits[0][1],
                        'secondary_value': dominant_traits[1][1]
                    }
                else:
                    # No blending available, prioritize primary
                    return {
                        'style': primary,
                        'primary_trait': primary,
                        'trait_value': dominant_traits[0][1],
                        'blend': False,
                        'note': f"Focusing on {primary} since blending with {secondary} isn't available"
                    }
                    
        except Exception as e:
            logging.error(f"Error in get_response_style: {e}")
            return {'style': 'neutral', 'primary_trait': None, 'blend': False, 'error': str(e)}
    
    def generate_response(self, user_input: str, traits: Dict[str, float], 
                         context: Dict = None, base_response: str = None) -> Dict[str, Any]:
        """
        Generate trait-influenced response with proper fallbacks
        """
        try:
            # Get response style
            style_info = self.get_response_style(traits, context)
            
            # Initialize response data
            response_data = {
                'response': '',
                'style_info': style_info,
                'traits_used': [],
                'explanation': '',
                'fallback_used': False
            }
            
            # Handle different response styles
            if style_info['blend']:
                response_data = self._generate_blended_response(
                    style_info, user_input, traits, context, base_response
                )
            elif style_info['primary_trait']:
                response_data = self._generate_single_trait_response(
                    style_info, user_input, traits, context, base_response
                )
            else:
                # Fallback to neutral response
                response_data = self._generate_fallback_response(user_input, base_response)
            
            # Add contextual memory if available
            if context and context.get('memory_recall'):
                response_data['response'] = self._add_memory_context(
                    response_data['response'], context['memory_recall']
                )
            
            # Ensure we never return empty response
            if not response_data['response'] or response_data['response'].strip() == '':
                response_data['response'] = self._get_safe_fallback()
                response_data['fallback_used'] = True
            
            return response_data
            
        except Exception as e:
            logging.error(f"Error in generate_response: {e}")
            return {
                'response': self._get_safe_fallback(),
                'style_info': {'style': 'error'},
                'traits_used': [],
                'explanation': f'Error occurred: {str(e)}',
                'fallback_used': True,
                'error': str(e)
            }
    
    def _generate_blended_response(self, style_info: Dict, user_input: str, 
                                 traits: Dict[str, float], context: Dict = None, 
                                 base_response: str = None) -> Dict[str, Any]:
        """Generate response blending two traits"""
        try:
            primary_trait = style_info['primary_trait']
            secondary_trait = style_info['secondary_trait']
            blend_key = style_info['blend_key']
            
            # Get phrase components for each trait
            primary_phrase = self._get_trait_phrase(primary_trait, user_input, context)
            secondary_phrase = self._get_trait_phrase(secondary_trait, user_input, context)
            
            # Get blending template
            templates = self.blending_templates.get(blend_key, [])
            if not templates:
                # Fallback to sequential approach
                return {
                    'response': f"{primary_phrase} {secondary_phrase}",
                    'style_info': style_info,
                    'traits_used': [primary_trait, secondary_trait],
                    'explanation': f"Combined {primary_trait} and {secondary_trait} responses",
                    'fallback_used': True
                }
            
            # Select template and fill in phrases
            template = random.choice(templates)
            blended_response = template.format(
                empathy_phrase=primary_phrase if primary_trait == 'empathy' else secondary_phrase,
                analytical_phrase=primary_phrase if primary_trait == 'analyticalness' else secondary_phrase,
                humor_phrase=primary_phrase if primary_trait == 'humor' else secondary_phrase,
                creativity_phrase=primary_phrase if primary_trait == 'creativity' else secondary_phrase,
                curiosity_phrase=primary_phrase if primary_trait == 'curiosity' else secondary_phrase,
                supportiveness_phrase=primary_phrase if primary_trait == 'supportiveness' else secondary_phrase,
                assertiveness_phrase=primary_phrase if primary_trait == 'assertiveness' else secondary_phrase
            )
            
            return {
                'response': blended_response,
                'style_info': style_info,
                'traits_used': [primary_trait, secondary_trait],
                'explanation': f"Blended {primary_trait} ({style_info['primary_value']:.1f}) with {secondary_trait} ({style_info['secondary_value']:.1f})",
                'fallback_used': False
            }
            
        except Exception as e:
            logging.error(f"Error in _generate_blended_response: {e}")
            return self._generate_fallback_response(user_input, base_response)
    
    def _generate_single_trait_response(self, style_info: Dict, user_input: str,
                                      traits: Dict[str, float], context: Dict = None,
                                      base_response: str = None) -> Dict[str, Any]:
        """Generate response based on single dominant trait"""
        try:
            trait = style_info['primary_trait']
            trait_value = style_info.get('trait_value', 0.7)
            
            # Get appropriate phrase for the trait
            response_phrase = self._get_trait_phrase(trait, user_input, context)
            
            return {
                'response': response_phrase,
                'style_info': style_info,
                'traits_used': [trait],
                'explanation': f"Response driven by {trait} trait (level: {trait_value:.1f})",
                'fallback_used': False
            }
            
        except Exception as e:
            logging.error(f"Error in _generate_single_trait_response: {e}")
            return self._generate_fallback_response(user_input, base_response)
    
    def _get_trait_phrase(self, trait: str, user_input: str, context: Dict = None) -> str:
        """Get appropriate phrase for a specific trait"""
        try:
            trait_templates = self.response_templates.get(trait, {})
            
            if not trait_templates:
                return self._get_safe_fallback()
            
            # Choose appropriate subcategory based on context
            subcategory = self._choose_subcategory(trait, user_input, context)
            phrases = trait_templates.get(subcategory, [])
            
            if not phrases:
                # Try first available subcategory
                for subcat, phrase_list in trait_templates.items():
                    if phrase_list:
                        phrases = phrase_list
                        break
            
            if phrases:
                return random.choice(phrases)
            else:
                return self._get_safe_fallback()
                
        except Exception as e:
            logging.error(f"Error getting trait phrase for {trait}: {e}")
            return self._get_safe_fallback()
    
    def _choose_subcategory(self, trait: str, user_input: str, context: Dict = None) -> str:
        """Choose appropriate subcategory based on input analysis"""
        try:
            input_lower = user_input.lower() if user_input else ""
            
            # Trait-specific subcategory selection
            if trait == 'empathy':
                if any(word in input_lower for word in ['sad', 'upset', 'hurt', 'pain', 'difficult']):
                    return 'understanding'
                elif any(word in input_lower for word in ['help', 'support', 'alone', 'scared']):
                    return 'supportive'
                else:
                    return 'validation'
                    
            elif trait == 'humor':
                if any(word in input_lower for word in ['serious', 'problem', 'issue']):
                    return 'lighthearted'
                elif any(word in input_lower for word in ['clever', 'smart', 'interesting']):
                    return 'witty'
                else:
                    return 'playful'
                    
            elif trait == 'analyticalness':
                if any(word in input_lower for word in ['analyze', 'breakdown', 'explain']):
                    return 'structured'
                elif any(word in input_lower for word in ['logic', 'reason', 'rational']):
                    return 'logical'
                else:
                    return 'methodical'
                    
            elif trait == 'creativity':
                if any(word in input_lower for word in ['new', 'different', 'alternative']):
                    return 'innovative'
                elif any(word in input_lower for word in ['beautiful', 'elegant', 'design']):
                    return 'artistic'
                else:
                    return 'imaginative'
                    
            elif trait == 'curiosity':
                if any(word in input_lower for word in ['why', 'how', 'what']):
                    return 'inquisitive'
                elif any(word in input_lower for word in ['explore', 'discover', 'find']):
                    return 'exploratory'
                else:
                    return 'investigative'
                    
            elif trait == 'supportiveness':
                if any(word in input_lower for word in ['try', 'attempt', 'effort']):
                    return 'encouraging'
                elif any(word in input_lower for word in ['help', 'guide', 'assist']):
                    return 'helpful'
                else:
                    return 'nurturing'
                    
            elif trait == 'assertiveness':
                if any(word in input_lower for word in ['should', 'must', 'need']):
                    return 'decisive'
                elif any(word in input_lower for word in ['honest', 'truth', 'reality']):
                    return 'direct'
                else:
                    return 'confident'
            
            # Default to first available subcategory
            trait_templates = self.response_templates.get(trait, {})
            if trait_templates:
                return list(trait_templates.keys())[0]
            
            return 'default'
            
        except Exception as e:
            logging.error(f"Error choosing subcategory for {trait}: {e}")
            return 'default'
    
    def _generate_fallback_response(self, user_input: str, base_response: str = None) -> Dict[str, Any]:
        """Generate safe fallback response"""
        try:
            if base_response and base_response.strip():
                response = base_response
            else:
                response = random.choice(self.fallback_responses)
            
            return {
                'response': response,
                'style_info': {'style': 'fallback'},
                'traits_used': [],
                'explanation': 'Using fallback response due to trait processing error',
                'fallback_used': True
            }
            
        except Exception as e:
            logging.error(f"Error in fallback response generation: {e}")
            return {
                'response': "I'm here to help you.",
                'style_info': {'style': 'emergency_fallback'},
                'traits_used': [],
                'explanation': 'Emergency fallback used',
                'fallback_used': True,
                'error': str(e)
            }
    
    def _add_memory_context(self, response: str, memory_recall: str) -> str:
        """Add memory context to response"""
        try:
            if memory_recall and memory_recall.strip():
                memory_phrases = [
                    f"Earlier, you mentioned {memory_recall}. {response}",
                    f"Remembering what you said about {memory_recall}, {response}",
                    f"Based on our previous conversation about {memory_recall}, {response}"
                ]
                return random.choice(memory_phrases)
            return response
            
        except Exception as e:
            logging.error(f"Error adding memory context: {e}")
            return response
    
    def _get_safe_fallback(self) -> str:
        """Get absolutely safe fallback response"""
        try:
            return random.choice(self.fallback_responses)
        except Exception:
            return "I'm here to help you."
    
    def explain_response_choice(self, response_data: Dict[str, Any]) -> str:
        """
        Generate explanation of why this response style was chosen
        """
        try:
            style_info = response_data.get('style_info', {})
            traits_used = response_data.get('traits_used', [])
            
            if response_data.get('fallback_used'):
                return "I used a general response because I couldn't properly analyze your personality traits for this interaction."
            
            if style_info.get('blend'):
                primary = style_info.get('primary_trait', 'unknown')
                secondary = style_info.get('secondary_trait', 'unknown')
                primary_val = style_info.get('primary_value', 0)
                secondary_val = style_info.get('secondary_value', 0)
                
                return (f"I blended my {primary} trait (level {primary_val:.1f}) with my {secondary} trait "
                       f"(level {secondary_val:.1f}) because both were highly active, creating a response "
                       f"that combines understanding with the appropriate analytical or emotional approach.")
            
            elif traits_used:
                trait = traits_used[0]
                trait_val = style_info.get('trait_value', 0)
                
                trait_explanations = {
                    'empathy': "emotional understanding and support",
                    'humor': "lightness and playfulness",
                    'analyticalness': "logical analysis and structure",
                    'creativity': "innovative and imaginative thinking",
                    'curiosity': "inquisitive exploration",
                    'supportiveness': "encouragement and guidance",
                    'assertiveness': "confidence and directness"
                }
                
                explanation = trait_explanations.get(trait, "that particular communication style")
                return (f"My {trait} trait was highly active (level {trait_val:.1f}), so I focused on "
                       f"{explanation} in my response to best match what seemed appropriate for your message.")
            
            return "I chose a balanced approach since no particular personality trait was strongly activated."
            
        except Exception as e:
            logging.error(f"Error explaining response choice: {e}")
            return "I'm not sure why I chose this particular response style."
    
    def get_template_variety_stats(self) -> Dict[str, Any]:
        """Get statistics about template variety for monitoring"""
        try:
            stats = {
                'total_traits': len(self.response_templates),
                'total_templates': 0,
                'templates_per_trait': {},
                'blending_combinations': len(self.blending_templates),
                'fallback_options': len(self.fallback_responses)
            }
            
            for trait, subcategories in self.response_templates.items():
                trait_total = sum(len(phrases) for phrases in subcategories.values())
                stats['templates_per_trait'][trait] = trait_total
                stats['total_templates'] += trait_total
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting template stats: {e}")
            return {'error': str(e)}


# Global instance for use across the application
trait_response_engine = TraitResponseEngine()