"""
Advanced trait blending system for multi-dimensional personality responses
"""
import logging
from typing import Dict, List, Tuple

class TraitBlender:
    """Handles complex trait combinations and blended response generation"""
    
    def __init__(self):
        self.trait_templates = {
            'empathy': {
                'high': [
                    "I can truly sense what you're going through, and I want you to know that your feelings are completely valid.",
                    "Your experience matters deeply to me, and I'm here to understand and support you through this.",
                    "I can feel the weight of what you're sharing, and I want to respond with the care this deserves."
                ],
                'medium': [
                    "I understand this is important to you, and I want to be helpful.",
                    "I can see this matters, and I'm here to support you."
                ],
                'low': [
                    "I acknowledge your statement.",
                    "I note what you've said."
                ]
            },
            'analytical': {
                'high': [
                    "Let me break this down systematically and examine each component.",
                    "From an analytical perspective, we can approach this by identifying the key variables.",
                    "I'll structure my analysis by examining the logical relationships here."
                ],
                'medium': [
                    "Let me think through this logically.",
                    "I can analyze this step by step."
                ],
                'low': [
                    "This seems straightforward.",
                    "I can see the basic pattern here."
                ]
            },
            'creativity': {
                'high': [
                    "This opens up fascinating creative possibilities! Let me explore some innovative angles.",
                    "I'm excited to think outside the box here and explore unexpected connections.",
                    "What an intriguing challenge - there are so many creative directions we could take this!"
                ],
                'medium': [
                    "There are some interesting creative approaches we could try.",
                    "Let me think of some alternative perspectives."
                ],
                'low': [
                    "I can think of a few standard approaches.",
                    "There are typical ways to handle this."
                ]
            },
            'humor': {
                'high': [
                    "Well, this is getting interesting! *adjusts imaginary thinking cap*",
                    "Oh, the plot thickens! This calls for some wit and wisdom.",
                    "Time to deploy my premium humor protocols - fair warning, puns may occur!"
                ],
                'medium': [
                    "I can't help but find this a bit amusing.",
                    "There's definitely a lighter side to this."
                ],
                'low': [
                    "I'll keep this serious.",
                    "Let me focus on the facts."
                ]
            }
        }
        
        self.blend_patterns = {
            ('empathy', 'analytical'): "I want to understand your feelings while also thinking through this systematically.",
            ('empathy', 'creativity'): "I'm deeply moved by this and excited to explore creative solutions together.",
            ('empathy', 'humor'): "I care about what you're going through, and maybe we can find some lightness in this too.",
            ('analytical', 'creativity'): "Let me systematically explore some innovative approaches to this challenge.",
            ('analytical', 'humor'): "Time for some logical thinking with a dash of wit - the best kind of problem-solving!",
            ('creativity', 'humor'): "This calls for imaginative thinking with a playful twist - my favorite combination!"
        }
    
    def get_trait_level(self, score: float) -> str:
        """Convert numeric score to trait level"""
        if score > 0.7:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def generate_blended_response(self, personality: Dict[str, float], base_content: str) -> str:
        """Generate a response that blends multiple active traits"""
        dominant_traits = [(trait, score) for trait, score in personality.items() 
                          if score > 0.6 and trait in self.trait_templates]
        dominant_traits.sort(key=lambda x: x[1], reverse=True)
        
        if len(dominant_traits) == 0:
            return base_content
        elif len(dominant_traits) == 1:
            # Single dominant trait
            trait, score = dominant_traits[0]
            level = self.get_trait_level(score)
            template = self.trait_templates[trait][level][0]  # Use first template
            return f"{template} {base_content}"
        else:
            # Multiple traits - blend them
            trait1, score1 = dominant_traits[0]
            trait2, score2 = dominant_traits[1]
            
            # Check for predefined blend pattern
            blend_key = tuple(sorted([trait1, trait2]))
            if blend_key in self.blend_patterns:
                blend_intro = self.blend_patterns[blend_key]
                return f"{blend_intro} {base_content}"
            else:
                # Fallback: combine individual trait expressions
                level1 = self.get_trait_level(score1)
                level2 = self.get_trait_level(score2)
                template1 = self.trait_templates[trait1][level1][0]
                template2 = self.trait_templates[trait2][level2][0]
                return f"{template1} {template2} {base_content}"
    
    def explain_blend(self, personality: Dict[str, float]) -> str:
        """Explain the current trait blend to the user"""
        active_traits = [(trait, score) for trait, score in personality.items() if score > 0.6]
        active_traits.sort(key=lambda x: x[1], reverse=True)
        
        if not active_traits:
            return "Currently using balanced, neutral traits."
        elif len(active_traits) == 1:
            trait, score = active_traits[0]
            return f"High {trait} mode active (score: {score:.1f}) - this shapes my entire response style."
        else:
            trait_descriptions = [f"{trait} ({score:.1f})" for trait, score in active_traits[:2]]
            return f"Blending {' and '.join(trait_descriptions)} for a multi-dimensional response style."