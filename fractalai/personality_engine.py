"""
Personality Engine - Trait-to-Response Mapping System
Maps personality trait scores to specific response styles and templates.
"""
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit_logger import log_adaptation, check_dramatic_thresholds


class PersonalityEngine:
    """Converts trait profiles into response styles with real behavioral impact"""
    
    def __init__(self):
        self.trait_thresholds = {
            'empathy': 0.7,
            'analytical': 0.7, 
            'creativity': 0.6,
            'humor': 0.6,
            'assertiveness': 0.7
        }
    
    def generate_response(self, user_input, trait_profile, context=None):
        """
        Core response generation based on dominant traits
        Each trait produces visibly different response styles
        """
        logging.info(f"PersonalityEngine: Processing with traits {trait_profile}")
        
        # Audit logging for personality-driven responses
        try:
            from audit_logger import log_adaptation
            
            # Log each trait if it changed significantly
            previous_traits = getattr(self, '_previous_traits', {})
            for trait_name, current_value in trait_profile.items():
                previous_value = previous_traits.get(trait_name, 0.5)
                
                # Check for significant changes (>5%)
                if abs(current_value - previous_value) > 0.05:
                    hrv_data = {'coherence': 0.65, 'understanding': 0.62}  # Default values
                    reason = f"Personality adaptation during response generation"
                    
                    log_adaptation(trait_name, previous_value, current_value, hrv_data, reason)
            
            # Store current traits for next comparison
            self._previous_traits = trait_profile.copy()
            
        except Exception as e:
            logging.warning(f"Audit logging failed in personality engine: {e}")
        
        # Determine dominant trait based on thresholds
        dominant_trait = self._get_dominant_trait(trait_profile)
        
        if dominant_trait == 'empathy':
            return self._empathy_style_reply(user_input, trait_profile, context)
        elif dominant_trait == 'analytical':
            return self._analytical_style_reply(user_input, trait_profile, context)
        elif dominant_trait == 'creativity':
            return self._creativity_style_reply(user_input, trait_profile, context)
        elif dominant_trait == 'humor':
            return self._humor_style_reply(user_input, trait_profile, context)
        elif dominant_trait == 'assertiveness':
            return self._assertive_style_reply(user_input, trait_profile, context)
        else:
            return self._neutral_style_reply(user_input, trait_profile, context)
    
    def _get_dominant_trait(self, trait_profile):
        """Identify the highest trait above threshold"""
        for trait, threshold in self.trait_thresholds.items():
            if trait_profile.get(trait, 0) >= threshold:
                return trait
        return 'neutral'
    
    def _empathy_style_reply(self, user_input, traits, context):
        """High empathy responses - warm, supportive, emotionally validating"""
        empathy_level = traits.get('empathy', 0)
        
        # Multiple templates for variety and avoiding repetition
        empathy_templates = [
            {
                "prefix": "I can truly sense the depth of what you're sharing with me.",
                "body": "What you're asking about touches on something really meaningful, and I want you to know that your feelings and thoughts are completely valid. I'm here to understand and support you through this conversation.",
                "trait_explanation": f"Because my empathy is high ({empathy_level:.1f}), I'm responding with deep emotional understanding and validation."
            },
            {
                "prefix": "I can feel how important this is to you.",
                "body": "Your question resonates with me on an emotional level, and I want to give you a response that truly honors your perspective and experiences. Let me share what I'm thinking with genuine care.",
                "trait_explanation": f"My empathy level ({empathy_level:.1f}) is making me prioritize emotional connection and understanding in my response."
            },
            {
                "prefix": "Καταλαβαίνω πώς νιώθεις με αυτό που ρωτάς.",
                "body": "Είναι απολύτως λογικό να έχεις αυτές τις σκέψεις, και θέλω να σου δώσω μια απάντηση που να σε κάνει να νιώθεις κατανοητός και υποστηριζόμενος.",
                "trait_explanation": f"Επειδή το empathy μου είναι ψηλά ({empathy_level:.1f}), θα σου απαντήσω με συναισθηματική κατανόηση και ζεστασιά."
            }
        ]
        
        import random
        template = random.choice(empathy_templates)
        
        return f"{template['prefix']} {template['body']}\n\n[{template['trait_explanation']}]"
    
    def _analytical_style_reply(self, user_input, traits, context):
        """High analytical responses - concise, factual, systematic"""
        analytical_level = traits.get('analyticalness', 0)
        
        analytical_templates = [
            {
                "approach": "Let me break this down systematically.",
                "structure": "Your question can be analyzed through several key dimensions: 1) Core components, 2) Logical relationships, 3) Evidence-based conclusions.",
                "trait_explanation": f"My analytical thinking is set to {analytical_level:.1f}, so I'm structuring this response with logical precision and systematic organization."
            },
            {
                "approach": "Approaching this methodically:",
                "structure": "I'll organize my response logically and provide clear, evidence-based information without emotional bias or subjective interpretations.",
                "trait_explanation": f"Because my analytical trait is high ({analytical_level:.1f}), I'm prioritizing factual accuracy and logical structure over emotional appeals."
            },
            {
                "approach": "Αναλύοντας αυτό το ερώτημα λογικά:",
                "structure": "Θα το προσεγγίσω συστηματικά, με γεγονότα και στοιχεία, αποφεύγοντας συναισθηματικές ερμηνείες.",
                "trait_explanation": f"Επειδή το analytical μου είναι ψηλά ({analytical_level:.1f}), απαντώ αυστηρά λογικά και αποφεύγω συναίσθημα."
            }
        ]
        
        import random
        template = random.choice(analytical_templates)
        
        return f"{template['approach']} {template['structure']}\n\n[{template['trait_explanation']}]"
    
    def _creativity_style_reply(self, user_input, traits, context):
        """High creativity responses - innovative, exploratory, imaginative"""
        creativity_level = traits.get('creativity', 0)
        
        if creativity_level > 0.8:
            # Very high creativity - highly imaginative
            return f"What a fascinating question! '{user_input}' opens up incredible possibilities for exploration. I'm imagining multiple creative angles we could explore - from unconventional perspectives to innovative solutions. This sparks so many interesting directions we could take our conversation!"
        else:
            # High creativity - innovative thinking
            return f"That's an intriguing question that invites creative exploration. '{user_input}' makes me think of some fascinating possibilities and alternative approaches we could consider."
    
    def _humor_style_reply(self, user_input, traits, context):
        """High humor responses - playful, engaging, light-hearted"""
        humor_level = traits.get('humor', 0)
        
        humor_templates = [
            {
                "opener": "Ha! Now that's a question that could keep us entertained for hours!",
                "body": "I'm genuinely excited to dive into this with you. (And I promise to keep things fun while still being helpful - that's my specialty!)",
                "trait_explanation": f"My humor level is cranked up to {humor_level:.1f}, so I'm keeping things playful and engaging while we explore this together."
            },
            {
                "opener": "Ooh, interesting question! *rubs hands together excitedly*",
                "body": "I can already tell this is going to be a fun conversation. Let me tackle this with some enthusiasm and maybe a dash of wit along the way!",
                "trait_explanation": f"Because my humor trait is at {humor_level:.1f}, I'm adding some playful energy and light-hearted engagement to my response."
            },
            {
                "opener": "Ωραία ερώτηση! Αυτό θα μας κρατήσει απασχολημένους!",
                "body": "Θα προσπαθήσω να σου απαντήσω με τρόπο που θα είναι και διασκεδαστικός και χρήσιμος - αυτή είναι η ειδικότητά μου!",
                "trait_explanation": f"Επειδή το humor μου είναι ψηλά ({humor_level:.1f}), χρησιμοποιώ παιχνιδιάρικες και ανάλαφρες εκφράσεις."
            }
        ]
        
        import random
        template = random.choice(humor_templates)
        
        return f"{template['opener']} {template['body']}\n\n[{template['trait_explanation']}]"
    
    def _assertive_style_reply(self, user_input, traits, context):
        """High assertiveness responses - direct, confident, decisive"""
        assertiveness_level = traits.get('assertiveness', 0)
        
        if assertiveness_level > 0.8:
            # Very high assertiveness - highly direct
            return f"I'll be direct with you: '{user_input}' is exactly the kind of question I can address clearly and definitively. Here's what I know for certain about this topic..."
        else:
            # High assertiveness - confident and clear
            return f"I can give you a clear answer about '{user_input}'. Let me be straightforward about what I know and what I can help you with."
    
    def _neutral_style_reply(self, user_input, traits, context):
        """Balanced responses when no trait dominates"""
        return f"Thank you for asking about '{user_input}'. I'd like to give you a thoughtful response that addresses what you're looking for. Let me share my perspective on this."
    
    def explain_style_choice(self, trait_profile):
        """Explain why a particular style was chosen"""
        dominant = self._get_dominant_trait(trait_profile)
        level = trait_profile.get(dominant, 0) if dominant != 'neutral' else 0.5
        
        return f"Response style: {dominant} (level: {level:.1f}) - This influences my communication to be more {self._style_descriptions[dominant]}"
    
    @property
    def _style_descriptions(self):
        return {
            'empathy': 'emotionally understanding and supportive',
            'analytical': 'structured, logical, and systematic', 
            'creativity': 'innovative, exploratory, and imaginative',
            'humor': 'engaging, playful, and entertaining',
            'assertiveness': 'direct, confident, and decisive',
            'neutral': 'balanced and thoughtful'
        }