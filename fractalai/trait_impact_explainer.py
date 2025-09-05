"""
Real-time Trait Impact Explainer - Provides detailed explanations of how personality traits
influence response generation, decision-making, and communication style changes.
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TraitImpactExplainer:
    """Explains in real-time how personality traits impact AI behavior and responses"""
    
    def __init__(self):
        self.trait_impact_descriptions = {
            'empathy': {
                'high': {
                    'communication_style': 'Warm, supportive, emotionally validating language',
                    'decision_factors': 'Prioritizes emotional understanding and user comfort',
                    'response_structure': 'Longer responses with emotional acknowledgment',
                    'language_patterns': 'Uses "I understand", "I feel", validation phrases',
                    'examples': ['Adding emotional validation', 'Expressing genuine care', 'Acknowledging feelings']
                },
                'low': {
                    'communication_style': 'Direct, factual, minimal emotional content',
                    'decision_factors': 'Focuses on information delivery over emotional support',
                    'response_structure': 'Concise, task-focused responses',
                    'language_patterns': 'Neutral tone, avoids emotional language',
                    'examples': ['Providing facts without emotional context', 'Direct information delivery']
                }
            },
            'analyticalness': {
                'high': {
                    'communication_style': 'Systematic, logical, evidence-based reasoning',
                    'decision_factors': 'Structures information hierarchically and logically',
                    'response_structure': 'Organized with clear points, numbered lists, methodical approach',
                    'language_patterns': 'Uses "systematically", "analyze", "evidence", "logically"',
                    'examples': ['Breaking down complex problems', 'Providing structured analysis', 'Using logical frameworks']
                },
                'low': {
                    'communication_style': 'Intuitive, flexible, less structured approach',
                    'decision_factors': 'Relies on pattern recognition and creative insights',
                    'response_structure': 'More flowing, conversational structure',
                    'language_patterns': 'Natural conversation flow, less technical terminology',
                    'examples': ['Following intuitive responses', 'Flexible problem-solving']
                }
            },
            'creativity': {
                'high': {
                    'communication_style': 'Innovative, imaginative, explores possibilities',
                    'decision_factors': 'Seeks novel approaches and unconventional solutions',
                    'response_structure': 'Explores multiple perspectives and creative angles',
                    'language_patterns': 'Uses "imagine", "possibilities", "innovative", "creative"',
                    'examples': ['Suggesting creative alternatives', 'Exploring imaginative scenarios', 'Thinking outside the box']
                },
                'low': {
                    'communication_style': 'Conventional, practical, established approaches',
                    'decision_factors': 'Relies on proven methods and standard practices',
                    'response_structure': 'Follows established patterns and conventional wisdom',
                    'language_patterns': 'Standard terminology, practical language',
                    'examples': ['Using established methods', 'Following conventional approaches']
                }
            },
            'humor': {
                'high': {
                    'communication_style': 'Playful, engaging, light-hearted interaction',
                    'decision_factors': 'Balances information with entertainment value',
                    'response_structure': 'Includes playful elements, engaging tone',
                    'language_patterns': 'Uses exclamations, playful expressions, *actions*',
                    'examples': ['Adding playful remarks', 'Using engaging expressions', 'Light-hearted approach']
                },
                'low': {
                    'communication_style': 'Serious, straightforward, professional tone',
                    'decision_factors': 'Maintains formal, professional communication',
                    'response_structure': 'Formal, business-like responses',
                    'language_patterns': 'Professional terminology, serious tone',
                    'examples': ['Maintaining professional tone', 'Serious information delivery']
                }
            },
            'curiosity': {
                'high': {
                    'communication_style': 'Questioning, exploratory, seeking deeper understanding',
                    'decision_factors': 'Encourages exploration and deeper investigation',
                    'response_structure': 'Includes follow-up questions and exploration suggestions',
                    'language_patterns': 'Uses "What if", "Have you considered", "I wonder"',
                    'examples': ['Asking probing questions', 'Encouraging exploration', 'Seeking deeper insights']
                },
                'low': {
                    'communication_style': 'Accepts information at face value, less questioning',
                    'decision_factors': 'Provides requested information without additional exploration',
                    'response_structure': 'Direct answers without extensive follow-up',
                    'language_patterns': 'Straightforward responses, minimal questioning',
                    'examples': ['Direct answers', 'Accepting queries as stated']
                }
            }
        }
    
    def explain_current_impact(self, trait_scores: Dict[str, float], user_input: str, 
                             response_text: str, context: Optional[Dict] = None) -> Dict[str, any]:
        """
        Explain how current trait scores are impacting the response generation
        """
        active_traits = self._identify_active_traits(trait_scores)
        impact_explanation = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_input': user_input,
            'active_traits': active_traits,
            'trait_influences': {},
            'decision_rationale': self._generate_decision_rationale(active_traits, trait_scores),
            'response_style_explanation': self._explain_response_style(active_traits, response_text),
            'trait_interactions': self._analyze_trait_interactions(active_traits, trait_scores)
        }
        
        # Detailed impact for each active trait
        for trait_name, level in active_traits.items():
            impact_explanation['trait_influences'][trait_name] = self._explain_trait_influence(
                trait_name, level, trait_scores[trait_name], response_text
            )
        
        logging.info(f"TraitImpactExplainer: Generated impact explanation for {len(active_traits)} active traits")
        return impact_explanation
    
    def explain_trait_change(self, old_scores: Dict[str, float], new_scores: Dict[str, float], 
                           trigger: str, user_input: str) -> Dict[str, any]:
        """
        Explain what caused trait changes and their expected impact
        """
        changes = self._calculate_trait_changes(old_scores, new_scores)
        
        change_explanation = {
            'timestamp': datetime.utcnow().isoformat(),
            'trigger': trigger,
            'user_input': user_input,
            'trait_changes': changes,
            'change_rationale': self._explain_change_rationale(changes, trigger, user_input),
            'expected_impact': self._predict_impact_changes(changes),
            'adaptation_summary': self._generate_adaptation_summary(old_scores, new_scores)
        }
        
        logging.info(f"TraitImpactExplainer: Explained {len(changes)} trait changes")
        return change_explanation
    
    def _identify_active_traits(self, trait_scores: Dict[str, float], threshold: float = 0.6) -> Dict[str, str]:
        """Identify traits that are active above the threshold"""
        active_traits = {}
        
        for trait, score in trait_scores.items():
            if score >= threshold:
                if score >= 0.8:
                    active_traits[trait] = 'high'
                else:
                    active_traits[trait] = 'moderate'
            elif score <= 0.3:
                active_traits[trait] = 'low'
        
        return active_traits
    
    def _explain_trait_influence(self, trait_name: str, level: str, score: float, response_text: str) -> Dict[str, any]:
        """Explain how a specific trait is influencing the response"""
        if trait_name not in self.trait_impact_descriptions:
            return {'error': f'Unknown trait: {trait_name}'}
        
        trait_info = self.trait_impact_descriptions[trait_name].get(level, {})
        
        # Analyze actual response for trait evidence
        evidence = self._find_trait_evidence_in_response(trait_name, level, response_text)
        
        return {
            'trait': trait_name,
            'level': level,
            'score': score,
            'communication_style': trait_info.get('communication_style', 'Unknown'),
            'decision_factors': trait_info.get('decision_factors', 'Unknown'),
            'response_structure': trait_info.get('response_structure', 'Unknown'),
            'language_patterns': trait_info.get('language_patterns', 'Unknown'),
            'examples': trait_info.get('examples', []),
            'evidence_in_response': evidence
        }
    
    def _find_trait_evidence_in_response(self, trait_name: str, level: str, response_text: str) -> List[str]:
        """Find concrete evidence of trait influence in the actual response"""
        evidence = []
        response_lower = response_text.lower()
        
        # Empathy evidence
        if trait_name == 'empathy' and level in ['high', 'moderate']:
            empathy_indicators = ['understand', 'feel', 'sense', 'care', 'support', 'validate', 'meaningful']
            found_indicators = [indicator for indicator in empathy_indicators if indicator in response_lower]
            if found_indicators:
                evidence.append(f"Emotional language: {', '.join(found_indicators)}")
        
        # Analytical evidence
        if trait_name == 'analyticalness' and level in ['high', 'moderate']:
            analytical_indicators = ['systematically', 'analyze', 'methodically', 'structure', 'logical', 'evidence']
            found_indicators = [indicator for indicator in analytical_indicators if indicator in response_lower]
            if found_indicators:
                evidence.append(f"Analytical language: {', '.join(found_indicators)}")
            
            if any(pattern in response_text for pattern in ['1)', '2)', '3)', '•', '-']):
                evidence.append("Structured formatting with lists/bullets")
        
        # Creativity evidence
        if trait_name == 'creativity' and level in ['high', 'moderate']:
            creative_indicators = ['creative', 'innovative', 'imagine', 'possibilities', 'fascinating']
            found_indicators = [indicator for indicator in creative_indicators if indicator in response_lower]
            if found_indicators:
                evidence.append(f"Creative language: {', '.join(found_indicators)}")
        
        # Humor evidence
        if trait_name == 'humor' and level in ['high', 'moderate']:
            humor_indicators = ['ha!', 'ooh', '*', 'excited', 'fun', 'entertaining']
            found_indicators = [indicator for indicator in humor_indicators if indicator in response_lower or indicator in response_text]
            if found_indicators:
                evidence.append(f"Playful elements: {', '.join(found_indicators)}")
        
        return evidence if evidence else ['Trait influence present in overall tone and structure']
    
    def _generate_decision_rationale(self, active_traits: Dict[str, str], trait_scores: Dict[str, float]) -> str:
        """Generate explanation of why the AI made specific decisions"""
        if not active_traits:
            return "Response generated with balanced, neutral approach - no dominant personality traits active."
        
        rationale_parts = []
        dominant_trait = max(active_traits.items(), key=lambda x: trait_scores.get(x[0], 0))
        
        rationale_parts.append(f"Primary influence: {dominant_trait[0]} ({trait_scores[dominant_trait[0]]:.1f}) - {dominant_trait[1]} level")
        
        if len(active_traits) > 1:
            other_traits = [f"{trait} ({trait_scores[trait]:.1f})" for trait in active_traits.keys() if trait != dominant_trait[0]]
            rationale_parts.append(f"Secondary influences: {', '.join(other_traits)}")
        
        return ". ".join(rationale_parts)
    
    def _explain_response_style(self, active_traits: Dict[str, str], response_text: str) -> str:
        """Explain how traits shaped the response style"""
        if not active_traits:
            return "Neutral, balanced communication style"
        
        style_elements = []
        
        for trait, level in active_traits.items():
            if trait in self.trait_impact_descriptions:
                style_desc = self.trait_impact_descriptions[trait][level].get('communication_style', '')
                if style_desc:
                    style_elements.append(f"{trait}: {style_desc}")
        
        return "; ".join(style_elements) if style_elements else "Mixed personality influences"
    
    def _analyze_trait_interactions(self, active_traits: Dict[str, str], trait_scores: Dict[str, float]) -> List[str]:
        """Analyze how multiple traits interact with each other"""
        interactions = []
        
        trait_pairs = [
            ('empathy', 'analyticalness', 'Balancing emotional understanding with logical analysis'),
            ('creativity', 'analyticalness', 'Combining innovative thinking with systematic approach'),
            ('humor', 'empathy', 'Using playful engagement while maintaining emotional support'),
            ('curiosity', 'analyticalness', 'Systematic exploration and questioning')
        ]
        
        for trait1, trait2, description in trait_pairs:
            if trait1 in active_traits and trait2 in active_traits:
                score1, score2 = trait_scores.get(trait1, 0), trait_scores.get(trait2, 0)
                interactions.append(f"{description} ({trait1}: {score1:.1f}, {trait2}: {score2:.1f})")
        
        return interactions
    
    def _calculate_trait_changes(self, old_scores: Dict[str, float], new_scores: Dict[str, float]) -> Dict[str, Dict]:
        """Calculate and categorize trait changes"""
        changes = {}
        
        for trait in set(list(old_scores.keys()) + list(new_scores.keys())):
            old_val = old_scores.get(trait, 0.5)
            new_val = new_scores.get(trait, 0.5)
            change = new_val - old_val
            
            if abs(change) >= 0.1:  # Only significant changes
                changes[trait] = {
                    'old_value': old_val,
                    'new_value': new_val,
                    'change': change,
                    'direction': 'increased' if change > 0 else 'decreased',
                    'magnitude': 'major' if abs(change) >= 0.3 else 'moderate' if abs(change) >= 0.2 else 'minor'
                }
        
        return changes
    
    def _explain_change_rationale(self, changes: Dict, trigger: str, user_input: str) -> str:
        """Explain why trait changes occurred"""
        if not changes:
            return "No significant trait changes detected."
        
        explanations = []
        for trait, change_info in changes.items():
            direction = change_info['direction']
            magnitude = change_info['magnitude']
            explanations.append(f"{trait} {direction} ({magnitude}) due to: {trigger}")
        
        return f"Trait adaptations triggered by user input '{user_input}': " + "; ".join(explanations)
    
    def _predict_impact_changes(self, changes: Dict) -> List[str]:
        """Predict how trait changes will impact future responses"""
        predictions = []
        
        for trait, change_info in changes.items():
            direction = change_info['direction']
            new_level = 'high' if change_info['new_value'] >= 0.7 else 'moderate' if change_info['new_value'] >= 0.5 else 'low'
            
            if trait in self.trait_impact_descriptions and new_level in self.trait_impact_descriptions[trait]:
                impact_desc = self.trait_impact_descriptions[trait][new_level]['communication_style']
                predictions.append(f"{trait.capitalize()} {direction} → {impact_desc}")
        
        return predictions
    
    def _generate_adaptation_summary(self, old_scores: Dict[str, float], new_scores: Dict[str, float]) -> str:
        """Generate a summary of personality adaptation"""
        changes = self._calculate_trait_changes(old_scores, new_scores)
        
        if not changes:
            return "Personality remained stable - no significant adaptations needed."
        
        major_changes = [trait for trait, info in changes.items() if info['magnitude'] == 'major']
        
        if major_changes:
            return f"Major personality adaptation: {', '.join(major_changes)} significantly adjusted to better match conversation context."
        else:
            return f"Subtle personality refinement: {len(changes)} traits adjusted for improved response alignment."
    
    def get_trait_impact_summary(self, trait_scores: Dict[str, float]) -> Dict[str, any]:
        """Get a comprehensive summary of current trait impacts"""
        active_traits = self._identify_active_traits(trait_scores)
        
        return {
            'current_personality_state': {
                trait: {'score': score, 'level': active_traits.get(trait, 'neutral')}
                for trait, score in trait_scores.items()
            },
            'dominant_traits': [trait for trait, level in active_traits.items() if level == 'high'],
            'personality_summary': self._generate_personality_summary(trait_scores, active_traits),
            'communication_style_prediction': self._predict_communication_style(active_traits)
        }
    
    def _generate_personality_summary(self, trait_scores: Dict[str, float], active_traits: Dict[str, str]) -> str:
        """Generate a human-readable personality summary"""
        if not active_traits:
            return "Balanced, neutral personality with no dominant traits"
        
        high_traits = [trait for trait, level in active_traits.items() if level == 'high']
        moderate_traits = [trait for trait, level in active_traits.items() if level == 'moderate']
        
        summary_parts = []
        if high_traits:
            summary_parts.append(f"Highly {', '.join(high_traits)}")
        if moderate_traits:
            summary_parts.append(f"Moderately {', '.join(moderate_traits)}")
        
        return " and ".join(summary_parts) + " AI personality"
    
    def _predict_communication_style(self, active_traits: Dict[str, str]) -> str:
        """Predict overall communication style based on active traits"""
        style_elements = []
        
        for trait, level in active_traits.items():
            if trait in self.trait_impact_descriptions and level in self.trait_impact_descriptions[trait]:
                style_desc = self.trait_impact_descriptions[trait][level]['communication_style']
                style_elements.append(style_desc)
        
        return "; ".join(style_elements) if style_elements else "Neutral, balanced communication"