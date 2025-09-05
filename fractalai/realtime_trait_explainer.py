"""
Real-Time Trait Influence Explainer
Provides live explanations of how personality traits influence AI responses in real-time
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class RealtimeTraitExplainer:
    """
    Explains trait influence on responses in real-time with detailed breakdowns
    """
    
    def __init__(self, persistent_personality):
        self.persistent_personality = persistent_personality
        self.explanation_history = []
        
    def explain_trait_influence_realtime(self, session_id: str, user_input: str, 
                                       ai_response: str, trait_profile: Dict[str, float]) -> Dict:
        """
        Generate real-time explanation of how traits influenced the response
        """
        try:
            # Analyze trait influence on the response
            influence_analysis = self._analyze_trait_influence(user_input, ai_response, trait_profile)
            
            # Generate detailed explanations
            detailed_explanations = self._generate_detailed_explanations(influence_analysis, trait_profile)
            
            # Calculate influence strength
            influence_strength = self._calculate_influence_strength(influence_analysis)
            
            # Generate user-friendly summary
            summary = self._generate_influence_summary(influence_analysis, trait_profile)
            
            # Create visual representation data
            visual_data = self._create_visual_representation(trait_profile, influence_analysis)
            
            # Record the explanation
            explanation_record = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_input': user_input,
                'ai_response': ai_response,
                'trait_profile': trait_profile,
                'influence_analysis': influence_analysis,
                'influence_strength': influence_strength,
                'type': 'realtime_explanation'
            }
            self.explanation_history.append(explanation_record)
            
            return {
                'success': True,
                'trait_profile': trait_profile,
                'influence_analysis': influence_analysis,
                'detailed_explanations': detailed_explanations,
                'influence_strength': influence_strength,
                'summary': summary,
                'visual_data': visual_data,
                'timestamp': explanation_record['timestamp'],
                'explanation_confidence': self._calculate_explanation_confidence(influence_analysis)
            }
            
        except Exception as e:
            logging.error(f"Error explaining trait influence: {e}")
            return {'error': f'Failed to explain trait influence: {str(e)}'}
    
    def get_trait_impact_breakdown(self, session_id: str, specific_traits: List[str] = None) -> Dict:
        """
        Get detailed breakdown of how specific traits impact responses
        """
        try:
            current_personality = self.persistent_personality.get_personality(session_id)
            
            if specific_traits:
                traits_to_analyze = {trait: current_personality.get(trait, 0.5) 
                                   for trait in specific_traits if trait in current_personality}
            else:
                traits_to_analyze = current_personality
            
            impact_breakdown = {}
            
            for trait, value in traits_to_analyze.items():
                impact_breakdown[trait] = {
                    'current_value': value,
                    'impact_level': self._classify_impact_level(value),
                    'behavioral_effects': self._get_behavioral_effects(trait, value),
                    'communication_changes': self._get_communication_changes(trait, value),
                    'interaction_examples': self._get_interaction_examples(trait, value),
                    'optimization_suggestions': self._get_optimization_suggestions(trait, value)
                }
            
            return {
                'session_id': session_id,
                'analyzed_traits': list(traits_to_analyze.keys()),
                'impact_breakdown': impact_breakdown,
                'dominant_influences': self._identify_dominant_influences(traits_to_analyze),
                'interaction_prediction': self._predict_interaction_style(traits_to_analyze)
            }
            
        except Exception as e:
            logging.error(f"Error getting trait impact breakdown: {e}")
            return {'error': f'Failed to get impact breakdown: {str(e)}'}
    
    def explain_response_decision_process(self, user_input: str, ai_response: str, 
                                        trait_profile: Dict[str, float]) -> Dict:
        """
        Explain the decision process behind response generation
        """
        try:
            # Analyze input processing
            input_analysis = self._analyze_input_processing(user_input, trait_profile)
            
            # Explain response formation
            response_formation = self._explain_response_formation(ai_response, trait_profile)
            
            # Identify decision points
            decision_points = self._identify_decision_points(user_input, ai_response, trait_profile)
            
            # Generate step-by-step explanation
            step_by_step = self._generate_step_by_step_explanation(
                input_analysis, response_formation, decision_points
            )
            
            return {
                'input_analysis': input_analysis,
                'response_formation': response_formation,
                'decision_points': decision_points,
                'step_by_step_explanation': step_by_step,
                'trait_decision_weights': self._calculate_trait_decision_weights(trait_profile),
                'alternative_responses': self._suggest_alternative_responses(user_input, trait_profile)
            }
            
        except Exception as e:
            logging.error(f"Error explaining decision process: {e}")
            return {'error': f'Failed to explain decision process: {str(e)}'}
    
    def get_live_trait_monitoring(self, session_id: str) -> Dict:
        """
        Get live monitoring data for trait influence during conversations
        """
        try:
            current_personality = self.persistent_personality.get_personality(session_id)
            
            # Calculate trait activation levels
            activation_levels = self._calculate_trait_activation(current_personality)
            
            # Get influence patterns
            influence_patterns = self._get_current_influence_patterns(session_id)
            
            # Generate real-time insights
            realtime_insights = self._generate_realtime_insights(current_personality, influence_patterns)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_trait_stability(session_id)
            
            return {
                'session_id': session_id,
                'current_personality': current_personality,
                'trait_activation_levels': activation_levels,
                'influence_patterns': influence_patterns,
                'realtime_insights': realtime_insights,
                'stability_metrics': stability_metrics,
                'monitoring_timestamp': datetime.now().isoformat(),
                'next_update_in': 30  # seconds
            }
            
        except Exception as e:
            logging.error(f"Error getting live trait monitoring: {e}")
            return {'error': f'Failed to get live monitoring: {str(e)}'}
    
    def explain_trait_conflicts_and_resolutions(self, trait_profile: Dict[str, float]) -> Dict:
        """
        Explain any conflicts between traits and how they are resolved
        """
        try:
            # Identify potential conflicts
            conflicts = self._identify_trait_conflicts(trait_profile)
            
            # Explain resolution strategies
            resolutions = self._explain_conflict_resolutions(conflicts, trait_profile)
            
            # Calculate harmony score
            harmony_score = self._calculate_trait_harmony(trait_profile, conflicts)
            
            # Generate optimization suggestions
            optimization = self._suggest_trait_optimization(conflicts, trait_profile)
            
            return {
                'trait_profile': trait_profile,
                'identified_conflicts': conflicts,
                'resolution_strategies': resolutions,
                'trait_harmony_score': harmony_score,
                'optimization_suggestions': optimization,
                'conflict_impact_on_responses': self._analyze_conflict_impact(conflicts)
            }
            
        except Exception as e:
            logging.error(f"Error explaining trait conflicts: {e}")
            return {'error': f'Failed to explain conflicts: {str(e)}'}
    
    def get_explanation_history(self, session_id: str = None, limit: int = 20) -> List[Dict]:
        """
        Get history of trait influence explanations
        """
        history = self.explanation_history
        
        if session_id:
            history = [record for record in history if record.get('session_id') == session_id]
        
        # Sort by timestamp and limit
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return history[:limit]
    
    def _analyze_trait_influence(self, user_input: str, ai_response: str, 
                                trait_profile: Dict[str, float]) -> Dict:
        """
        Analyze how traits influenced the response
        """
        influence_analysis = {}
        
        # Analyze each trait's influence
        for trait, value in trait_profile.items():
            influence = self._calculate_single_trait_influence(trait, value, user_input, ai_response)
            if influence['strength'] > 0.1:  # Only include significant influences
                influence_analysis[trait] = influence
        
        return influence_analysis
    
    def _calculate_single_trait_influence(self, trait: str, value: float, 
                                        user_input: str, ai_response: str) -> Dict:
        """
        Calculate how a single trait influenced the response
        """
        influence_indicators = {
            'empathy': {
                'keywords': ['understand', 'feel', 'sorry', 'care', 'support', 'help'],
                'patterns': ['emotional validation', 'supportive language', 'concern expression']
            },
            'humor': {
                'keywords': ['haha', 'funny', 'joke', '*', 'playful'],
                'patterns': ['wordplay', 'light tone', 'casual expression']
            },
            'analyticalness': {
                'keywords': ['analysis', 'process', 'data', 'method', 'systematic', 'logical'],
                'patterns': ['structured response', 'step-by-step', 'factual approach']
            },
            'creativity': {
                'keywords': ['imagine', 'create', 'innovative', 'unique', 'artistic'],
                'patterns': ['metaphors', 'novel ideas', 'creative solutions']
            },
            'curiosity': {
                'keywords': ['question', 'wonder', 'explore', 'investigate', 'discover'],
                'patterns': ['asking questions', 'exploring topics', 'seeking information']
            },
            'supportiveness': {
                'keywords': ['help', 'assist', 'support', 'encourage', 'guide'],
                'patterns': ['offering help', 'encouraging tone', 'guidance provision']
            },
            'assertiveness': {
                'keywords': ['confident', 'clear', 'direct', 'certain', 'definite'],
                'patterns': ['direct statements', 'confident tone', 'clear positions']
            }
        }
        
        indicators = influence_indicators.get(trait, {'keywords': [], 'patterns': []})
        
        # Count keyword matches in response
        keyword_matches = sum(1 for keyword in indicators['keywords'] 
                            if keyword in ai_response.lower())
        
        # Calculate influence strength based on trait value and evidence
        base_strength = value if value > 0.5 else 0
        evidence_boost = min(keyword_matches * 0.1, 0.3)
        influence_strength = min(base_strength + evidence_boost, 1.0)
        
        return {
            'strength': influence_strength,
            'evidence_keywords': [keyword for keyword in indicators['keywords'] 
                                if keyword in ai_response.lower()],
            'detected_patterns': indicators['patterns'] if keyword_matches > 0 else [],
            'value_contribution': value,
            'response_alignment': self._calculate_response_alignment(trait, value, ai_response)
        }
    
    def _generate_detailed_explanations(self, influence_analysis: Dict, 
                                      trait_profile: Dict[str, float]) -> Dict:
        """
        Generate detailed explanations for each trait influence
        """
        explanations = {}
        
        for trait, influence_data in influence_analysis.items():
            strength = influence_data['strength']
            value = trait_profile[trait]
            
            explanation = self._create_trait_explanation(trait, value, strength, influence_data)
            explanations[trait] = explanation
        
        return explanations
    
    def _create_trait_explanation(self, trait: str, value: float, strength: float, 
                                influence_data: Dict) -> str:
        """
        Create detailed explanation for a specific trait's influence
        """
        trait_descriptions = {
            'empathy': f"Empathy level of {value:.1f} made the response more understanding and emotionally aware",
            'humor': f"Humor level of {value:.1f} added playfulness and light-hearted elements",
            'analyticalness': f"Analytical thinking at {value:.1f} structured the response logically",
            'creativity': f"Creativity level of {value:.1f} brought innovative and imaginative elements",
            'curiosity': f"Curiosity at {value:.1f} encouraged exploration and questioning",
            'supportiveness': f"Supportiveness level of {value:.1f} made the response more helpful and encouraging",
            'assertiveness': f"Assertiveness at {value:.1f} created more confident and direct communication"
        }
        
        base_explanation = trait_descriptions.get(trait, f"{trait} at level {value:.1f} influenced the response")
        
        # Add evidence details
        evidence = influence_data.get('evidence_keywords', [])
        if evidence:
            evidence_text = f" Evidence: {', '.join(evidence)}"
            base_explanation += evidence_text
        
        # Add strength qualifier
        if strength > 0.7:
            base_explanation += " (Strong influence)"
        elif strength > 0.4:
            base_explanation += " (Moderate influence)"
        else:
            base_explanation += " (Subtle influence)"
        
        return base_explanation
    
    def _calculate_influence_strength(self, influence_analysis: Dict) -> Dict:
        """
        Calculate overall influence strength metrics
        """
        if not influence_analysis:
            return {'total': 0, 'strongest_trait': None, 'average': 0}
        
        strengths = [data['strength'] for data in influence_analysis.values()]
        total_strength = sum(strengths)
        average_strength = total_strength / len(strengths)
        strongest_trait = max(influence_analysis.keys(), 
                            key=lambda t: influence_analysis[t]['strength'])
        
        return {
            'total': total_strength,
            'average': average_strength,
            'strongest_trait': strongest_trait,
            'strongest_value': influence_analysis[strongest_trait]['strength'],
            'active_traits': len(influence_analysis)
        }
    
    def _generate_influence_summary(self, influence_analysis: Dict, 
                                  trait_profile: Dict[str, float]) -> str:
        """
        Generate user-friendly summary of trait influence
        """
        if not influence_analysis:
            return "Balanced response with no dominant trait influence detected."
        
        # Find dominant influences
        dominant_traits = sorted(influence_analysis.items(), 
                               key=lambda x: x[1]['strength'], reverse=True)[:2]
        
        if len(dominant_traits) == 1:
            trait, data = dominant_traits[0]
            return f"Response primarily influenced by {trait} (level {trait_profile[trait]:.1f}), creating a {self._get_trait_style_description(trait)} communication style."
        else:
            trait1, data1 = dominant_traits[0]
            trait2, data2 = dominant_traits[1]
            return f"Response shaped by a combination of {trait1} ({trait_profile[trait1]:.1f}) and {trait2} ({trait_profile[trait2]:.1f}), resulting in a {self._get_combined_style_description(trait1, trait2)} approach."
    
    def _get_trait_style_description(self, trait: str) -> str:
        """
        Get style description for a trait
        """
        descriptions = {
            'empathy': 'caring and understanding',
            'humor': 'playful and entertaining',
            'analyticalness': 'logical and structured',
            'creativity': 'imaginative and innovative',
            'curiosity': 'inquisitive and exploratory',
            'supportiveness': 'helpful and encouraging',
            'assertiveness': 'confident and direct'
        }
        return descriptions.get(trait, 'distinctive')
    
    def _get_combined_style_description(self, trait1: str, trait2: str) -> str:
        """
        Get description for combined trait influences
        """
        combinations = {
            ('empathy', 'humor'): 'warm and lighthearted',
            ('empathy', 'analyticalness'): 'thoughtful and systematic',
            ('humor', 'creativity'): 'playful and imaginative',
            ('analyticalness', 'assertiveness'): 'confident and methodical',
            ('creativity', 'curiosity'): 'innovative and exploratory',
            ('supportiveness', 'empathy'): 'deeply caring and helpful'
        }
        
        # Try both orders
        combo_key = (trait1, trait2)
        reverse_key = (trait2, trait1)
        
        return combinations.get(combo_key, combinations.get(reverse_key, 'balanced and multifaceted'))
    
    def _create_visual_representation(self, trait_profile: Dict[str, float], 
                                    influence_analysis: Dict) -> Dict:
        """
        Create data for visual representation of trait influence
        """
        visual_data = {
            'trait_levels': [],
            'influence_strength': [],
            'trait_colors': {},
            'influence_indicators': []
        }
        
        # Color scheme for traits
        trait_colors = {
            'empathy': '#ff6b6b',      # Red
            'humor': '#ffd93d',        # Yellow
            'analyticalness': '#4ecdc4', # Teal
            'creativity': '#a8e6cf',   # Green
            'curiosity': '#ff8b94',    # Pink
            'supportiveness': '#88d8b0', # Light green
            'assertiveness': '#b4a7d6'  # Purple
        }
        
        for trait, value in trait_profile.items():
            visual_data['trait_levels'].append({
                'trait': trait,
                'value': value,
                'display_name': trait.capitalize(),
                'color': trait_colors.get(trait, '#cccccc')
            })
            
            # Add influence strength if trait was influential
            if trait in influence_analysis:
                influence_strength = influence_analysis[trait]['strength']
                visual_data['influence_strength'].append({
                    'trait': trait,
                    'strength': influence_strength,
                    'color': trait_colors.get(trait, '#cccccc')
                })
        
        visual_data['trait_colors'] = trait_colors
        
        return visual_data
    
    def _calculate_explanation_confidence(self, influence_analysis: Dict) -> float:
        """
        Calculate confidence level for the explanation
        """
        if not influence_analysis:
            return 0.3
        
        # Base confidence on evidence strength
        evidence_scores = []
        for trait_data in influence_analysis.values():
            evidence_keywords = len(trait_data.get('evidence_keywords', []))
            evidence_scores.append(min(evidence_keywords * 0.2, 1.0))
        
        if evidence_scores:
            avg_evidence = sum(evidence_scores) / len(evidence_scores)
            return min(0.5 + avg_evidence * 0.4, 0.95)
        else:
            return 0.4
    
    def _classify_impact_level(self, value: float) -> str:
        """
        Classify the impact level of a trait value
        """
        if value >= 0.8:
            return 'Very High'
        elif value >= 0.6:
            return 'High'
        elif value >= 0.4:
            return 'Moderate'
        elif value >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def _get_behavioral_effects(self, trait: str, value: float) -> List[str]:
        """
        Get behavioral effects for a trait at a specific value
        """
        effects_map = {
            'empathy': [
                "Shows understanding of emotions",
                "Validates feelings and concerns",
                "Offers emotional support",
                "Uses caring language"
            ],
            'humor': [
                "Includes jokes and wordplay",
                "Uses casual, playful language",
                "Lightens serious topics appropriately",
                "Adds entertainment value"
            ],
            'analyticalness': [
                "Provides structured responses",
                "Uses logical reasoning",
                "Breaks down complex problems",
                "Focuses on facts and data"
            ],
            'creativity': [
                "Offers innovative solutions",
                "Uses metaphors and analogies",
                "Thinks outside the box",
                "Suggests creative approaches"
            ],
            'curiosity': [
                "Asks follow-up questions",
                "Explores topics deeply",
                "Shows interest in learning",
                "Investigates different angles"
            ],
            'supportiveness': [
                "Offers help and assistance",
                "Encourages and motivates",
                "Provides practical guidance",
                "Shows belief in user capabilities"
            ],
            'assertiveness': [
                "States opinions clearly",
                "Takes confident positions",
                "Provides direct feedback",
                "Leads conversations when appropriate"
            ]
        }
        
        base_effects = effects_map.get(trait, [])
        
        # Adjust effects based on value
        if value < 0.3:
            return [f"Minimal {effect.lower()}" for effect in base_effects[:2]]
        elif value < 0.7:
            return base_effects[:3]
        else:
            return base_effects
    
    def _get_communication_changes(self, trait: str, value: float) -> List[str]:
        """
        Get communication changes for a trait at a specific value
        """
        changes_map = {
            'empathy': ["Warmer tone", "More personal responses", "Emotional validation"],
            'humor': ["Playful language", "Casual expressions", "Light-hearted tone"],
            'analyticalness': ["Structured format", "Logical flow", "Detailed explanations"],
            'creativity': ["Imaginative examples", "Novel perspectives", "Creative metaphors"],
            'curiosity': ["Probing questions", "Exploratory dialogue", "Learning focus"],
            'supportiveness': ["Encouraging language", "Helpful suggestions", "Positive reinforcement"],
            'assertiveness': ["Direct statements", "Confident tone", "Clear positions"]
        }
        
        return changes_map.get(trait, ["Characteristic communication style"])
    
    def _get_interaction_examples(self, trait: str, value: float) -> List[str]:
        """
        Get example interactions for a trait at a specific value
        """
        examples_map = {
            'empathy': [
                "I understand how challenging that must be for you",
                "Your feelings about this situation are completely valid",
                "I can sense this is important to you"
            ],
            'humor': [
                "Well, that's one way to keep things interesting! 😄",
                "Looks like we're in for an adventure!",
                "Time to put on our problem-solving hats!"
            ],
            'analyticalness': [
                "Let me break this down into key components:",
                "Based on the data, we can conclude that...",
                "Here's a systematic approach to this problem:"
            ],
            'creativity': [
                "What if we approached this from a completely different angle?",
                "Imagine this scenario as a blank canvas...",
                "Let's think outside the conventional framework"
            ],
            'curiosity': [
                "That's fascinating! Can you tell me more about...?",
                "I'm curious to understand the reasoning behind...",
                "What other factors might be influencing this?"
            ],
            'supportiveness': [
                "I'm here to help you work through this",
                "You're on the right track with this thinking",
                "Let me guide you toward a solution"
            ],
            'assertiveness': [
                "I believe the best approach is clearly...",
                "Based on my analysis, I recommend...",
                "The evidence strongly suggests that..."
            ]
        }
        
        base_examples = examples_map.get(trait, ["Trait-influenced response"])
        
        # Return appropriate number based on value
        if value < 0.3:
            return base_examples[:1]
        elif value < 0.7:
            return base_examples[:2]
        else:
            return base_examples
    
    def _get_optimization_suggestions(self, trait: str, value: float) -> List[str]:
        """
        Get optimization suggestions for a trait
        """
        if value > 0.8:
            return [f"Consider balancing {trait} with other traits to avoid dominance"]
        elif value < 0.2:
            return [f"Consider increasing {trait} for more balanced interactions"]
        else:
            return [f"{trait.capitalize()} level is well-balanced"]
    
    def _identify_dominant_influences(self, trait_profile: Dict[str, float]) -> List[Dict]:
        """
        Identify the most dominant trait influences
        """
        sorted_traits = sorted(trait_profile.items(), key=lambda x: x[1], reverse=True)
        
        dominant = []
        for trait, value in sorted_traits[:3]:
            if value > 0.6:
                dominant.append({
                    'trait': trait,
                    'value': value,
                    'dominance_level': self._classify_impact_level(value),
                    'influence_description': self._get_trait_style_description(trait)
                })
        
        return dominant
    
    def _predict_interaction_style(self, trait_profile: Dict[str, float]) -> str:
        """
        Predict overall interaction style based on trait profile
        """
        dominant_traits = sorted(trait_profile.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if not dominant_traits:
            return "Balanced and neutral interaction style"
        
        primary_trait, primary_value = dominant_traits[0]
        
        if len(dominant_traits) > 1:
            secondary_trait, secondary_value = dominant_traits[1]
            if abs(primary_value - secondary_value) < 0.2:
                return f"Interaction style will blend {primary_trait} and {secondary_trait} characteristics"
        
        style_predictions = {
            'empathy': "Warm, understanding, and emotionally supportive interactions",
            'humor': "Playful, entertaining, and lighthearted conversations",
            'analyticalness': "Structured, logical, and detail-oriented discussions",
            'creativity': "Imaginative, innovative, and outside-the-box thinking",
            'curiosity': "Inquisitive, exploratory, and learning-focused exchanges",
            'supportiveness': "Helpful, encouraging, and solution-oriented assistance",
            'assertiveness': "Confident, direct, and decisive communication"
        }
        
        return style_predictions.get(primary_trait, "Distinctive interaction style")
    
    def _calculate_trait_activation(self, trait_profile: Dict[str, float]) -> Dict:
        """
        Calculate real-time trait activation levels
        """
        activation = {}
        
        for trait, value in trait_profile.items():
            # Calculate activation based on value and context
            if value > 0.7:
                activation_level = 'High'
                activation_score = value
            elif value > 0.5:
                activation_level = 'Moderate'
                activation_score = value * 0.8
            elif value > 0.3:
                activation_level = 'Low'
                activation_score = value * 0.6
            else:
                activation_level = 'Minimal'
                activation_score = value * 0.4
            
            activation[trait] = {
                'level': activation_level,
                'score': activation_score,
                'predicted_influence': activation_score > 0.5
            }
        
        return activation
    
    def _get_current_influence_patterns(self, session_id: str) -> Dict:
        """
        Get current influence patterns for a session
        """
        # Get recent explanations for this session
        recent_explanations = [
            record for record in self.explanation_history[-10:]
            if record.get('session_id') == session_id
        ]
        
        if not recent_explanations:
            return {'pattern': 'No established pattern yet'}
        
        # Analyze patterns
        trait_usage = {}
        for record in recent_explanations:
            influence_analysis = record.get('influence_analysis', {})
            for trait in influence_analysis:
                trait_usage[trait] = trait_usage.get(trait, 0) + 1
        
        if trait_usage:
            most_used = max(trait_usage, key=trait_usage.get)
            return {
                'most_active_trait': most_used,
                'usage_frequency': trait_usage[most_used] / len(recent_explanations),
                'trait_usage_distribution': trait_usage,
                'pattern_strength': max(trait_usage.values()) / len(recent_explanations)
            }
        
        return {'pattern': 'Balanced usage across traits'}
    
    def _generate_realtime_insights(self, trait_profile: Dict[str, float], 
                                  influence_patterns: Dict) -> List[str]:
        """
        Generate real-time insights about trait usage
        """
        insights = []
        
        # Analyze current state
        dominant_trait = max(trait_profile, key=trait_profile.get)
        dominant_value = trait_profile[dominant_trait]
        
        if dominant_value > 0.8:
            insights.append(f"{dominant_trait.capitalize()} is currently very influential in responses")
        
        # Pattern-based insights
        if 'most_active_trait' in influence_patterns:
            most_active = influence_patterns['most_active_trait']
            frequency = influence_patterns.get('usage_frequency', 0)
            if frequency > 0.7:
                insights.append(f"{most_active.capitalize()} has been consistently active in recent interactions")
        
        # Balance insights
        trait_values = list(trait_profile.values())
        if max(trait_values) - min(trait_values) < 0.3:
            insights.append("Personality traits are well-balanced, creating versatile responses")
        
        return insights
    
    def _calculate_trait_stability(self, session_id: str) -> Dict:
        """
        Calculate trait stability metrics
        """
        # This would integrate with the personality evolution tracker
        # For now, return basic stability info
        return {
            'stability_score': 0.8,
            'most_stable_trait': 'empathy',
            'most_variable_trait': 'humor',
            'change_velocity': 0.1
        }
    
    def _identify_trait_conflicts(self, trait_profile: Dict[str, float]) -> List[Dict]:
        """
        Identify potential conflicts between traits
        """
        conflicts = []
        
        # Define conflicting trait pairs
        conflict_pairs = [
            ('empathy', 'assertiveness'),
            ('humor', 'analyticalness'),
            ('creativity', 'analyticalness'),
            ('curiosity', 'assertiveness')
        ]
        
        for trait1, trait2 in conflict_pairs:
            if trait1 in trait_profile and trait2 in trait_profile:
                value1 = trait_profile[trait1]
                value2 = trait_profile[trait2]
                
                # High values in both conflicting traits
                if value1 > 0.7 and value2 > 0.7:
                    conflict_strength = min(value1, value2)
                    conflicts.append({
                        'traits': [trait1, trait2],
                        'values': [value1, value2],
                        'conflict_strength': conflict_strength,
                        'type': 'high_both',
                        'description': f"High {trait1} and {trait2} may create tension in responses"
                    })
        
        return conflicts
    
    def _explain_conflict_resolutions(self, conflicts: List[Dict], 
                                    trait_profile: Dict[str, float]) -> Dict:
        """
        Explain how trait conflicts are resolved
        """
        if not conflicts:
            return {'message': 'No significant trait conflicts detected'}
        
        resolutions = {}
        
        for conflict in conflicts:
            trait1, trait2 = conflict['traits']
            value1, value2 = conflict['values']
            
            # Determine resolution strategy
            if abs(value1 - value2) < 0.1:
                strategy = 'balanced_blend'
                explanation = f"AI blends {trait1} and {trait2} characteristics situationally"
            elif value1 > value2:
                strategy = 'primary_dominance'
                explanation = f"{trait1} takes precedence while {trait2} provides subtle influence"
            else:
                strategy = 'primary_dominance'
                explanation = f"{trait2} takes precedence while {trait1} provides subtle influence"
            
            resolutions[f"{trait1}_{trait2}"] = {
                'strategy': strategy,
                'explanation': explanation,
                'effectiveness': self._estimate_resolution_effectiveness(strategy, value1, value2)
            }
        
        return resolutions
    
    def _calculate_trait_harmony(self, trait_profile: Dict[str, float], 
                               conflicts: List[Dict]) -> float:
        """
        Calculate overall trait harmony score
        """
        if not conflicts:
            return 0.95
        
        # Start with base harmony
        harmony = 1.0
        
        # Reduce harmony based on conflicts
        for conflict in conflicts:
            conflict_strength = conflict['conflict_strength']
            harmony -= conflict_strength * 0.2
        
        # Ensure minimum harmony
        return max(harmony, 0.3)
    
    def _suggest_trait_optimization(self, conflicts: List[Dict], 
                                  trait_profile: Dict[str, float]) -> List[str]:
        """
        Suggest optimizations for trait conflicts
        """
        if not conflicts:
            return ["Trait profile is well-optimized with minimal conflicts"]
        
        suggestions = []
        
        for conflict in conflicts:
            trait1, trait2 = conflict['traits']
            suggestions.append(
                f"Consider slightly reducing either {trait1} or {trait2} to minimize conflict"
            )
        
        # Add general optimization suggestions
        trait_values = list(trait_profile.values())
        if max(trait_values) > 0.9:
            suggestions.append("Very high trait values may limit response flexibility")
        
        return suggestions
    
    def _analyze_conflict_impact(self, conflicts: List[Dict]) -> Dict:
        """
        Analyze impact of trait conflicts on responses
        """
        if not conflicts:
            return {'impact_level': 'None', 'description': 'No conflicts affecting responses'}
        
        total_conflict_strength = sum(conflict['conflict_strength'] for conflict in conflicts)
        avg_conflict = total_conflict_strength / len(conflicts)
        
        if avg_conflict > 0.8:
            impact_level = 'High'
            description = 'Conflicts may create inconsistent or contradictory response patterns'
        elif avg_conflict > 0.6:
            impact_level = 'Moderate'
            description = 'Some tension in responses, but generally manageable'
        else:
            impact_level = 'Low'
            description = 'Minor conflicts with minimal impact on response quality'
        
        return {
            'impact_level': impact_level,
            'description': description,
            'affected_traits': [trait for conflict in conflicts for trait in conflict['traits']]
        }
    
    def _estimate_resolution_effectiveness(self, strategy: str, value1: float, value2: float) -> float:
        """
        Estimate effectiveness of conflict resolution strategy
        """
        if strategy == 'balanced_blend':
            # More effective when values are closer
            difference = abs(value1 - value2)
            return max(0.9 - difference, 0.5)
        elif strategy == 'primary_dominance':
            # Effective when there's clear dominance
            difference = abs(value1 - value2)
            return min(0.5 + difference, 0.9)
        else:
            return 0.7
    
    def _calculate_response_alignment(self, trait: str, value: float, response: str) -> float:
        """
        Calculate how well the response aligns with the trait value
        """
        # This is a simplified alignment calculation
        # In practice, this would use more sophisticated NLP analysis
        
        trait_indicators = {
            'empathy': len([word for word in response.lower().split() 
                          if word in ['understand', 'feel', 'care', 'support']]),
            'humor': len([char for char in response if char in ['!', '😄', '😊']]),
            'analyticalness': len([word for word in response.lower().split() 
                                 if word in ['analysis', 'data', 'logical', 'systematic']])
        }
        
        indicator_count = trait_indicators.get(trait, 0)
        expected_indicators = value * 3  # Scale expected indicators by trait value
        
        if expected_indicators == 0:
            return 1.0 if indicator_count == 0 else 0.5
        
        alignment = min(indicator_count / expected_indicators, 1.0)
        return alignment
    
    def _analyze_input_processing(self, user_input: str, trait_profile: Dict[str, float]) -> Dict:
        """
        Analyze how traits influence input processing
        """
        analysis = {
            'input_classification': self._classify_input_type(user_input),
            'emotional_detection': self._detect_emotions_in_input(user_input),
            'trait_triggered_processing': {}
        }
        
        # Analyze which traits are triggered by the input
        for trait, value in trait_profile.items():
            trigger_strength = self._calculate_trait_trigger(trait, user_input, value)
            if trigger_strength > 0.3:
                analysis['trait_triggered_processing'][trait] = {
                    'trigger_strength': trigger_strength,
                    'processing_influence': self._describe_processing_influence(trait, user_input)
                }
        
        return analysis
    
    def _explain_response_formation(self, ai_response: str, trait_profile: Dict[str, float]) -> Dict:
        """
        Explain how traits influenced response formation
        """
        formation = {
            'response_structure': self._analyze_response_structure(ai_response),
            'tone_selection': self._analyze_tone_selection(ai_response, trait_profile),
            'content_choices': self._analyze_content_choices(ai_response, trait_profile),
            'trait_contributions': {}
        }
        
        # Analyze each trait's contribution to response formation
        for trait, value in trait_profile.items():
            if value > 0.4:
                contribution = self._analyze_trait_contribution(trait, ai_response, value)
                formation['trait_contributions'][trait] = contribution
        
        return formation
    
    def _identify_decision_points(self, user_input: str, ai_response: str, 
                                trait_profile: Dict[str, float]) -> List[Dict]:
        """
        Identify key decision points in response generation
        """
        decision_points = []
        
        # Tone decision
        if any(value > 0.7 for value in trait_profile.values()):
            dominant_trait = max(trait_profile, key=trait_profile.get)
            decision_points.append({
                'decision': 'tone_selection',
                'description': f"Chose {self._get_trait_style_description(dominant_trait)} tone",
                'influencing_trait': dominant_trait,
                'confidence': trait_profile[dominant_trait]
            })
        
        # Content depth decision
        if trait_profile.get('analyticalness', 0.5) > 0.6:
            decision_points.append({
                'decision': 'content_depth',
                'description': "Provided detailed, analytical response",
                'influencing_trait': 'analyticalness',
                'confidence': trait_profile['analyticalness']
            })
        
        # Emotional support decision
        if trait_profile.get('empathy', 0.5) > 0.6 and self._detect_emotions_in_input(user_input):
            decision_points.append({
                'decision': 'emotional_response',
                'description': "Included emotional validation and support",
                'influencing_trait': 'empathy',
                'confidence': trait_profile['empathy']
            })
        
        return decision_points
    
    def _generate_step_by_step_explanation(self, input_analysis: Dict, 
                                         response_formation: Dict, 
                                         decision_points: List[Dict]) -> List[str]:
        """
        Generate step-by-step explanation of the response process
        """
        steps = []
        
        # Step 1: Input analysis
        input_type = input_analysis.get('input_classification', 'general')
        steps.append(f"1. Analyzed input as {input_type} requiring appropriate response type")
        
        # Step 2: Trait activation
        triggered_traits = input_analysis.get('trait_triggered_processing', {})
        if triggered_traits:
            active_traits = list(triggered_traits.keys())
            steps.append(f"2. Activated traits: {', '.join(active_traits)} based on input content")
        
        # Step 3: Decision points
        for i, decision in enumerate(decision_points, 3):
            steps.append(f"{i}. {decision['description']} (influenced by {decision['influencing_trait']})")
        
        # Step 4: Response generation
        steps.append(f"{len(steps) + 1}. Generated response combining all active trait influences")
        
        return steps
    
    def _calculate_trait_decision_weights(self, trait_profile: Dict[str, float]) -> Dict:
        """
        Calculate decision weights for each trait
        """
        total_activation = sum(max(value - 0.5, 0) for value in trait_profile.values())
        
        if total_activation == 0:
            return {trait: 1.0 / len(trait_profile) for trait in trait_profile}
        
        weights = {}
        for trait, value in trait_profile.items():
            weight = max(value - 0.5, 0) / total_activation
            weights[trait] = weight
        
        return weights
    
    def _suggest_alternative_responses(self, user_input: str, trait_profile: Dict[str, float]) -> List[Dict]:
        """
        Suggest how response would differ with different trait configurations
        """
        alternatives = []
        
        # High empathy alternative
        if trait_profile.get('empathy', 0.5) < 0.8:
            alternatives.append({
                'configuration': 'High Empathy',
                'trait_changes': {'empathy': 0.9},
                'expected_difference': 'More emotionally supportive and understanding tone'
            })
        
        # High analytical alternative
        if trait_profile.get('analyticalness', 0.5) < 0.8:
            alternatives.append({
                'configuration': 'High Analytical',
                'trait_changes': {'analyticalness': 0.9},
                'expected_difference': 'More structured, detailed, and logical approach'
            })
        
        # High humor alternative
        if trait_profile.get('humor', 0.5) < 0.8:
            alternatives.append({
                'configuration': 'High Humor',
                'trait_changes': {'humor': 0.9},
                'expected_difference': 'More playful, entertaining, and lighthearted response'
            })
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _classify_input_type(self, user_input: str) -> str:
        """
        Classify the type of user input
        """
        input_lower = user_input.lower()
        
        if '?' in user_input:
            return 'question'
        elif any(word in input_lower for word in ['help', 'assist', 'support']):
            return 'help_request'
        elif any(word in input_lower for word in ['sad', 'frustrated', 'angry', 'upset']):
            return 'emotional_expression'
        elif any(word in input_lower for word in ['analyze', 'explain', 'understand']):
            return 'analysis_request'
        else:
            return 'general_conversation'
    
    def _detect_emotions_in_input(self, user_input: str) -> bool:
        """
        Detect if user input contains emotional content
        """
        emotional_words = ['sad', 'happy', 'angry', 'frustrated', 'excited', 'worried', 'anxious', 'stressed']
        return any(word in user_input.lower() for word in emotional_words)
    
    def _calculate_trait_trigger(self, trait: str, user_input: str, trait_value: float) -> float:
        """
        Calculate how strongly a trait is triggered by user input
        """
        input_lower = user_input.lower()
        
        trigger_words = {
            'empathy': ['feel', 'emotion', 'sad', 'upset', 'worried', 'support'],
            'humor': ['funny', 'joke', 'laugh', 'fun', 'entertain'],
            'analyticalness': ['analyze', 'explain', 'understand', 'logic', 'reason'],
            'creativity': ['create', 'imagine', 'innovative', 'design', 'art'],
            'curiosity': ['why', 'how', 'what', 'explore', 'discover', 'learn'],
            'supportiveness': ['help', 'assist', 'guide', 'support', 'advice'],
            'assertiveness': ['decide', 'recommend', 'opinion', 'suggest', 'should']
        }
        
        trigger_count = sum(1 for word in trigger_words.get(trait, []) if word in input_lower)
        base_trigger = min(trigger_count * 0.3, 1.0)
        
        # Scale by trait value
        return base_trigger * trait_value
    
    def _describe_processing_influence(self, trait: str, user_input: str) -> str:
        """
        Describe how a trait influences input processing
        """
        descriptions = {
            'empathy': 'Focuses on emotional content and underlying feelings',
            'humor': 'Looks for opportunities to add lightness and entertainment',
            'analyticalness': 'Breaks down the request into logical components',
            'creativity': 'Considers innovative and imaginative approaches',
            'curiosity': 'Identifies areas for deeper exploration and questioning',
            'supportiveness': 'Prioritizes helping and providing assistance',
            'assertiveness': 'Prepares to provide clear, direct guidance'
        }
        
        return descriptions.get(trait, f'Influences processing through {trait} perspective')
    
    def _analyze_response_structure(self, ai_response: str) -> Dict:
        """
        Analyze the structure of the AI response
        """
        sentences = len([s for s in ai_response.split('.') if s.strip()])
        paragraphs = len([p for p in ai_response.split('\n\n') if p.strip()])
        word_count = len(ai_response.split())
        
        return {
            'sentences': sentences,
            'paragraphs': paragraphs,
            'word_count': word_count,
            'structure_type': 'detailed' if word_count > 100 else 'concise' if word_count > 30 else 'brief'
        }
    
    def _analyze_tone_selection(self, ai_response: str, trait_profile: Dict[str, float]) -> str:
        """
        Analyze the tone selected for the response
        """
        if trait_profile.get('humor', 0) > 0.7:
            return 'playful and humorous'
        elif trait_profile.get('empathy', 0) > 0.7:
            return 'warm and understanding'
        elif trait_profile.get('analyticalness', 0) > 0.7:
            return 'professional and structured'
        elif trait_profile.get('assertiveness', 0) > 0.7:
            return 'confident and direct'
        else:
            return 'balanced and neutral'
    
    def _analyze_content_choices(self, ai_response: str, trait_profile: Dict[str, float]) -> List[str]:
        """
        Analyze content choices made in the response
        """
        choices = []
        
        if '?' in ai_response and trait_profile.get('curiosity', 0) > 0.6:
            choices.append('Included questions to encourage exploration')
        
        if any(word in ai_response.lower() for word in ['understand', 'feel']) and trait_profile.get('empathy', 0) > 0.6:
            choices.append('Added emotional validation')
        
        if any(word in ai_response.lower() for word in ['analysis', 'step', 'method']) and trait_profile.get('analyticalness', 0) > 0.6:
            choices.append('Provided structured analysis')
        
        return choices
    
    def _analyze_trait_contribution(self, trait: str, ai_response: str, trait_value: float) -> Dict:
        """
        Analyze a specific trait's contribution to response formation
        """
        contribution_strength = self._calculate_single_trait_influence(
            trait, trait_value, '', ai_response
        )['strength']
        
        contributions = {
            'empathy': 'emotional warmth and understanding',
            'humor': 'playfulness and entertainment value',
            'analyticalness': 'logical structure and detail',
            'creativity': 'imaginative elements and novel perspectives',
            'curiosity': 'exploratory questions and deeper investigation',
            'supportiveness': 'helpful guidance and encouragement',
            'assertiveness': 'confident direction and clear opinions'
        }
        
        return {
            'contribution_strength': contribution_strength,
            'contribution_type': contributions.get(trait, f'{trait} characteristics'),
            'evidence_in_response': self._find_trait_evidence(trait, ai_response)
        }
    
    def _find_trait_evidence(self, trait: str, ai_response: str) -> List[str]:
        """
        Find evidence of trait influence in the response
        """
        evidence_patterns = {
            'empathy': ['understand', 'feel', 'care', 'support', 'sorry'],
            'humor': ['!', 'haha', 'funny', '*', 'playful'],
            'analyticalness': ['analysis', 'step', 'logical', 'systematic', 'method'],
            'creativity': ['imagine', 'creative', 'innovative', 'unique'],
            'curiosity': ['?', 'explore', 'discover', 'investigate', 'wonder'],
            'supportiveness': ['help', 'assist', 'guide', 'encourage', 'support'],
            'assertiveness': ['recommend', 'should', 'clearly', 'definitely', 'confident']
        }
        
        patterns = evidence_patterns.get(trait, [])
        found_evidence = [pattern for pattern in patterns if pattern in ai_response.lower()]
        
        return found_evidence[:3]  # Return top 3 pieces of evidence