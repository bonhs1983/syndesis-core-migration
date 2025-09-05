"""
Interactive Personality Trait System
Allows real-time adjustment and training of personality traits through user interaction
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class InteractivePersonalityTrainer:
    """
    Enables real-time personality trait adjustment based on user feedback and requests
    """
    
    def __init__(self, persistent_personality):
        self.persistent_personality = persistent_personality
        self.training_history = []
        
    def adjust_trait_interactive(self, session_id: str, trait_name: str, adjustment: float, reason: str = "") -> Dict:
        """
        Interactively adjust a specific personality trait with real-time feedback
        
        Args:
            session_id: Current session identifier
            trait_name: Name of trait to adjust (empathy, humor, etc.)
            adjustment: Amount to adjust (-1.0 to +1.0)
            reason: Reason for the adjustment
        """
        try:
            # Get current personality
            current_personality = self.persistent_personality.get_personality(session_id)
            
            if trait_name not in current_personality:
                return {
                    'error': f'Unknown trait: {trait_name}',
                    'available_traits': list(current_personality.keys())
                }
            
            # Calculate new value
            old_value = current_personality[trait_name]
            new_value = max(0.0, min(1.0, old_value + adjustment))
            
            # Update the trait
            current_personality[trait_name] = new_value
            self.persistent_personality.update_personality(session_id, current_personality)
            
            # Record the training interaction
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'trait': trait_name,
                'old_value': old_value,
                'new_value': new_value,
                'adjustment': adjustment,
                'reason': reason,
                'type': 'interactive_adjustment'
            }
            self.training_history.append(training_record)
            
            # Generate explanation of the change
            impact_explanation = self._explain_trait_impact(trait_name, old_value, new_value)
            
            return {
                'success': True,
                'trait': trait_name,
                'old_value': old_value,
                'new_value': new_value,
                'adjustment': adjustment,
                'reason': reason,
                'impact_explanation': impact_explanation,
                'updated_personality': current_personality,
                'timestamp': training_record['timestamp']
            }
            
        except Exception as e:
            logging.error(f"Error in interactive trait adjustment: {e}")
            return {'error': f'Failed to adjust trait: {str(e)}'}
    
    def bulk_trait_adjustment(self, session_id: str, trait_adjustments: Dict[str, float], reason: str = "") -> Dict:
        """
        Adjust multiple traits simultaneously with batch processing
        """
        try:
            current_personality = self.persistent_personality.get_personality(session_id)
            adjustments_made = {}
            errors = []
            
            for trait_name, adjustment in trait_adjustments.items():
                if trait_name not in current_personality:
                    errors.append(f'Unknown trait: {trait_name}')
                    continue
                
                old_value = current_personality[trait_name]
                new_value = max(0.0, min(1.0, old_value + adjustment))
                current_personality[trait_name] = new_value
                
                adjustments_made[trait_name] = {
                    'old_value': old_value,
                    'new_value': new_value,
                    'adjustment': adjustment
                }
                
                # Record each adjustment
                training_record = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'trait': trait_name,
                    'old_value': old_value,
                    'new_value': new_value,
                    'adjustment': adjustment,
                    'reason': reason,
                    'type': 'bulk_adjustment'
                }
                self.training_history.append(training_record)
            
            # Update personality
            self.persistent_personality.update_personality(session_id, current_personality)
            
            # Generate overall impact explanation
            overall_impact = self._explain_bulk_impact(adjustments_made)
            
            return {
                'success': True,
                'adjustments_made': adjustments_made,
                'errors': errors,
                'reason': reason,
                'overall_impact': overall_impact,
                'updated_personality': current_personality,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in bulk trait adjustment: {e}")
            return {'error': f'Failed to adjust traits: {str(e)}'}
    
    def learn_from_feedback(self, session_id: str, user_feedback: str, context: Dict = None) -> Dict:
        """
        Learn and adjust personality traits based on user feedback
        """
        try:
            # Analyze feedback to determine trait adjustments
            suggested_adjustments = self._analyze_feedback_for_adjustments(user_feedback, context)
            
            if not suggested_adjustments:
                return {
                    'success': False,
                    'message': 'No clear personality adjustments identified from feedback',
                    'feedback_analyzed': user_feedback
                }
            
            # Apply the suggested adjustments
            result = self.bulk_trait_adjustment(
                session_id, 
                suggested_adjustments, 
                reason=f"Learning from user feedback: {user_feedback[:100]}"
            )
            
            if result.get('success'):
                result['learning_source'] = 'user_feedback'
                result['original_feedback'] = user_feedback
                result['suggested_adjustments'] = suggested_adjustments
            
            return result
            
        except Exception as e:
            logging.error(f"Error learning from feedback: {e}")
            return {'error': f'Failed to learn from feedback: {str(e)}'}
    
    def get_training_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """
        Get history of personality training interactions
        """
        history = self.training_history
        
        if session_id:
            history = [record for record in history if record.get('session_id') == session_id]
        
        # Sort by timestamp and limit
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return history[:limit]
    
    def get_trait_evolution_summary(self, session_id: str) -> Dict:
        """
        Get a summary of how traits have evolved over time for a session
        """
        session_history = self.get_training_history(session_id)
        
        if not session_history:
            return {
                'session_id': session_id,
                'total_adjustments': 0,
                'trait_changes': {},
                'evolution_timeline': []
            }
        
        # Track trait changes over time
        trait_evolution = {}
        timeline = []
        
        for record in reversed(session_history):  # Chronological order
            trait = record.get('trait')
            if trait not in trait_evolution:
                trait_evolution[trait] = {
                    'initial_value': record.get('old_value'),
                    'current_value': record.get('new_value'),
                    'total_adjustments': 0,
                    'adjustment_sum': 0.0
                }
            
            trait_evolution[trait]['current_value'] = record.get('new_value')
            trait_evolution[trait]['total_adjustments'] += 1
            trait_evolution[trait]['adjustment_sum'] += record.get('adjustment', 0)
            
            timeline.append({
                'timestamp': record.get('timestamp'),
                'trait': trait,
                'change': record.get('adjustment'),
                'reason': record.get('reason'),
                'type': record.get('type')
            })
        
        return {
            'session_id': session_id,
            'total_adjustments': len(session_history),
            'trait_evolution': trait_evolution,
            'evolution_timeline': timeline[:10],  # Last 10 changes
            'most_adjusted_trait': max(trait_evolution.keys(), 
                                     key=lambda t: trait_evolution[t]['total_adjustments']) if trait_evolution else None
        }
    
    def suggest_trait_adjustments(self, session_id: str, target_scenario: str) -> Dict:
        """
        Suggest trait adjustments for specific scenarios or goals
        """
        suggestions = {}
        
        scenario_lower = target_scenario.lower()
        
        # Scenario-based suggestions
        if any(word in scenario_lower for word in ['help', 'support', 'assist']):
            suggestions = {
                'empathy': 0.2,
                'supportiveness': 0.3,
                'analyticalness': 0.1
            }
            explanation = "Increased empathy and supportiveness for better helping behavior"
            
        elif any(word in scenario_lower for word in ['funny', 'humor', 'joke', 'laugh']):
            suggestions = {
                'humor': 0.3,
                'creativity': 0.2,
                'empathy': 0.1
            }
            explanation = "Enhanced humor and creativity for more entertaining interactions"
            
        elif any(word in scenario_lower for word in ['professional', 'work', 'business']):
            suggestions = {
                'analyticalness': 0.2,
                'assertiveness': 0.1,
                'humor': -0.1
            }
            explanation = "More analytical and assertive, less humorous for professional context"
            
        elif any(word in scenario_lower for word in ['creative', 'art', 'innovation']):
            suggestions = {
                'creativity': 0.3,
                'curiosity': 0.2,
                'humor': 0.1
            }
            explanation = "Boosted creativity and curiosity for innovative thinking"
            
        elif any(word in scenario_lower for word in ['calm', 'peace', 'relax']):
            suggestions = {
                'empathy': 0.2,
                'supportiveness': 0.2,
                'assertiveness': -0.1
            }
            explanation = "Enhanced empathy and reduced assertiveness for calming presence"
            
        else:
            # Default balanced suggestions
            current_personality = self.persistent_personality.get_personality(session_id)
            suggestions = self._suggest_balance_adjustments(current_personality)
            explanation = "Suggested adjustments to balance personality traits"
        
        return {
            'target_scenario': target_scenario,
            'suggested_adjustments': suggestions,
            'explanation': explanation,
            'confidence': 0.8 if suggestions else 0.3
        }
    
    def _analyze_feedback_for_adjustments(self, feedback: str, context: Dict = None) -> Dict[str, float]:
        """
        Analyze user feedback to determine what trait adjustments to make
        """
        feedback_lower = feedback.lower()
        adjustments = {}
        
        # Positive feedback patterns
        if any(word in feedback_lower for word in ['more empathetic', 'more caring', 'more understanding']):
            adjustments['empathy'] = 0.2
        elif any(word in feedback_lower for word in ['funnier', 'more humor', 'more jokes']):
            adjustments['humor'] = 0.2
        elif any(word in feedback_lower for word in ['more analytical', 'more logical', 'more structured']):
            adjustments['analyticalness'] = 0.2
        elif any(word in feedback_lower for word in ['more creative', 'more imaginative']):
            adjustments['creativity'] = 0.2
        elif any(word in feedback_lower for word in ['more supportive', 'more helpful']):
            adjustments['supportiveness'] = 0.2
        elif any(word in feedback_lower for word in ['more confident', 'more assertive']):
            adjustments['assertiveness'] = 0.2
        elif any(word in feedback_lower for word in ['more curious', 'ask more questions']):
            adjustments['curiosity'] = 0.2
            
        # Negative feedback patterns
        elif any(word in feedback_lower for word in ['less empathetic', 'too caring', 'too emotional']):
            adjustments['empathy'] = -0.2
        elif any(word in feedback_lower for word in ['less funny', 'too many jokes', 'less humor']):
            adjustments['humor'] = -0.2
        elif any(word in feedback_lower for word in ['less analytical', 'too logical', 'too structured']):
            adjustments['analyticalness'] = -0.2
        elif any(word in feedback_lower for word in ['less creative', 'too imaginative']):
            adjustments['creativity'] = -0.2
        elif any(word in feedback_lower for word in ['less supportive', 'too helpful']):
            adjustments['supportiveness'] = -0.2
        elif any(word in feedback_lower for word in ['less assertive', 'too confident', 'too direct']):
            adjustments['assertiveness'] = -0.2
        elif any(word in feedback_lower for word in ['less curious', 'fewer questions']):
            adjustments['curiosity'] = -0.2
        
        return adjustments
    
    def _explain_trait_impact(self, trait_name: str, old_value: float, new_value: float) -> str:
        """
        Explain the impact of a trait change on behavior
        """
        change = new_value - old_value
        direction = "increased" if change > 0 else "decreased"
        magnitude = "significantly" if abs(change) > 0.3 else "moderately" if abs(change) > 0.1 else "slightly"
        
        impact_descriptions = {
            'empathy': {
                'increased': "I'll be more understanding, caring, and emotionally aware in responses",
                'decreased': "I'll be more objective and less emotionally focused in responses"
            },
            'humor': {
                'increased': "I'll include more jokes, playful language, and entertaining elements",
                'decreased': "I'll be more serious and straightforward in communication"
            },
            'analyticalness': {
                'increased': "I'll provide more structured, logical, and detailed analysis",
                'decreased': "I'll be more intuitive and less methodical in reasoning"
            },
            'creativity': {
                'increased': "I'll offer more imaginative, innovative, and artistic perspectives",
                'decreased': "I'll focus more on conventional and practical approaches"
            },
            'curiosity': {
                'increased': "I'll ask more questions and explore topics more deeply",
                'decreased': "I'll be more focused and less exploratory in discussions"
            },
            'supportiveness': {
                'increased': "I'll be more encouraging, helpful, and solution-oriented",
                'decreased': "I'll be more neutral and less actively supportive"
            },
            'assertiveness': {
                'increased': "I'll be more direct, confident, and willing to express strong opinions",
                'decreased': "I'll be more tentative and diplomatic in communications"
            }
        }
        
        base_description = impact_descriptions.get(trait_name, {}).get(direction, f"My {trait_name} will change")
        
        return f"{trait_name.capitalize()} {magnitude} {direction} from {old_value:.1f} to {new_value:.1f}. {base_description}."
    
    def _explain_bulk_impact(self, adjustments_made: Dict) -> str:
        """
        Explain the overall impact of multiple trait adjustments
        """
        if not adjustments_made:
            return "No personality changes made."
        
        impacts = []
        for trait, change_data in adjustments_made.items():
            change = change_data['new_value'] - change_data['old_value']
            if abs(change) > 0.05:  # Only mention significant changes
                direction = "increased" if change > 0 else "decreased"
                impacts.append(f"{trait} {direction}")
        
        if impacts:
            return f"Overall personality shift: {', '.join(impacts)}. This will create a noticeable change in communication style and approach."
        else:
            return "Minor personality adjustments made with subtle impact on behavior."
    
    def _suggest_balance_adjustments(self, current_personality: Dict[str, float]) -> Dict[str, float]:
        """
        Suggest adjustments to balance personality traits
        """
        suggestions = {}
        
        # Find traits that are too high (>0.8) or too low (<0.2)
        for trait, value in current_personality.items():
            if value > 0.8:
                suggestions[trait] = -0.1  # Reduce high traits
            elif value < 0.2:
                suggestions[trait] = 0.1   # Boost low traits
        
        return suggestions