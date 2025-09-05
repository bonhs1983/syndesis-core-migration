"""
Adaptive Communication Style Learning System
Learns and adapts communication patterns based on user interactions and feedback
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re


class AdaptiveCommunicationLearner:
    """
    Learns communication patterns and adapts style based on user preferences and feedback
    """
    
    def __init__(self, persistent_personality):
        self.persistent_personality = persistent_personality
        self.communication_patterns = {}
        self.learning_history = []
        self.style_preferences = {}
        
    def analyze_communication_effectiveness(self, session_id: str, user_input: str, 
                                          ai_response: str, user_feedback: str = None) -> Dict:
        """
        Analyze the effectiveness of communication and learn from it
        """
        try:
            # Extract communication features
            communication_features = self._extract_communication_features(ai_response)
            
            # Analyze user reaction (if feedback provided)
            effectiveness_score = self._calculate_effectiveness(user_input, ai_response, user_feedback)
            
            # Learn from the interaction
            learning_insights = self._generate_learning_insights(
                communication_features, effectiveness_score, user_feedback
            )
            
            # Record the learning
            learning_record = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_input': user_input,
                'ai_response': ai_response,
                'user_feedback': user_feedback,
                'communication_features': communication_features,
                'effectiveness_score': effectiveness_score,
                'learning_insights': learning_insights,
                'type': 'communication_analysis'
            }
            self.learning_history.append(learning_record)
            
            # Update communication patterns
            self._update_communication_patterns(session_id, learning_insights)
            
            return {
                'success': True,
                'effectiveness_score': effectiveness_score,
                'communication_features': communication_features,
                'learning_insights': learning_insights,
                'recommendations': self._generate_style_recommendations(learning_insights),
                'timestamp': learning_record['timestamp']
            }
            
        except Exception as e:
            logging.error(f"Error analyzing communication effectiveness: {e}")
            return {'error': f'Failed to analyze communication: {str(e)}'}
    
    def learn_preferred_communication_style(self, session_id: str, style_preferences: Dict) -> Dict:
        """
        Learn and store user's preferred communication style
        """
        try:
            # Update style preferences for the session
            if session_id not in self.style_preferences:
                self.style_preferences[session_id] = {}
            
            self.style_preferences[session_id].update(style_preferences)
            
            # Convert preferences to personality adjustments
            personality_adjustments = self._convert_style_to_traits(style_preferences)
            
            # Apply the adjustments
            if personality_adjustments:
                current_personality = self.persistent_personality.get_personality(session_id)
                for trait, adjustment in personality_adjustments.items():
                    if trait in current_personality:
                        current_personality[trait] = max(0.0, min(1.0, 
                                                                current_personality[trait] + adjustment))
                
                self.persistent_personality.update_personality(session_id, current_personality)
            
            # Record the learning
            learning_record = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'style_preferences': style_preferences,
                'personality_adjustments': personality_adjustments,
                'type': 'style_learning'
            }
            self.learning_history.append(learning_record)
            
            return {
                'success': True,
                'learned_preferences': style_preferences,
                'personality_adjustments': personality_adjustments,
                'updated_personality': current_personality,
                'explanation': self._explain_style_learning(style_preferences)
            }
            
        except Exception as e:
            logging.error(f"Error learning communication style: {e}")
            return {'error': f'Failed to learn style: {str(e)}'}
    
    def get_adaptive_style_recommendations(self, session_id: str, context: Dict = None) -> Dict:
        """
        Get recommendations for adapting communication style based on learned patterns
        """
        try:
            # Get session-specific patterns
            session_patterns = self._get_session_patterns(session_id)
            
            # Get style preferences
            preferences = self.style_preferences.get(session_id, {})
            
            # Analyze context for style adaptation
            context_recommendations = self._analyze_context_for_style(context) if context else {}
            
            # Generate comprehensive recommendations
            recommendations = {
                'session_patterns': session_patterns,
                'style_preferences': preferences,
                'context_recommendations': context_recommendations,
                'adaptive_suggestions': self._generate_adaptive_suggestions(
                    session_patterns, preferences, context_recommendations
                ),
                'confidence_level': self._calculate_recommendation_confidence(session_patterns)
            }
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error getting style recommendations: {e}")
            return {'error': f'Failed to get recommendations: {str(e)}'}
    
    def track_communication_evolution(self, session_id: str, days_back: int = 30) -> Dict:
        """
        Track how communication style has evolved over time
        """
        try:
            # Get recent learning history
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_history = [
                record for record in self.learning_history
                if (record.get('session_id') == session_id and 
                    datetime.fromisoformat(record.get('timestamp', '')) > cutoff_date)
            ]
            
            if not recent_history:
                return {
                    'session_id': session_id,
                    'evolution_period': f'{days_back} days',
                    'total_interactions': 0,
                    'style_evolution': {},
                    'effectiveness_trend': []
                }
            
            # Analyze evolution patterns
            style_evolution = self._analyze_style_evolution(recent_history)
            effectiveness_trend = self._calculate_effectiveness_trend(recent_history)
            communication_improvements = self._identify_improvements(recent_history)
            
            return {
                'session_id': session_id,
                'evolution_period': f'{days_back} days',
                'total_interactions': len(recent_history),
                'style_evolution': style_evolution,
                'effectiveness_trend': effectiveness_trend,
                'communication_improvements': communication_improvements,
                'learning_velocity': len(recent_history) / max(days_back, 1),
                'adaptation_summary': self._generate_adaptation_summary(style_evolution)
            }
            
        except Exception as e:
            logging.error(f"Error tracking communication evolution: {e}")
            return {'error': f'Failed to track evolution: {str(e)}'}
    
    def get_learning_analytics(self, session_id: str = None) -> Dict:
        """
        Get comprehensive analytics about communication learning
        """
        try:
            history = self.learning_history
            if session_id:
                history = [record for record in history if record.get('session_id') == session_id]
            
            # Calculate analytics
            total_interactions = len(history)
            avg_effectiveness = sum(record.get('effectiveness_score', 0) for record in history) / max(total_interactions, 1)
            
            # Most effective communication features
            feature_effectiveness = {}
            for record in history:
                features = record.get('communication_features', {})
                score = record.get('effectiveness_score', 0)
                for feature, value in features.items():
                    if feature not in feature_effectiveness:
                        feature_effectiveness[feature] = []
                    feature_effectiveness[feature].append((value, score))
            
            # Calculate feature correlations with effectiveness
            feature_analysis = {}
            for feature, values_scores in feature_effectiveness.items():
                if len(values_scores) > 1:
                    avg_score = sum(score for _, score in values_scores) / len(values_scores)
                    feature_analysis[feature] = {
                        'average_effectiveness': avg_score,
                        'sample_size': len(values_scores)
                    }
            
            return {
                'session_id': session_id,
                'total_interactions': total_interactions,
                'average_effectiveness': avg_effectiveness,
                'feature_analysis': feature_analysis,
                'learning_trends': self._calculate_learning_trends(history),
                'most_effective_features': sorted(
                    feature_analysis.items(), 
                    key=lambda x: x[1]['average_effectiveness'], 
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            logging.error(f"Error getting learning analytics: {e}")
            return {'error': f'Failed to get analytics: {str(e)}'}
    
    def _extract_communication_features(self, ai_response: str) -> Dict:
        """
        Extract key communication features from an AI response
        """
        features = {}
        
        # Length and structure
        features['response_length'] = len(ai_response)
        features['sentence_count'] = len(re.split(r'[.!?]+', ai_response))
        features['word_count'] = len(ai_response.split())
        
        # Language patterns
        features['question_count'] = ai_response.count('?')
        features['exclamation_count'] = ai_response.count('!')
        features['uses_first_person'] = 1 if any(word in ai_response.lower() for word in ['i ', 'me ', 'my ', 'myself']) else 0
        features['uses_second_person'] = 1 if any(word in ai_response.lower() for word in ['you ', 'your ', 'yourself']) else 0
        
        # Emotional indicators
        features['positive_words'] = len([word for word in ai_response.lower().split() 
                                        if word in ['great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'good']])
        features['empathy_indicators'] = len([word for word in ai_response.lower().split() 
                                            if word in ['understand', 'feel', 'sorry', 'care', 'support']])
        
        # Technical vs casual
        features['technical_words'] = len([word for word in ai_response.lower().split() 
                                         if word in ['analysis', 'process', 'system', 'method', 'algorithm', 'data']])
        features['casual_words'] = len([word for word in ai_response.lower().split() 
                                      if word in ['hey', 'yeah', 'cool', 'awesome', 'nice', 'fun']])
        
        # Humor indicators
        features['humor_indicators'] = 1 if any(indicator in ai_response.lower() 
                                              for indicator in ['haha', 'lol', '*', 'joke', 'funny']) else 0
        
        return features
    
    def _calculate_effectiveness(self, user_input: str, ai_response: str, user_feedback: str = None) -> float:
        """
        Calculate communication effectiveness score (0.0 to 1.0)
        """
        score = 0.5  # Default neutral score
        
        if user_feedback:
            feedback_lower = user_feedback.lower()
            
            # Positive feedback indicators
            positive_indicators = ['good', 'great', 'helpful', 'perfect', 'excellent', 'love', 'like', 'thanks']
            negative_indicators = ['bad', 'wrong', 'unhelpful', 'confusing', 'hate', 'dislike', 'terrible']
            
            positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
            negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
            
            if positive_count > negative_count:
                score = 0.7 + (positive_count * 0.1)
            elif negative_count > positive_count:
                score = 0.3 - (negative_count * 0.1)
            
            score = max(0.0, min(1.0, score))
        
        return score
    
    def _generate_learning_insights(self, features: Dict, effectiveness: float, feedback: str = None) -> Dict:
        """
        Generate insights for learning from communication patterns
        """
        insights = {}
        
        # Analyze feature effectiveness
        if effectiveness > 0.7:
            insights['effective_patterns'] = {
                'response_length': features.get('response_length', 0),
                'question_style': features.get('question_count', 0) > 0,
                'empathy_level': features.get('empathy_indicators', 0),
                'technical_level': features.get('technical_words', 0),
                'casual_level': features.get('casual_words', 0)
            }
        elif effectiveness < 0.4:
            insights['ineffective_patterns'] = {
                'response_length': features.get('response_length', 0),
                'question_style': features.get('question_count', 0) > 0,
                'empathy_level': features.get('empathy_indicators', 0),
                'technical_level': features.get('technical_words', 0),
                'casual_level': features.get('casual_words', 0)
            }
        
        # Feedback-based insights
        if feedback:
            insights['feedback_analysis'] = self._analyze_feedback_insights(feedback)
        
        return insights
    
    def _convert_style_to_traits(self, style_preferences: Dict) -> Dict[str, float]:
        """
        Convert communication style preferences to personality trait adjustments
        """
        adjustments = {}
        
        for preference, value in style_preferences.items():
            if preference == 'formality' and isinstance(value, str):
                if value == 'formal':
                    adjustments['analyticalness'] = 0.1
                    adjustments['humor'] = -0.1
                elif value == 'casual':
                    adjustments['humor'] = 0.1
                    adjustments['analyticalness'] = -0.1
                    
            elif preference == 'empathy_level' and isinstance(value, (int, float)):
                adjustment = (value - 0.5) * 0.4  # Convert 0-1 scale to adjustment
                adjustments['empathy'] = adjustment
                
            elif preference == 'humor_level' and isinstance(value, (int, float)):
                adjustment = (value - 0.5) * 0.4
                adjustments['humor'] = adjustment
                
            elif preference == 'technical_level' and isinstance(value, (int, float)):
                adjustment = (value - 0.5) * 0.4
                adjustments['analyticalness'] = adjustment
                
            elif preference == 'supportiveness' and isinstance(value, (int, float)):
                adjustment = (value - 0.5) * 0.4
                adjustments['supportiveness'] = adjustment
        
        return adjustments
    
    def _get_session_patterns(self, session_id: str) -> Dict:
        """
        Get communication patterns specific to a session
        """
        session_history = [record for record in self.learning_history 
                          if record.get('session_id') == session_id]
        
        if not session_history:
            return {}
        
        # Aggregate patterns
        patterns = {
            'average_effectiveness': sum(record.get('effectiveness_score', 0) 
                                       for record in session_history) / len(session_history),
            'total_interactions': len(session_history),
            'preferred_features': {},
            'learning_momentum': len(session_history) / max((datetime.now() - 
                                   datetime.fromisoformat(session_history[0]['timestamp'])).days, 1)
        }
        
        return patterns
    
    def _analyze_context_for_style(self, context: Dict) -> Dict:
        """
        Analyze context to recommend communication style adaptations
        """
        recommendations = {}
        
        if 'mood' in context:
            mood = context['mood'].lower()
            if mood in ['sad', 'frustrated', 'angry']:
                recommendations['increase_empathy'] = 0.2
                recommendations['decrease_humor'] = 0.1
            elif mood in ['happy', 'excited']:
                recommendations['increase_humor'] = 0.1
                recommendations['increase_creativity'] = 0.1
                
        if 'topic' in context:
            topic = context['topic'].lower()
            if 'technical' in topic or 'work' in topic:
                recommendations['increase_analytical'] = 0.2
                recommendations['decrease_humor'] = 0.1
            elif 'personal' in topic or 'emotional' in topic:
                recommendations['increase_empathy'] = 0.2
                recommendations['increase_supportiveness'] = 0.1
        
        return recommendations
    
    def _generate_adaptive_suggestions(self, patterns: Dict, preferences: Dict, context_rec: Dict) -> List[str]:
        """
        Generate adaptive communication suggestions
        """
        suggestions = []
        
        # Based on effectiveness patterns
        if patterns.get('average_effectiveness', 0.5) < 0.6:
            suggestions.append("Consider adjusting communication style for better engagement")
        
        # Based on preferences
        if preferences.get('formality') == 'formal':
            suggestions.append("Maintain formal, structured communication style")
        elif preferences.get('formality') == 'casual':
            suggestions.append("Use casual, friendly communication approach")
        
        # Based on context
        if context_rec.get('increase_empathy'):
            suggestions.append("Increase empathetic and understanding tone")
        if context_rec.get('increase_humor'):
            suggestions.append("Add appropriate humor and lightness")
        
        return suggestions
    
    def _calculate_recommendation_confidence(self, patterns: Dict) -> float:
        """
        Calculate confidence level for recommendations based on data quality
        """
        interaction_count = patterns.get('total_interactions', 0)
        
        if interaction_count >= 20:
            return 0.9
        elif interaction_count >= 10:
            return 0.7
        elif interaction_count >= 5:
            return 0.5
        else:
            return 0.3
    
    def _analyze_style_evolution(self, history: List[Dict]) -> Dict:
        """
        Analyze how communication style has evolved over time
        """
        if len(history) < 2:
            return {}
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get('timestamp', ''))
        
        early_period = history[:len(history)//2]
        recent_period = history[len(history)//2:]
        
        evolution = {}
        
        # Compare average effectiveness
        early_effectiveness = sum(record.get('effectiveness_score', 0) for record in early_period) / len(early_period)
        recent_effectiveness = sum(record.get('effectiveness_score', 0) for record in recent_period) / len(recent_period)
        
        evolution['effectiveness_change'] = recent_effectiveness - early_effectiveness
        evolution['improvement_trend'] = evolution['effectiveness_change'] > 0.1
        
        return evolution
    
    def _calculate_effectiveness_trend(self, history: List[Dict]) -> List[Dict]:
        """
        Calculate effectiveness trend over time
        """
        trend = []
        for record in history[-10:]:  # Last 10 interactions
            trend.append({
                'timestamp': record.get('timestamp'),
                'effectiveness': record.get('effectiveness_score', 0)
            })
        
        return trend
    
    def _identify_improvements(self, history: List[Dict]) -> List[str]:
        """
        Identify specific communication improvements made
        """
        improvements = []
        
        if len(history) < 5:
            return improvements
        
        recent_avg = sum(record.get('effectiveness_score', 0) for record in history[-5:]) / 5
        earlier_avg = sum(record.get('effectiveness_score', 0) for record in history[-10:-5]) / 5 if len(history) >= 10 else 0.5
        
        if recent_avg > earlier_avg + 0.1:
            improvements.append("Overall communication effectiveness has improved")
        
        return improvements
    
    def _generate_adaptation_summary(self, evolution: Dict) -> str:
        """
        Generate a summary of communication adaptations
        """
        if not evolution:
            return "Insufficient data for adaptation analysis"
        
        effectiveness_change = evolution.get('effectiveness_change', 0)
        
        if effectiveness_change > 0.2:
            return "Significant improvement in communication effectiveness through adaptive learning"
        elif effectiveness_change > 0.1:
            return "Moderate improvement in communication style adaptation"
        elif effectiveness_change > 0:
            return "Slight positive adaptation in communication patterns"
        elif effectiveness_change > -0.1:
            return "Communication style relatively stable with minor variations"
        else:
            return "Communication patterns showing room for further adaptation"
    
    def _analyze_feedback_insights(self, feedback: str) -> Dict:
        """
        Analyze user feedback for communication insights
        """
        feedback_lower = feedback.lower()
        insights = {}
        
        # Communication style feedback
        if any(word in feedback_lower for word in ['formal', 'professional']):
            insights['formality_preference'] = 'formal'
        elif any(word in feedback_lower for word in ['casual', 'friendly', 'relaxed']):
            insights['formality_preference'] = 'casual'
        
        # Tone feedback
        if any(word in feedback_lower for word in ['empathetic', 'understanding', 'caring']):
            insights['tone_preference'] = 'empathetic'
        elif any(word in feedback_lower for word in ['direct', 'straight', 'clear']):
            insights['tone_preference'] = 'direct'
        
        return insights
    
    def _calculate_learning_trends(self, history: List[Dict]) -> Dict:
        """
        Calculate learning trends from interaction history
        """
        if len(history) < 3:
            return {}
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''))
        
        # Calculate trend in effectiveness scores
        effectiveness_scores = [record.get('effectiveness_score', 0) for record in sorted_history]
        
        # Simple linear trend calculation
        n = len(effectiveness_scores)
        x_sum = sum(range(n))
        y_sum = sum(effectiveness_scores)
        xy_sum = sum(i * score for i, score in enumerate(effectiveness_scores))
        x_squared_sum = sum(i * i for i in range(n))
        
        if n * x_squared_sum - x_sum * x_sum != 0:
            slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)
            trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        else:
            slope = 0
            trend_direction = 'stable'
        
        return {
            'trend_direction': trend_direction,
            'slope': slope,
            'recent_average': sum(effectiveness_scores[-5:]) / min(5, len(effectiveness_scores))
        }