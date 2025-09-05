"""
Mood-Based Response Style Previewer
Allows users to preview how AI responses change based on different mood/personality combinations
"""
import logging
from typing import Dict, List, Tuple, Optional
from fractalai.adaptive_response_generator import AdaptiveResponseGenerator


class MoodResponseStylePreviewer:
    """
    Generates response previews for different mood/personality configurations
    """
    
    def __init__(self):
        self.response_generator = AdaptiveResponseGenerator()
        
        # Predefined mood configurations
        self.mood_presets = {
            'happy': {
                'empathy': 0.7,
                'humor': 0.8,
                'creativity': 0.7,
                'analyticalness': 0.4,
                'curiosity': 0.6,
                'supportiveness': 0.7,
                'assertiveness': 0.5
            },
            'excited': {
                'empathy': 0.6,
                'humor': 0.9,
                'creativity': 0.8,
                'analyticalness': 0.3,
                'curiosity': 0.8,
                'supportiveness': 0.6,
                'assertiveness': 0.7
            },
            'analytical': {
                'empathy': 0.3,
                'humor': 0.2,
                'creativity': 0.4,
                'analyticalness': 0.9,
                'curiosity': 0.8,
                'supportiveness': 0.4,
                'assertiveness': 0.6
            },
            'supportive': {
                'empathy': 0.9,
                'humor': 0.4,
                'creativity': 0.5,
                'analyticalness': 0.5,
                'curiosity': 0.6,
                'supportiveness': 0.9,
                'assertiveness': 0.3
            },
            'professional': {
                'empathy': 0.5,
                'humor': 0.3,
                'creativity': 0.4,
                'analyticalness': 0.7,
                'curiosity': 0.6,
                'supportiveness': 0.6,
                'assertiveness': 0.7
            },
            'creative': {
                'empathy': 0.6,
                'humor': 0.7,
                'creativity': 0.9,
                'analyticalness': 0.4,
                'curiosity': 0.8,
                'supportiveness': 0.5,
                'assertiveness': 0.5
            },
            'calm': {
                'empathy': 0.7,
                'humor': 0.3,
                'creativity': 0.5,
                'analyticalness': 0.6,
                'curiosity': 0.5,
                'supportiveness': 0.8,
                'assertiveness': 0.4
            },
            'energetic': {
                'empathy': 0.6,
                'humor': 0.8,
                'creativity': 0.7,
                'analyticalness': 0.5,
                'curiosity': 0.9,
                'supportiveness': 0.6,
                'assertiveness': 0.8
            }
        }
        
        # Sample questions for previewing
        self.sample_questions = [
            "How are you doing today?",
            "Can you help me solve a problem?",
            "What do you think about artificial intelligence?",
            "I'm feeling stressed about work",
            "Tell me something interesting",
            "What are your capabilities?",
            "How do you learn from conversations?"
        ]
    
    def generate_mood_preview(self, user_input: str, mood_name: str) -> Dict:
        """
        Generate a response preview for a specific mood configuration
        """
        if mood_name not in self.mood_presets:
            return {
                'error': f"Unknown mood: {mood_name}. Available moods: {list(self.mood_presets.keys())}"
            }
        
        mood_traits = self.mood_presets[mood_name]
        
        try:
            response = self.response_generator.generate_adaptive_response(
                user_input=user_input,
                trait_profile=mood_traits,
                context={'preview_mode': True},
                memory_context=None
            )
            
            return {
                'mood': mood_name,
                'traits': mood_traits,
                'input': user_input,
                'response': response,
                'dominant_traits': self._get_dominant_traits(mood_traits),
                'style_explanation': self._explain_style(mood_traits)
            }
        except Exception as e:
            logging.error(f"Error generating mood preview: {e}")
            return {
                'error': f"Failed to generate preview: {str(e)}"
            }
    
    def compare_mood_responses(self, user_input: str, mood_list: List[str]) -> Dict:
        """
        Compare how the same input gets responded to across different moods
        """
        comparisons = {}
        
        for mood in mood_list:
            if mood in self.mood_presets:
                preview = self.generate_mood_preview(user_input, mood)
                if 'error' not in preview:
                    comparisons[mood] = {
                        'response': preview['response'],
                        'dominant_traits': preview['dominant_traits'],
                        'style_explanation': preview['style_explanation']
                    }
                else:
                    comparisons[mood] = preview
        
        return {
            'input': user_input,
            'mood_comparisons': comparisons,
            'analysis': self._analyze_response_differences(comparisons)
        }
    
    def get_mood_recommendations(self, user_context: str, user_preferences: Dict = None) -> List[Dict]:
        """
        Recommend appropriate moods based on user context or preferences
        """
        recommendations = []
        context_lower = user_context.lower()
        
        # Context-based recommendations
        if any(word in context_lower for word in ['problem', 'help', 'stuck', 'challenge']):
            recommendations.append({
                'mood': 'analytical',
                'reason': 'Best for problem-solving and systematic thinking',
                'confidence': 0.9
            })
            recommendations.append({
                'mood': 'supportive',
                'reason': 'Provides emotional support while helping',
                'confidence': 0.7
            })
        
        elif any(word in context_lower for word in ['stressed', 'anxious', 'worried', 'upset']):
            recommendations.append({
                'mood': 'supportive',
                'reason': 'Offers empathy and emotional support',
                'confidence': 0.9
            })
            recommendations.append({
                'mood': 'calm',
                'reason': 'Provides a calming, peaceful interaction',
                'confidence': 0.8
            })
        
        elif any(word in context_lower for word in ['fun', 'joke', 'laugh', 'entertainment']):
            recommendations.append({
                'mood': 'happy',
                'reason': 'Brings humor and positive energy',
                'confidence': 0.9
            })
            recommendations.append({
                'mood': 'excited',
                'reason': 'High energy and enthusiasm',
                'confidence': 0.7
            })
        
        elif any(word in context_lower for word in ['creative', 'idea', 'brainstorm', 'innovative']):
            recommendations.append({
                'mood': 'creative',
                'reason': 'Maximizes creative and innovative thinking',
                'confidence': 0.9
            })
            recommendations.append({
                'mood': 'energetic',
                'reason': 'High curiosity and creative energy',
                'confidence': 0.7
            })
        
        elif any(word in context_lower for word in ['work', 'business', 'professional', 'formal']):
            recommendations.append({
                'mood': 'professional',
                'reason': 'Maintains appropriate business tone',
                'confidence': 0.8
            })
            recommendations.append({
                'mood': 'analytical',
                'reason': 'Structured and logical approach',
                'confidence': 0.7
            })
        
        # Default recommendations if no specific context matches
        if not recommendations:
            recommendations = [
                {
                    'mood': 'happy',
                    'reason': 'Balanced positive energy for general conversation',
                    'confidence': 0.6
                },
                {
                    'mood': 'supportive',
                    'reason': 'Warm and helpful for any situation',
                    'confidence': 0.6
                }
            ]
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:3]  # Return top 3 recommendations
    
    def generate_sample_previews(self, mood_name: str) -> Dict:
        """
        Generate previews for all sample questions in a specific mood
        """
        if mood_name not in self.mood_presets:
            return {
                'error': f"Unknown mood: {mood_name}"
            }
        
        previews = []
        for question in self.sample_questions:
            preview = self.generate_mood_preview(question, mood_name)
            if 'error' not in preview:
                previews.append({
                    'question': question,
                    'response': preview['response'],
                    'style_explanation': preview['style_explanation']
                })
        
        return {
            'mood': mood_name,
            'traits': self.mood_presets[mood_name],
            'sample_previews': previews,
            'mood_description': self._get_mood_description(mood_name)
        }
    
    def get_available_moods(self) -> Dict:
        """
        Get all available mood presets with descriptions
        """
        moods_info = {}
        for mood_name, traits in self.mood_presets.items():
            moods_info[mood_name] = {
                'traits': traits,
                'description': self._get_mood_description(mood_name),
                'dominant_traits': self._get_dominant_traits(traits),
                'use_cases': self._get_mood_use_cases(mood_name)
            }
        
        return moods_info
    
    def _get_dominant_traits(self, traits: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get the top 3 dominant traits"""
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        return sorted_traits[:3]
    
    def _explain_style(self, traits: Dict[str, float]) -> str:
        """Explain the response style based on trait configuration"""
        dominant = self._get_dominant_traits(traits)
        explanations = []
        
        for trait, value in dominant:
            if value > 0.7:
                if trait == 'empathy':
                    explanations.append(f"Very empathetic ({value:.1f}) - warm and understanding")
                elif trait == 'humor':
                    explanations.append(f"Very humorous ({value:.1f}) - playful and engaging")
                elif trait == 'analyticalness':
                    explanations.append(f"Highly analytical ({value:.1f}) - structured and logical")
                elif trait == 'creativity':
                    explanations.append(f"Very creative ({value:.1f}) - imaginative and innovative")
                elif trait == 'curiosity':
                    explanations.append(f"Highly curious ({value:.1f}) - inquisitive and exploratory")
                elif trait == 'supportiveness':
                    explanations.append(f"Very supportive ({value:.1f}) - helpful and encouraging")
                elif trait == 'assertiveness':
                    explanations.append(f"Highly assertive ({value:.1f}) - confident and direct")
        
        return "; ".join(explanations) if explanations else "Balanced trait combination"
    
    def _analyze_response_differences(self, comparisons: Dict) -> str:
        """Analyze the differences between mood-based responses"""
        if len(comparisons) < 2:
            return "Need at least 2 moods to analyze differences"
        
        analysis_points = []
        
        # Analyze response lengths
        lengths = {mood: len(data['response']) for mood, data in comparisons.items() if 'response' in data}
        if lengths:
            longest = max(lengths, key=lengths.get)
            shortest = min(lengths, key=lengths.get)
            analysis_points.append(f"Response length varies: {longest} gives longest responses, {shortest} gives shortest")
        
        # Analyze dominant traits
        trait_analysis = {}
        for mood, data in comparisons.items():
            if 'dominant_traits' in data:
                for trait, value in data['dominant_traits']:
                    if trait not in trait_analysis:
                        trait_analysis[trait] = []
                    trait_analysis[trait].append((mood, value))
        
        for trait, mood_values in trait_analysis.items():
            if len(mood_values) > 1:
                mood_values.sort(key=lambda x: x[1], reverse=True)
                highest_mood = mood_values[0][0]
                analysis_points.append(f"{trait.capitalize()} is highest in {highest_mood} mood")
        
        return "; ".join(analysis_points) if analysis_points else "Similar response patterns across moods"
    
    def _get_mood_description(self, mood_name: str) -> str:
        """Get a description for each mood"""
        descriptions = {
            'happy': 'Positive, upbeat, and optimistic with good humor',
            'excited': 'High energy, enthusiastic, and very curious',
            'analytical': 'Logical, structured, and fact-focused',
            'supportive': 'Empathetic, caring, and emotionally aware',
            'professional': 'Formal, competent, and business-appropriate',
            'creative': 'Imaginative, innovative, and artistic',
            'calm': 'Peaceful, steady, and reassuring',
            'energetic': 'Dynamic, enthusiastic, and highly engaged'
        }
        return descriptions.get(mood_name, 'Custom mood configuration')
    
    def _get_mood_use_cases(self, mood_name: str) -> List[str]:
        """Get typical use cases for each mood"""
        use_cases = {
            'happy': ['General conversation', 'Cheering someone up', 'Casual interactions'],
            'excited': ['Brainstorming sessions', 'Learning new topics', 'Creative projects'],
            'analytical': ['Problem solving', 'Technical discussions', 'Research tasks'],
            'supportive': ['Emotional support', 'Difficult situations', 'Personal advice'],
            'professional': ['Business meetings', 'Formal communications', 'Work-related tasks'],
            'creative': ['Art projects', 'Innovation sessions', 'Creative writing'],
            'calm': ['Stress relief', 'Meditation guidance', 'Peaceful conversations'],
            'energetic': ['Motivation', 'Exercise coaching', 'Adventure planning']
        }
        return use_cases.get(mood_name, ['General purpose'])