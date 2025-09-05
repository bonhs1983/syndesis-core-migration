"""
FractalTrain Content Generation Module
Ενσωματωμένη δημιουργία περιεχομένου στο FractalTrain AI system
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai

logger = logging.getLogger(__name__)

class FractalContentGenerator:
    """
    Content Generation Engine ενσωματωμένος στο FractalTrain system
    Χρησιμοποιεί την προσωπικότητα του agent για personalized content
    """
    
    def __init__(self, personality_traits: Optional[Dict] = None):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.personality_traits = personality_traits or {}
        
        # Content categories optimized για FractalTrain
        self.content_categories = {
            'conversational': {
                'description': 'Personality-driven conversational content',
                'templates': ['greeting', 'response', 'explanation', 'apology']
            },
            'educational': {
                'description': 'Learning and teaching content',
                'templates': ['tutorial', 'explanation', 'exercise', 'assessment']
            },
            'creative': {
                'description': 'Creative writing with personality influence',
                'templates': ['story', 'poem', 'dialogue', 'character']
            },
            'business': {
                'description': 'Business and professional content',
                'templates': ['proposal', 'email', 'presentation', 'report']
            },
            'marketing': {
                'description': 'AI-generated marketing content',
                'templates': ['ad_copy', 'social_post', 'campaign', 'branding']
            }
        }
    
    def update_personality(self, traits: Dict[str, float]):
        """
        Ενημερώνει τα personality traits για το content generation
        """
        self.personality_traits.update(traits)
        logger.info(f"Content generator personality updated: {traits}")
    
    def generate_personality_aware_content(self, 
                                         content_type: str,
                                         prompt: str,
                                         context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Δημιουργεί περιεχόμενο βάσει της προσωπικότητας του FractalTrain agent
        """
        try:
            # Build personality-aware prompt
            enhanced_prompt = self._build_personality_prompt(content_type, prompt, context)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are the FractalTrain AI content generator. Create content that reflects the personality traits and maintains consistency with the FractalTrain conversational style."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=1200,
                temperature=self._calculate_creativity_temperature(),
                response_format={"type": "json_object"}
            )
            
            content_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'content': content_data,
                'personality_influence': self._analyze_personality_influence(),
                'content_type': content_type,
                'generated_at': datetime.now().isoformat(),
                'traits_used': dict(self.personality_traits)
            }
            
        except Exception as e:
            logger.error(f"FractalTrain content generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'content_type': content_type
            }
    
    def generate_adaptive_response(self, 
                                 user_input: str, 
                                 conversation_context: List[Dict],
                                 response_style: str = "adaptive") -> Dict[str, Any]:
        """
        Δημιουργεί adaptive responses βάσει conversation context και personality
        """
        try:
            # Analyze conversation for personality adaptation cues
            context_analysis = self._analyze_conversation_context(conversation_context)
            
            # Build adaptive prompt
            prompt = f"""
Generate an adaptive response to: "{user_input}"

Conversation Context Analysis:
{json.dumps(context_analysis, indent=2)}

Current Personality Traits:
{json.dumps(self.personality_traits, indent=2)}

Response Style: {response_style}

Create a JSON response with:
1. "main_response": The primary response text
2. "personality_adjustments": Any trait adjustments this response should trigger
3. "emotional_tone": The emotional tone of the response
4. "reasoning": Why this response fits the personality and context
5. "alternative_responses": 2-3 alternative response variations

Make the response authentic to the FractalTrain personality system.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are the FractalTrain adaptive response generator. Create responses that evolve personality traits based on context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'response': response_data,
                'context_analysis': context_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Adaptive response generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': "I'm processing your request. Let me think about that..."
            }
    
    def generate_visual_concept_with_personality(self, 
                                               description: str,
                                               visual_style: Optional[str] = None) -> Dict[str, Any]:
        """
        Δημιουργεί visual concepts που αντικατοπτρίζουν την προσωπικότητα
        """
        try:
            # Enhance description with personality traits
            personality_visual_style = self._map_personality_to_visual_style()
            enhanced_description = f"{description}, {personality_visual_style}"
            
            if visual_style:
                enhanced_description += f", {visual_style}"
            
            # Add quality and style modifiers
            enhanced_description += ", high quality, professional, AI personality visualization"
            
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=enhanced_description,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            return {
                'success': True,
                'image_url': response.data[0].url,
                'enhanced_prompt': enhanced_description,
                'personality_influence': personality_visual_style,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Personality-aware image generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_conversation_summary(self, 
                                    conversation_history: List[Dict],
                                    summary_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Δημιουργεί summaries από conversation history με personality awareness
        """
        try:
            # Prepare conversation data
            conversation_text = self._format_conversation_for_summary(conversation_history)
            
            prompt = f"""
Analyze and summarize this FractalTrain conversation:

{conversation_text}

Summary Type: {summary_type}
Current AI Personality: {json.dumps(self.personality_traits, indent=2)}

Create a JSON response with:
1. "main_summary": Overall conversation summary
2. "personality_evolution": How personality traits changed during conversation
3. "key_topics": Main topics discussed
4. "emotional_progression": How the emotional tone evolved
5. "relationship_insights": Insights about the human-AI relationship
6. "suggestions": Suggestions for future interactions

Focus on the unique FractalTrain personality evolution aspects.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are the FractalTrain conversation analyzer. Focus on personality evolution and relationship dynamics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            
            summary_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'summary': summary_data,
                'conversation_length': len(conversation_history),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Conversation summary generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_personality_prompt(self, content_type: str, prompt: str, context: Optional[Dict]) -> str:
        """
        Χτίζει prompt που ενσωματώνει personality traits
        """
        personality_description = self._describe_current_personality()
        
        enhanced_prompt = f"""
Content Type: {content_type}
Original Request: {prompt}

FractalTrain AI Personality Profile:
{personality_description}

Additional Context:
{json.dumps(context or {}, indent=2)}

Create content that authentically reflects these personality traits while fulfilling the request.
Response should be in JSON format with appropriate structure for the content type.
"""
        return enhanced_prompt
    
    def _describe_current_personality(self) -> str:
        """
        Περιγράφει την τρέχουσα προσωπικότητα σε φυσική γλώσσα
        """
        if not self.personality_traits:
            return "Balanced, neutral personality"
        
        descriptions = []
        
        # Map traits to descriptions
        trait_descriptions = {
            'empathy': ('low empathy, direct and analytical', 'high empathy, caring and understanding'),
            'creativity': ('structured and conventional', 'highly creative and innovative'),
            'humor': ('serious and professional', 'playful and humorous'),
            'curiosity': ('focused and practical', 'highly curious and exploratory'),
            'supportiveness': ('independent and objective', 'very supportive and encouraging'),
            'analyticalness': ('intuitive and emotion-based', 'highly analytical and logical')
        }
        
        for trait, value in self.personality_traits.items():
            if trait in trait_descriptions:
                low_desc, high_desc = trait_descriptions[trait]
                if value < 0.3:
                    descriptions.append(low_desc)
                elif value > 0.7:
                    descriptions.append(high_desc)
                else:
                    descriptions.append(f"moderate {trait}")
        
        return "Personality: " + ", ".join(descriptions)
    
    def _calculate_creativity_temperature(self) -> float:
        """
        Υπολογίζει temperature βάσει creativity trait
        """
        creativity = self.personality_traits.get('creativity', 0.5)
        # Map creativity (0-1) to temperature (0.3-0.9)
        return 0.3 + (creativity * 0.6)
    
    def _analyze_personality_influence(self) -> Dict[str, str]:
        """
        Αναλύει πώς τα traits επηρέασαν το content
        """
        influences = {}
        
        for trait, value in self.personality_traits.items():
            if value > 0.7:
                influences[trait] = f"High {trait} strongly influenced content style"
            elif value < 0.3:
                influences[trait] = f"Low {trait} created more {trait.replace('ness', '')} approach"
        
        return influences
    
    def _analyze_conversation_context(self, conversation: List[Dict]) -> Dict[str, Any]:
        """
        Αναλύει το conversation context για adaptive responses
        """
        if not conversation:
            return {"context": "no_history", "sentiment": "neutral"}
        
        recent_messages = conversation[-3:] if len(conversation) > 3 else conversation
        
        # Simple context analysis
        total_length = sum(len(msg.get('content', '')) for msg in recent_messages)
        user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
        
        return {
            'recent_message_count': len(recent_messages),
            'average_message_length': total_length / len(recent_messages) if recent_messages else 0,
            'user_engagement': len(user_messages),
            'conversation_depth': 'deep' if total_length > 500 else 'casual',
            'context_hint': 'continuing_conversation' if len(conversation) > 5 else 'early_conversation'
        }
    
    def _map_personality_to_visual_style(self) -> str:
        """
        Μετατρέπει personality traits σε visual style descriptions
        """
        styles = []
        
        if self.personality_traits.get('creativity', 0.5) > 0.7:
            styles.append("creative and artistic")
        elif self.personality_traits.get('creativity', 0.5) < 0.3:
            styles.append("clean and minimalist")
        
        if self.personality_traits.get('analyticalness', 0.5) > 0.7:
            styles.append("structured and geometric")
        
        if self.personality_traits.get('empathy', 0.5) > 0.7:
            styles.append("warm and approachable")
        
        if self.personality_traits.get('humor', 0.5) > 0.7:
            styles.append("playful and engaging")
        
        return ", ".join(styles) if styles else "balanced and professional"
    
    def _format_conversation_for_summary(self, conversation: List[Dict]) -> str:
        """
        Μορφοποιεί conversation history για summary generation
        """
        formatted = []
        
        for i, msg in enumerate(conversation[-10:]):  # Last 10 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            formatted.append(f"{i+1}. {role.upper()}: {content}")
        
        return "\n".join(formatted)
    
    def get_personality_content_suggestions(self) -> Dict[str, List[str]]:
        """
        Προτάσεις περιεχομένου βάσει τρέχουσας προσωπικότητας
        """
        suggestions = {
            'recommended_content_types': [],
            'personality_strengths': [],
            'content_opportunities': []
        }
        
        # Analyze current traits for suggestions
        for trait, value in self.personality_traits.items():
            if value > 0.7:
                if trait == 'creativity':
                    suggestions['recommended_content_types'].extend(['creative writing', 'visual concepts', 'innovative solutions'])
                elif trait == 'empathy':
                    suggestions['recommended_content_types'].extend(['supportive messages', 'emotional content', 'relationship advice'])
                elif trait == 'analyticalness':
                    suggestions['recommended_content_types'].extend(['detailed explanations', 'data analysis', 'technical content'])
                elif trait == 'humor':
                    suggestions['recommended_content_types'].extend(['entertaining content', 'light-hearted responses', 'playful interactions'])
        
        return suggestions