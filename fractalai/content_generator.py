"""
Automated Content Generation Engine
Δημιουργεί αυτόματα περιεχόμενο (κείμενα, εικόνες, βίντεο concepts) χρησιμοποιώντας AI
"""
import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai

logger = logging.getLogger(__name__)

class ContentGenerationEngine:
    """
    Automated content generation using OpenAI models για κείμενα, εικόνες, και concepts
    """
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Content generation categories
        self.content_categories = {
            'marketing': {
                'description': 'Marketing campaigns, ads, social media posts',
                'templates': ['social_post', 'ad_copy', 'email_campaign', 'press_release']
            },
            'website': {
                'description': 'Website content, blog posts, product descriptions',
                'templates': ['blog_post', 'product_description', 'landing_page', 'about_page']
            },
            'creative': {
                'description': 'Creative writing, stories, scripts, poems',
                'templates': ['short_story', 'script', 'poem', 'character_description']
            },
            'technical': {
                'description': 'Documentation, tutorials, specifications',
                'templates': ['api_docs', 'tutorial', 'user_manual', 'technical_spec']
            },
            'visual': {
                'description': 'Image concepts, design briefs, visual content',
                'templates': ['logo_concept', 'illustration', 'infographic', 'ui_design']
            }
        }
    
    def generate_text_content(self, 
                            category: str, 
                            template: str, 
                            specifications: Dict[str, Any],
                            user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Δημιουργεί κειμενικό περιεχόμενο βάσει κατηγορίας και προδιαγραφών
        """
        try:
            # Build context-aware prompt
            prompt = self._build_content_prompt(category, template, specifications, user_context)
            
            # Generate content using GPT-4o
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a professional content creator specializing in high-quality, engaging content across multiple domains."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            content_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'content': content_data,
                'category': category,
                'template': template,
                'generated_at': datetime.now().isoformat(),
                'word_count': len(content_data.get('main_content', '').split()),
                'metadata': {
                    'specifications': specifications,
                    'user_context': user_context
                }
            }
            
        except Exception as e:
            logger.error(f"Text content generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'category': category,
                'template': template
            }
    
    def generate_image_concept(self, 
                             description: str, 
                             style: str = "professional",
                             specifications: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Δημιουργεί εικόνα χρησιμοποιώντας DALL-E 3
        """
        try:
            # Enhance the prompt based on style and specifications
            enhanced_prompt = self._enhance_image_prompt(description, style, specifications)
            
            # Generate image using DALL-E 3
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            image_url = response.data[0].url
            
            return {
                'success': True,
                'image_url': image_url,
                'prompt': enhanced_prompt,
                'style': style,
                'generated_at': datetime.now().isoformat(),
                'specifications': specifications or {}
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'prompt': description
            }
    
    def generate_video_concept(self, 
                             theme: str, 
                             duration: str = "30s",
                             target_audience: str = "general") -> Dict[str, Any]:
        """
        Δημιουργεί concept για βίντεο (storyboard και script)
        """
        try:
            prompt = f"""Create a comprehensive video concept for the following:
            
Theme: {theme}
Duration: {duration}
Target Audience: {target_audience}

Please provide a JSON response with:
1. "title": A catchy video title
2. "hook": Opening hook (first 3 seconds)
3. "storyboard": Array of scenes with descriptions and timing
4. "script": Complete narration script
5. "visual_elements": Key visual elements needed
6. "music_style": Recommended background music style
7. "call_to_action": Ending call to action

Make it engaging and professional."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional video content creator and scriptwriter."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            concept_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'concept': concept_data,
                'theme': theme,
                'duration': duration,
                'target_audience': target_audience,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Video concept generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'theme': theme
            }
    
    def generate_music_concept(self, 
                             genre: str, 
                             mood: str,
                             purpose: str = "background") -> Dict[str, Any]:
        """
        Δημιουργεί concept για μουσική (structure, instruments, mood progression)
        """
        try:
            prompt = f"""Create a detailed music composition concept:
            
Genre: {genre}
Mood: {mood}
Purpose: {purpose}

Provide JSON response with:
1. "title": Song/piece title
2. "structure": Musical structure (intro, verse, chorus, etc.)
3. "instruments": List of recommended instruments
4. "tempo": BPM and tempo description
5. "key_signature": Musical key
6. "mood_progression": How the mood changes throughout
7. "technical_notes": Production and arrangement notes
8. "use_cases": Where this music would work best

Make it detailed and musically accurate."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional music composer and producer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            music_data = json.loads(response.choices[0].message.content)
            
            return {
                'success': True,
                'concept': music_data,
                'genre': genre,
                'mood': mood,
                'purpose': purpose,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Music concept generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'genre': genre
            }
    
    def generate_campaign_package(self, 
                                brand_info: Dict[str, Any],
                                campaign_goal: str,
                                target_audience: str) -> Dict[str, Any]:
        """
        Δημιουργεί πλήρες campaign package (κείμενα + εικόνες + βίντεο concepts)
        """
        try:
            package = {
                'campaign_overview': {},
                'text_content': {},
                'visual_content': {},
                'video_concepts': {},
                'generated_at': datetime.now().isoformat()
            }
            
            # 1. Generate campaign overview
            overview_specs = {
                'brand': brand_info.get('name', 'Brand'),
                'goal': campaign_goal,
                'audience': target_audience,
                'channels': ['social_media', 'email', 'website']
            }
            
            overview = self.generate_text_content('marketing', 'campaign_overview', overview_specs)
            package['campaign_overview'] = overview
            
            # 2. Generate social media posts
            social_specs = {
                'brand': brand_info.get('name', 'Brand'),
                'goal': campaign_goal,
                'audience': target_audience,
                'platforms': ['instagram', 'twitter', 'linkedin']
            }
            
            social_content = self.generate_text_content('marketing', 'social_posts', social_specs)
            package['text_content']['social_posts'] = social_content
            
            # 3. Generate email campaign
            email_specs = {
                'brand': brand_info.get('name', 'Brand'),
                'goal': campaign_goal,
                'audience': target_audience,
                'type': 'promotional'
            }
            
            email_content = self.generate_text_content('marketing', 'email_campaign', email_specs)
            package['text_content']['email_campaign'] = email_content
            
            # 4. Generate visual concepts
            visual_description = f"Professional marketing visual for {brand_info.get('name', 'brand')} campaign about {campaign_goal}, targeting {target_audience}"
            visual_concept = self.generate_image_concept(visual_description, "professional", {
                'brand': brand_info.get('name'),
                'campaign_goal': campaign_goal
            })
            package['visual_content']['main_visual'] = visual_concept
            
            # 5. Generate video concept
            video_concept = self.generate_video_concept(
                f"{brand_info.get('name', 'Brand')} - {campaign_goal}",
                "30s",
                target_audience
            )
            package['video_concepts']['main_video'] = video_concept
            
            return {
                'success': True,
                'package': package,
                'brand': brand_info.get('name', 'Brand'),
                'campaign_goal': campaign_goal,
                'components_generated': len([k for k in package.keys() if k != 'generated_at'])
            }
            
        except Exception as e:
            logger.error(f"Campaign package generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'brand': brand_info.get('name', 'Unknown')
            }
    
    def _build_content_prompt(self, category: str, template: str, specifications: Dict, user_context: Optional[Dict]) -> str:
        """
        Χτίζει το prompt για content generation
        """
        base_prompt = f"""Generate {category} content using the {template} template.

Specifications:
{json.dumps(specifications, indent=2)}
"""
        
        if user_context:
            base_prompt += f"""
User Context:
{json.dumps(user_context, indent=2)}
"""
        
        # Template-specific instructions
        template_instructions = {
            'social_post': "Create engaging social media posts with hashtags and calls to action. Include post variants for different platforms.",
            'blog_post': "Write a comprehensive blog post with introduction, main content, and conclusion. Include SEO-friendly headings.",
            'product_description': "Write compelling product descriptions that highlight benefits and features.",
            'email_campaign': "Create email subject line, body, and call-to-action. Make it personalized and engaging.",
            'ad_copy': "Write attention-grabbing ad copy with clear value proposition and strong call-to-action.",
            'landing_page': "Create landing page content with headline, subheadline, benefits, and conversion elements.",
            'campaign_overview': "Create comprehensive campaign strategy with objectives, messaging, and tactics.",
            'social_posts': "Generate multiple social media posts for different platforms with appropriate hashtags and engagement elements."
        }
        
        if template in template_instructions:
            base_prompt += f"\n\nSpecific Instructions: {template_instructions[template]}"
        
        base_prompt += """

Please provide your response in JSON format with the following structure:
{
  "title": "Content title or headline",
  "main_content": "The main content/copy",
  "additional_elements": ["hashtags", "call_to_action", "etc"],
  "variants": ["alternative versions if applicable"],
  "metadata": {
    "tone": "professional/casual/etc",
    "target_audience": "description",
    "key_messages": ["main points"]
  }
}

Make the content engaging, professional, and tailored to the specifications provided."""
        
        return base_prompt
    
    def _enhance_image_prompt(self, description: str, style: str, specifications: Optional[Dict]) -> str:
        """
        Βελτιώνει το image prompt για καλύτερα αποτελέσματα
        """
        style_modifiers = {
            'professional': 'clean, professional, corporate, high-quality',
            'creative': 'artistic, creative, unique, expressive',
            'minimalist': 'simple, clean, minimal, elegant',
            'modern': 'contemporary, sleek, modern design',
            'vintage': 'retro, vintage style, classic',
            'playful': 'fun, colorful, playful, engaging'
        }
        
        enhanced_prompt = description
        
        if style in style_modifiers:
            enhanced_prompt += f", {style_modifiers[style]}"
        
        if specifications:
            if 'colors' in specifications:
                enhanced_prompt += f", using colors: {', '.join(specifications['colors'])}"
            if 'mood' in specifications:
                enhanced_prompt += f", {specifications['mood']} mood"
            if 'format' in specifications:
                enhanced_prompt += f", suitable for {specifications['format']}"
        
        enhanced_prompt += ", high quality, professional photography style"
        
        return enhanced_prompt
    
    def get_content_categories(self) -> Dict[str, Any]:
        """
        Επιστρέφει διαθέσιμες κατηγορίες content
        """
        return self.content_categories
    
    def validate_content_request(self, category: str, template: str) -> Dict[str, Any]:
        """
        Επικυρώνει αίτημα για content generation
        """
        if category not in self.content_categories:
            return {
                'valid': False,
                'error': f"Category '{category}' not supported. Available: {list(self.content_categories.keys())}"
            }
        
        if template not in self.content_categories[category]['templates']:
            return {
                'valid': False,
                'error': f"Template '{template}' not available for category '{category}'. Available: {self.content_categories[category]['templates']}"
            }
        
        return {'valid': True}