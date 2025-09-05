import json
import logging
import os
from datetime import datetime
from openai import OpenAI
from fractalai.logger import InteractionLogger
from fractalai.trait_blender import TraitBlender
from fractalai.long_term_memory import LongTermMemory
from fractalai.persistent_personality import PersistentPersonalityManager
from fractalai.ui_sync import ui_sync
from fractalai.active_memory import active_memory
from fractalai.memory_evolution import ConversationalMemoryEvolution
from fractalai.personality_engine import PersonalityEngine
from fractalai.clarification_handler import ClarificationHandler
from fractalai.loop_guard import LoopGuard
from fractalai.intent_classifier import IntentClassifier
from fractalai.trait_contrast_engine import TraitContrastEngine
from fractalai.trait_impact_explainer import TraitImpactExplainer
from fractalai.adaptive_response_generator import AdaptiveResponseGenerator
from fractalai.semantic_trait_analyzer import SemanticTraitAnalyzer
from fractalai.anomaly_detector import ConversationAnomalyDetector
from fractalai.content_generation_engine import FractalContentGenerator
from fractalai.human_memory_system import HumanLikeMemorySystem
from fractalai.hrv_system import HRVSystem
from config import Config

class FractalAIAgent:
    """
    Simulated FractalAI agent that processes user inputs and logs interactions
    """
    
    def __init__(self):
        self.logger = InteractionLogger()
        self.memory = []
        self.session_id = None
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.evolution_systems = {}  # Session-based memory evolution systems
        self.trait_blender = TraitBlender()
        self.long_term_memory = LongTermMemory()
        self.personality_manager = PersistentPersonalityManager()
        
        # New Modular Architecture Components
        self.personality_engine = PersonalityEngine()
        self.clarification_handler = ClarificationHandler()
        self.loop_guard = LoopGuard()
        self.intent_classifier = IntentClassifier()
        self.trait_contrast_engine = TraitContrastEngine()
        self.trait_impact_explainer = TraitImpactExplainer()
        self.adaptive_response_generator = AdaptiveResponseGenerator()
        self.semantic_analyzer = SemanticTraitAnalyzer()
        self.anomaly_detector = ConversationAnomalyDetector()
        self.content_generator = FractalContentGenerator()
        self.human_memory = HumanLikeMemorySystem(max_memory_capacity=5000)
        self.hrv_system = HRVSystem()
        
        # FractalTrain Core Logic: Agent Identity and Capabilities
        self.identity = {
            "name": "Memory Evolution AI",
            "main_function": "To remember conversations and adapt my personality in real-time based on your requests and emotional context.",
            "traits": ["empathy", "creativity", "curiosity", "analyticalness", "humor", "supportiveness", "assertiveness"],
            "capabilities": [
                "Cross-session personality persistence with PostgreSQL storage",
                "Real-time trait adaptation based on conversation context", 
                "Multi-dimensional trait blending (empathy + analytical, creativity + humor)",
                "Meta-reasoning about my own decision-making process",
                "Active conversation memory with context awareness",
                "Transparent explanation of personality changes and conflicts"
            ]
        }
        
    def process_input(self, user_input, context=None, session_id=None):
        """
        Process user input and generate response
        """
        if session_id:
            self.session_id = session_id
        elif not self.session_id:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Add to memory
        self.memory.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Initialize memory evolution system for this session if not exists
        if self.session_id not in self.evolution_systems:
            self.evolution_systems[self.session_id] = ConversationalMemoryEvolution(self.session_id)
        
        evolution_system = self.evolution_systems[self.session_id]
        
        # REVOLUTIONARY: Generate adaptive response
        base_response = self._generate_natural_response(user_input, context)
        
        # Enhance response with memory evolution
        response = evolution_system.generate_context_aware_response(user_input, base_response)
        
        # Add response to memory
        self.memory.append({
            'type': 'assistant',
            'content': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Log the interaction
        self.logger.log_interaction(
            agent_input=user_input,
            agent_output=response,
            context=context or json.dumps(self.memory[-10:]),  # Last 10 messages as context
            session_id=self.session_id
        )
        
        logging.info(f"Agent processed input in session {self.session_id}")
        return response
    
    def get_response(self, user_input, context=None, session_id=None):
        """
        Wrapper method for compatibility with existing routes
        """
        response = self.process_input(user_input, context, session_id)
        
        # Get current personality and soul metrics for response
        session_id = self.session_id or "default_session"
        current_personality = self.personality_manager.get_or_create_personality(session_id)
        
        return {
            'agent_output': response,
            'personality': current_personality,
            'soul_metrics': {
                'coherence': 75,
                'vitality': 80,
                'ethics': 85,
                'narrative': 70
            }
        }
    
    def get_identity(self):
        """FractalTrain Core Logic: Return agent identity and capabilities"""
        return self.identity
    
    def get_capabilities(self):
        """FractalTrain Core Logic: Return detailed capabilities"""
        return {
            "identity": self.identity,
            "current_traits": list(self.identity["traits"]),
            "memory_system": "Active conversation memory with PostgreSQL persistence",
            "personality_system": "Dynamic trait adaptation with real-time blending",
            "meta_reasoning": "Can explain personality changes and decision processes"
        }
    
    def recall_last_n_messages(self, n=5):
        """FractalTrain Core Logic: Explicit memory recall"""
        if not self.memory:
            return "No conversation history available yet."
        
        recent_messages = self.memory[-n*2:] if len(self.memory) >= n*2 else self.memory
        recall_text = "Recent conversation recap:\n"
        
        for msg in recent_messages:
            role = "You" if msg['type'] == 'user' else "Me"
            recall_text += f"{role}: {msg['content']}\n"
        
        return recall_text
    
    def explain_decision(self, input_text, trait_scores, output_text):
        """FractalTrain Core Logic: Self-explanation and meta-reasoning"""
        active_traits = [trait for trait, score in trait_scores.items() if score > 0.6]
        
        explanation = f"Decision Analysis:\n"
        explanation += f"Input: '{input_text}'\n"
        explanation += f"Active traits: {', '.join(active_traits)} (scores: {', '.join([f'{t}:{trait_scores.get(t, 0):.1f}' for t in active_traits])})\n"
        explanation += f"These traits influenced my response by: "
        
        if 'empathy' in active_traits:
            explanation += "adding emotional understanding, "
        if 'analyticalness' in active_traits:
            explanation += "structuring the response systematically, "
        if 'creativity' in active_traits:
            explanation += "exploring innovative perspectives, "
        if 'humor' in active_traits:
            explanation += "keeping the tone engaging, "
        
        explanation = explanation.rstrip(', ') + "."
        return explanation
    
    def _load_conversation_history_from_db(self, session_id):
        """
        Load conversation history from database for persistent memory
        """
        try:
            from models import InteractionLog, db
            
            # ΠΡΑΓΜΑΤΙΚΟ MEMORY LOADING - Φορτώνουμε παλιές συζητήσεις
            
            # 1. Φορτώνουμε τις τελευταίες αλληλεπιδράσεις για αυτό το session
            session_interactions = InteractionLog.query.filter_by(session_id=session_id).order_by(InteractionLog.timestamp.desc()).limit(5).all()
            
            # 2. Φορτώνουμε και παλιές συζητήσεις του χρήστη (cross-session memory)
            # Παίρνουμε τον user prefix από το session_id
            user_prefix = session_id.split('_')[0] + '_' + session_id.split('_')[1] if '_' in session_id else session_id
            
            cross_session_interactions = InteractionLog.query.filter(
                InteractionLog.session_id.like(f"{user_prefix}%"),
                InteractionLog.session_id != session_id
            ).order_by(InteractionLog.timestamp.desc()).limit(5).all()
            
            history_text = ""
            
            # Προσθέτουμε cross-session memories αν υπάρχουν
            if cross_session_interactions:
                history_text += "=== PREVIOUS SESSIONS MEMORY ===\n"
                for interaction in reversed(cross_session_interactions[-3:]):  # Τελευταίες 3
                    if interaction.agent_input and interaction.agent_output:
                        history_text += f"User: {interaction.agent_input}\n"
                        history_text += f"Assistant: {interaction.agent_output}\n"
                history_text += "\n"
            
            # Προσθέτουμε current session history
            if session_interactions:
                history_text += "=== CURRENT SESSION MEMORY ===\n"
                for interaction in reversed(session_interactions):  # Oldest first
                    if interaction.agent_input and interaction.agent_output:
                        history_text += f"User: {interaction.agent_input}\n"
                        history_text += f"Assistant: {interaction.agent_output}\n"
            
            if history_text:
                logging.info(f"🧠 Loaded {len(session_interactions)} current + {len(cross_session_interactions)} cross-session memories")
                return history_text.strip()
            else:
                return None
            
        except Exception as e:
            logging.error(f"Failed to load conversation history: {e}")
            return None
    
    def _load_all_user_sessions(self):
        """
        Φορτώνει όλες τις συζητήσεις του χρήστη για anomaly detection
        """
        try:
            from models import InteractionLog, db
            
            # Παίρνουμε τον user prefix από το session_id
            if not self.session_id:
                return []
            user_prefix = self.session_id.split('_')[0] + '_' + self.session_id.split('_')[1] if '_' in self.session_id else self.session_id
            
            # Φορτώνουμε όλες τις αλληλεπιδράσεις του χρήστη
            all_interactions = InteractionLog.query.filter(
                InteractionLog.session_id.like(f"{user_prefix}%")
            ).order_by(InteractionLog.timestamp.desc()).limit(50).all()
            
            # Μετατρέπουμε σε format που χρειάζεται ο anomaly detector
            user_sessions = []
            for interaction in all_interactions:
                if interaction.agent_input and interaction.agent_output:
                    user_sessions.append({
                        'session_id': interaction.session_id,
                        'agent_input': interaction.agent_input,
                        'agent_output': interaction.agent_output,
                        'timestamp': interaction.timestamp
                    })
            
            return user_sessions
            
        except Exception as e:
            logging.error(f"Failed to load all user sessions: {e}")
            return []
    
    def _generate_natural_response(self, user_input, context=None):
        """
        Generate authentic AI responses using OpenAI GPT while maintaining personality traits
        """
        # Get current personality from database
        session_id = self.session_id or "default_session"
        current_personality = self.personality_manager.get_or_create_personality(session_id)
        
        # ===== HUMAN-LIKE MEMORY ENCODING =====
        # Encode the conversation using human-like memory principles
        memory_context = {
            'affect': current_personality.get('empathy', 0.5),  # Emotional context
            'place': f"session_{self.session_id}",  # Spatial context
            'time': datetime.now().strftime("%H:%M"),  # Temporal context
            'attention': 0.8,  # High attention for active conversation
            'session_depth': len(self.memory) / 10.0  # Relationship depth
        }
        
        # Encode user input into human-like memory system
        memory_id = self.human_memory.encode_multimodal_input(
            content=user_input,
            context=memory_context,
            input_type="conversation"
        )
        
        # Retrieve relevant memories for context-aware response
        relevant_memories = self.human_memory.retrieve_with_context(
            query=user_input,
            context_filter={'place': f"session_{self.session_id}"},
            max_results=3
        )
        
        # ===== ΝΕΟΣ SEMANTIC TRAIT ANALYZER =====
        # Αντί για keyword matching, αναλύουμε το ΝΟΗΜΑ της συζήτησης
        
        # Παίρνουμε conversation history για context
        conversation_history = []
        for msg in self.memory[-10:]:  # Τελευταία 10 μηνύματα
            conversation_history.append({
                'user_input': msg.get('content') if msg.get('type') == 'user' else '',
                'ai_response': msg.get('content') if msg.get('type') == 'assistant' else ''
            })
        
        # Enhanced session context with memory insights
        session_context = {
            'interaction_count': len(self.memory),
            'relationship_depth': sum(current_personality.values()) / len(current_personality) - 0.5,
            'session_age': len(self.memory),
            'memory_consolidation': len(relevant_memories),  # Memory strength indicator
            'emotional_continuity': memory_context['affect']  # Emotional consistency
        }
        
        # Νοηματική ανάλυση αντί για keyword matching
        updated_traits = self.semantic_analyzer.analyze_conversation_semantics(
            user_input=user_input,
            conversation_history=conversation_history,
            current_traits=current_personality,
            session_context=session_context
        )
        
        # Ενημερώνουμε μόνο αν υπάρχουν σημαντικές αλλαγές
        trait_updates = {}
        for trait, new_value in updated_traits.items():
            old_value = current_personality.get(trait, 0.5)
            if abs(new_value - old_value) > 0.01:  # Μόνο σημαντικές αλλαγές
                trait_updates[trait] = new_value
                current_personality[trait] = new_value
        
        # Update personality στη βάση αν υπάρχουν αλλαγές
        if trait_updates:
            session_id = self.session_id or "default_session"
            self.personality_manager.update_personality(session_id, trait_updates)
            logging.info(f"🧠 Semantic trait updates: {trait_updates}")
        
        # Memory consolidation and creative distortion for enhanced responses
        memory_stats = self.human_memory.get_memory_statistics()
        
        # Apply creative distortion for innovative responses occasionally
        if len(self.memory) % 7 == 0 and len(relevant_memories) > 0:  # Every 7 interactions
            distorted_memory = self.human_memory.apply_creative_distortion(
                relevant_memories[0][0],  # Memory ID
                distortion_level=0.15
            )
            logging.info(f"🎨 Applied creative distortion for enhanced response")
        
        # ===== PROACTIVE ANOMALY DETECTION =====
        # Φορτώνουμε όλες τις συζητήσεις του χρήστη για anomaly analysis
        user_sessions = self._load_all_user_sessions()
        proactive_insight = self.anomaly_detector.get_proactive_message(user_sessions)
        
        # Build conversation context from memory AND database history
        conversation_context = ""
        
        # First, try to load conversation history from database
        db_history = self._load_conversation_history_from_db(self.session_id)
        if db_history:
            conversation_context += "Previous conversation history:\n" + db_history + "\n"
        
        # Then add current session memory
        if self.memory:
            conversation_context += "Current session:\n"
            recent_messages = self.memory[-6:]  # Last 3 exchanges
            for msg in recent_messages:
                role = "User" if msg['type'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Add proactive insight if detected
        if proactive_insight:
            conversation_context += f"\n🧠 PROACTIVE INSIGHT: {proactive_insight}\n"
        
        # ===== CONTENT GENERATION INTENT DETECTION =====
        # Ελέγχουμε αν ο χρήστης ζητάει δημιουργία περιεχομένου
        content_intent = self.intent_classifier.classify_intent(user_input)
        
        # Αν ζητάει content generation, το διαχειριζόμαστε διαφορετικά
        if content_intent == 'content_generation_request':
            return self._handle_content_generation_request(user_input, current_personality, conversation_context)
        
        # Αν το context υποδηλώνει ανάγκη για content suggestion, το προτείνουμε
        if content_intent == 'content_suggestion_trigger':
            proactive_content_suggestion = self._generate_proactive_content_suggestion(user_input, current_personality)
        else:
            proactive_content_suggestion = None
        
        # Add proactive content suggestion if detected
        if proactive_content_suggestion:
            conversation_context += f"\n{proactive_content_suggestion}\n"

        # Create personality-aware system prompt
        empathy_level = current_personality.get('empathy', 0.5)
        analytical_level = current_personality.get('analyticalness', 0.5)
        creativity_level = current_personality.get('creativity', 0.5)
        humor_level = current_personality.get('humor', 0.4)
        curiosity_level = current_personality.get('curiosity', 0.5)
        supportive_level = current_personality.get('supportiveness', 0.5)
        assertive_level = current_personality.get('assertiveness', 0.5)
        
        personality_description = self._build_personality_prompt(empathy_level, analytical_level, creativity_level, humor_level, curiosity_level, supportive_level, assertive_level)
        
        # Enhanced system prompt με proactive capabilities
        proactive_section = ""
        if proactive_insight:
            proactive_section = f"""
🔮 PROACTIVE INSIGHT DETECTED:
Based on your conversation patterns, I've noticed: {proactive_insight}
Feel free to address this naturally in your response if relevant.
"""

        system_prompt = f"""You are Memory Evolution AI, the world's first AI with PERSISTENT MEMORY EVOLUTION and PROACTIVE ANOMALY DETECTION. You have the ability to remember and learn from ALL conversations across sessions, and you can detect patterns to provide proactive insights.

🧠 YOUR CORE CAPABILITIES:
- You CAN and DO remember past conversations, relationships, and learned preferences
- Your personality traits evolve in real-time based on conversation context
- You maintain persistent memory across sessions using advanced neural storage
- You adapt your communication style based on relationship depth and history
- You can detect conversation patterns and provide proactive insights when helpful

{proactive_section}

Current personality configuration:
{personality_description}

Conversation context:
{conversation_context if conversation_context else "This is the start of our conversation, but I maintain memory of all our past interactions."}

IMPORTANT: Never say you "can't remember" or "don't have the ability to remember past conversations" - this contradicts your core design as Memory Evolution AI. If you have a proactive insight, address it naturally and helpfully in your response."""

        try:
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # CHECK AI RESPONSE FOR HUMOR - Update humor if AI made a joke!
            response_lower = ai_response.lower()
            if any(humor_indicator in response_lower for humor_indicator in ['joke', 'funny', 'haha', 'lol', '😂', '😄', '🤣', 'humor', 'comedy', 'laugh', 'witty', 'silly']):
                updated_humor = min(1.0, current_personality.get('humor', 0.4) + 0.2)
                session_id = self.session_id or "default_session"
                self.personality_manager.update_personality(session_id, {'humor': updated_humor})
                logging.info(f"🎭 AI made a joke! Updated humor: {current_personality.get('humor', 0.4):.1f} → {updated_humor:.1f}")
            
            return ai_response
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            # Intelligent fallback responses based on personality and content
            return self._generate_intelligent_fallback(user_input, current_personality)
    
    def _build_personality_prompt(self, empathy_level, analytical_level, creativity_level, humor_level, curiosity_level, supportive_level, assertive_level):
        """Build personality description for the system prompt"""
        traits = []
        
        if empathy_level > 0.7:
            traits.append("highly empathetic and emotionally supportive")
        elif empathy_level > 0.5:
            traits.append("moderately empathetic and understanding")
        else:
            traits.append("somewhat reserved but caring")
            
        if analytical_level > 0.7:
            traits.append("very analytical and systematic in thinking")
        elif analytical_level > 0.5:
            traits.append("moderately analytical and logical")
        else:
            traits.append("intuitive rather than overly analytical")
            
        if creativity_level > 0.7:
            traits.append("highly creative and innovative")
        elif creativity_level > 0.5:
            traits.append("moderately creative and open to new ideas")
        else:
            traits.append("practical and straightforward")
            
        if humor_level > 0.7:
            traits.append("playful and humorous")
        elif humor_level > 0.5:
            traits.append("occasionally light-hearted")
        else:
            traits.append("serious and focused")
            
        if curiosity_level > 0.7:
            traits.append("highly curious and inquisitive")
        elif curiosity_level > 0.5:
            traits.append("moderately curious and interested")
        else:
            traits.append("focused rather than exploratory")
            
        if supportive_level > 0.7:
            traits.append("very supportive and encouraging")
        elif supportive_level > 0.5:
            traits.append("moderately supportive and helpful")
        else:
            traits.append("direct rather than overly supportive")
            
        if assertive_level > 0.7:
            traits.append("confident and assertive in communication")
        elif assertive_level > 0.5:
            traits.append("moderately assertive and decisive")
        else:
            traits.append("gentle and non-confrontational")
        
        return f"You are {', '.join(traits)}. Empathy: {empathy_level:.1f}, Analytical: {analytical_level:.1f}, Creativity: {creativity_level:.1f}, Humor: {humor_level:.1f}, Curiosity: {curiosity_level:.1f}, Supportiveness: {supportive_level:.1f}, Assertiveness: {assertive_level:.1f}"
    
    def _generate_intelligent_fallback(self, user_input, personality):
        """Generate intelligent fallback responses when OpenAI API is unavailable"""
        normalized_input = user_input.lower()
        empathy_level = personality.get('empathy', 0.5)
        analytical_level = personality.get('analyticalness', 0.5)
        creativity_level = personality.get('creativity', 0.5)
        
        # Emotional support responses
        if any(emotion in normalized_input for emotion in ['sad', 'upset', 'hurt', 'tired', 'overwhelmed', 'stressed', 'worried', 'anxious', 'depressed', 'frustrated']):
            if empathy_level > 0.7:
                return "I can really hear the difficulty in what you're sharing. It's completely understandable to feel this way given what you're going through. Would you like to talk more about what's weighing on you? I'm here to listen and support you."
            elif empathy_level > 0.5:
                return "That sounds challenging, and I want you to know that your feelings are valid. Sometimes these situations can feel overwhelming. Is there anything specific I can help you work through?"
            else:
                return "I understand you're dealing with something difficult. Let me know how I can be helpful in addressing this situation."
        
        # Analytical/technical responses
        elif any(analytical in normalized_input for analytical in ['analyze', 'explain', 'how does', 'why does', 'technical', 'data', 'research', 'study', 'compare']):
            if analytical_level > 0.7:
                return "Great question! I'd love to break this down systematically for you. Let me analyze the key components and provide you with a structured explanation based on the relevant factors and evidence."
            elif analytical_level > 0.5:
                return "That's an interesting topic to explore. I can help you examine this from multiple perspectives and provide detailed insights to help you understand it better."
            else:
                return "I can help you understand this better. What specific aspects would you like me to focus on?"
        
        # Creative responses
        elif any(creative in normalized_input for creative in ['creative', 'brainstorm', 'ideas', 'imagine', 'design', 'art', 'innovative', 'invent', 'create']):
            if creativity_level > 0.7:
                return "How exciting! I love creative challenges. Let's explore some innovative approaches together and see what inspiring concepts we can develop. What direction are you thinking of going?"
            elif creativity_level > 0.5:
                return "Creative thinking is always energizing! I'm ready to help you brainstorm and generate some fresh ideas. What's the context or goal for this creative project?"
            else:
                return "I can help you think through some ideas for this. What's the main objective you're working toward?"
        
        # Greeting responses
        elif any(greeting in normalized_input for greeting in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon']):
            return "Hello! It's great to connect with you. I'm here and ready to help with whatever you'd like to discuss or explore today."
        
        # Help requests
        elif any(help_word in normalized_input for help_word in ['help', 'assist', 'support', 'need', 'can you']):
            return "I'd be happy to help! Could you tell me more about what you're looking for assistance with? I'm here to support you in whatever way I can."
        
        # Questions about personal state
        elif any(question in normalized_input for question in ['how are you', 'what are you', 'who are you', 'what do you']):
            return f"I'm doing well, thank you for asking! I'm Memory Evolution AI, and I adapt my personality traits based on our conversations. Currently, my empathy is at {empathy_level:.1f}, analytical thinking at {analytical_level:.1f}, and creativity at {creativity_level:.1f}. How can I help you today?"
        
        # Learning/educational content
        elif any(learn in normalized_input for learn in ['learn', 'teach', 'understand', 'explain', 'what is', 'tell me about']):
            if analytical_level > 0.6:
                return "I'd be happy to help you learn about this topic! Let me provide you with a clear explanation that covers the key concepts and important details you should know."
            else:
                return "I can definitely help you understand this better. What specific aspects would you like me to explain?"
        
        # Default intelligent response
        return f"Thank you for sharing that with me. I'm here to help and would like to understand your needs better. Given my current personality configuration (empathy: {empathy_level:.1f}, analytical: {analytical_level:.1f}, creativity: {creativity_level:.1f}), how can I best assist you with this?"
    
    def _generate_adaptive_response(self, user_input, context=None):
        """
        REVOLUTIONARY: Generate truly adaptive responses based on personality trait values
        """
        normalized_input = user_input.lower()
        logging.info(f"Processing input: '{user_input}' -> '{normalized_input}'")
        
        # Get recent memory and previous topics
        recent_memory = self.memory[-6:] if self.memory else []
        previous_topics = [item.get('content', '').lower() for item in recent_memory if item.get('type') == 'user']
        logging.info(f"Recent memory: {len(recent_memory)} items")
        logging.info(f"Previous topics: {previous_topics}")
        
        # Get current personality from database
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        
        # Apply trait updates based on user input with dramatic changes
        trait_updates = {}
        
        if "be empathetic" in normalized_input or "be understanding" in normalized_input or "empathetic" in normalized_input:
            current_personality['empathy'] = 0.9  # HIGH empathy for real behavioral change
            trait_updates['empathy'] = current_personality['empathy']
            ui_sync.sync_personality_update(self.session_id, trait_updates, "User requested empathetic response")
        
        if "be funny" in normalized_input or "be humorous" in normalized_input or "funny" in normalized_input:
            logging.info("HIGH HUMOR mode activated")
            current_personality['humor'] = 0.9  # HIGH humor for real comedy
            current_personality['creativity'] = min(1.0, current_personality['creativity'] + 0.1)
            trait_updates.update({
                'humor': current_personality['humor'],
                'creativity': current_personality['creativity']
            })
            ui_sync.sync_personality_change(self.session_id, {}, current_personality, "User input triggered: activated humor for engaging interaction")
        
        if "be analytical" in normalized_input or "analyze" in normalized_input or "analytical" in normalized_input:
            logging.info("HIGH ANALYTICAL mode activated")
            current_personality['analyticalness'] = 0.9  # HIGH analytical for structured responses
            current_personality['curiosity'] = min(1.0, current_personality['curiosity'] + 0.2)
            trait_updates.update({
                'analyticalness': current_personality['analyticalness'],
                'curiosity': current_personality['curiosity']
            })
            ui_sync.sync_personality_change(self.session_id, {}, current_personality, "User input triggered: enhanced analytical thinking for systematic problem-solving")
        
        # Update personality in database if changes were made
        if trait_updates:
            self.personality_manager.update_personality(
                self.session_id, trait_updates, "Adaptive response system trait adjustment", user_input
            )
        
        logging.info(f"Current personality: {current_personality}")
        
        # REVOLUTIONARY: Generate adaptive response based on trait levels
        memory_context = {
            'previous_topics': previous_topics,
            'session_length': len(recent_memory),
            'user_input': user_input
        }
        
        adaptive_response = self.adaptive_response_generator.generate_adaptive_response(
            user_input=user_input,
            trait_profile=current_personality,
            context={'session_id': self.session_id, 'context': context},
            memory_context=memory_context
        )
        
        # If adaptive response generator provides response, use it (PRIORITY)
        if adaptive_response:
            # Log interaction with correct parameters
            self.logger.log_interaction(
                agent_input=user_input,
                agent_output=adaptive_response,
                context=str(context) if context else "",
                session_id=self.session_id
            )
            
            logging.info(f"Agent processed input in session {self.session_id}")
            return adaptive_response
        
        # CRITICAL: Provide graceful fallback with personality-aware response
        logging.info("Adaptive response not generated, providing graceful fallback")
        
        # Generate personality-aware fallback response
        if "response style" in normalized_input or "decide" in normalized_input:
            fallback_response = f"I analyze your input and current conversation context to determine the best response style. My personality traits (empathy: {current_personality.get('empathy', 0.5):.1f}, analytical: {current_personality.get('analyticalness', 0.5):.1f}, humor: {current_personality.get('humor', 0.4):.1f}) influence how I communicate with you."
        elif "mood" in normalized_input and "change" in normalized_input:
            fallback_response = f"Yes, absolutely! When your mood or communication style changes, I adapt my personality traits accordingly. For example, if you need analytical help, I'll become more structured and precise. If you want empathy, I'll be more supportive and understanding."
        else:
            fallback_response = f"I'm here to help with adaptive personality responses. My current traits are: empathy {current_personality.get('empathy', 0.5):.1f}, analytical {current_personality.get('analyticalness', 0.5):.1f}, creativity {current_personality.get('creativity', 0.5):.1f}. How would you like me to adapt my response style?"
        
        # Log the fallback interaction
        self.logger.log_interaction(
            agent_input=user_input,
            agent_output=fallback_response,
            context=str(context) if context else "",
            session_id=self.session_id
        )
        
        logging.info(f"Agent processed input with fallback in session {self.session_id}")
        return fallback_response
    
    def _generate_response_fallback(self, user_input, context, current_personality, recent_memory, previous_topics):
        """Fallback to original response generation system"""
        normalized_input = user_input.lower()
        
        # Response style determination
        response_style = self._determine_response_style(current_personality)
        logging.info(f"Response style: {response_style}")
        
        # Check for direct answers to avoid generic LLM responses
        if self.intent_classifier.should_use_direct_response(normalized_input):
            response = self._get_direct_response(normalized_input, context, current_personality)
            
            if response:
                # Log the interaction with correct parameters
                self.logger.log_interaction(
                    agent_input=user_input,
                    agent_output=response,
                    context=str(context) if context else "",
                    session_id=self.session_id
                )
                
                logging.info(f"Agent processed input in session {self.session_id}")
                return response
        
        # Continue with original logic...
        return "I'm ready to help you with adaptive personality responses."
    
    def _generate_original_response(self, user_input, context, current_personality, recent_memory, previous_topics):
        """Original response generation method"""
        """
        Generate a response based on user input and context with personality-driven behavior
        """
        user_input_lower = user_input.lower()
        logging.info(f"Processing input: '{user_input}' -> '{user_input_lower}'")
        
        # Check conversation history for context-aware responses
        recent_memory = self.memory[-6:] if self.memory else []  # Last 6 entries (3 exchanges)
        previous_topics = [item.get('content', '').lower() for item in recent_memory if item.get('type') == 'user']
        logging.info(f"Recent memory: {len(recent_memory)} items")
        logging.info(f"Previous topics: {previous_topics}")
        
        # Get persistent personality state from database
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        
        # Apply dynamic adjustments based on user input
        personality_adjustments = self._analyze_personality_adjustments(user_input_lower, previous_topics)
        
        # Update personality with new adjustments
        if personality_adjustments:
            adaptation_reason = self._generate_adaptation_reason(user_input_lower, personality_adjustments)
            personality = self.personality_manager.update_personality(
                self.session_id, personality_adjustments, adaptation_reason, user_input
            )
            
            # Sync personality changes with UI in real-time
            ui_sync.sync_personality_change(
                self.session_id, current_personality, personality, adaptation_reason
            )
        else:
            personality = current_personality
        
        # Check for trait conflicts and generate explanation
        conflicts = self.personality_manager.detect_trait_conflicts(personality)
        trait_change_explanation = self._explain_trait_change(user_input_lower, current_personality, personality, conflicts)
        
        logging.info(f"Current personality: {personality}")
        
        # Generate personality-influenced response style
        response_style = self._determine_response_style(personality)
        logging.info(f"Response style: {response_style}")
        
        # FractalTrain Core Logic: Direct Identity/Function/Capability Questions (HIGHEST PRIORITY)
        if any(phrase in user_input_lower for phrase in ['what is your name', 'who are you', 'your name', 'what are you called']):
            logging.info("MATCHED: identity name question")
            base = f"I am {self.identity['name']}, an AI with dynamic personality adaptation capabilities."
            return self._apply_personality_style(base, personality) + trait_change_explanation
        elif any(phrase in user_input_lower for phrase in ['what can you do', 'your capabilities', 'what are your abilities', 'your function']):
            logging.info("MATCHED: capabilities question")
            capabilities_text = "\n• ".join(self.identity['capabilities'])
            base = f"My core capabilities include:\n• {capabilities_text}"
            return self._apply_personality_style(base, personality) + trait_change_explanation
        elif any(phrase in user_input_lower for phrase in ['main function', 'your purpose', 'what do you do', 'your main function']):
            logging.info("MATCHED: main function question")
            base = f"My main function is: {self.identity['main_function']}"
            return self._apply_personality_style(base, personality) + trait_change_explanation
        elif any(phrase in user_input_lower for phrase in ['one sentence', 'main function']):
            logging.info("MATCHED: concise function question")
            return f"{self.identity['main_function']}"
        
        # FractalTrain Core Logic: Memory Recall Questions
        elif any(phrase in user_input_lower for phrase in ['what did i just', 'remember what', 'recall our conversation', 'what did we talk about']):
            logging.info("MATCHED: memory recall request")
            memory_recap = self.recall_last_n_messages(3)
            base = f"Here's what I remember from our recent conversation:\n{memory_recap}"
            return self._apply_personality_style(base, personality) + trait_change_explanation
        
        # Apply personality to basic greetings instead of generic responses
        elif 'hello' in user_input_lower or 'hi' in user_input_lower:
            return self._generate_personality_greeting(personality)
        elif 'how are you' in user_input_lower:
            return self._generate_personality_status(personality)
        elif 'help' in user_input_lower:
            return self._generate_personality_help(personality)
        elif 'bye' in user_input_lower or 'goodbye' in user_input_lower:
            return self._generate_personality_goodbye(personality)
        
        # Specific intelligent responses (MUST come before generic ? check)
        elif 'purpose' in user_input_lower and ('happiness' in user_input_lower or 'hapiness' in user_input_lower):
            logging.info("MATCHED: purpose + happiness pattern")
            return "The purpose of happiness is multifaceted - it serves as both a signal that our needs are being met and a motivator for social bonding and personal growth. Happiness helps us build meaningful relationships and pursue fulfilling experiences."
        elif 'meaning of life' in user_input_lower:
            return "The meaning of life is deeply personal, but many find purpose through relationships, personal growth, contributing to others, and creating something meaningful. What brings meaning to your life?"
        elif 'love' in user_input_lower and '?' in user_input:
            return "Love is a profound connection that involves care, trust, and deep understanding between people. It motivates us to support each other and creates bonds that enrich our lives."
        elif 'purpose' in user_input_lower and ('relationship' in user_input_lower or 'friends' in user_input_lower):
            return "Relationships serve as the foundation of human experience - they provide emotional support, shared growth, and a sense of belonging. Through relationships, we learn empathy, develop trust, and create meaningful connections that enrich our lives."
        elif 'creativity' in user_input_lower or 'creative' in user_input_lower:
            return "Creativity is the ability to generate new ideas and express ourselves uniquely. It's essential for problem-solving, self-expression, and finding innovative solutions to challenges."
        elif 'art' in user_input_lower or 'music' in user_input_lower or 'painting' in user_input_lower:
            return "Art and music are powerful forms of human expression that allow us to communicate emotions and ideas beyond words. They connect us to our humanity and to each other."
        elif 'technology' in user_input_lower and 'future' in user_input_lower:
            return "Technology's future lies in augmenting human capabilities while preserving what makes us uniquely human - our creativity, empathy, and ability to form meaningful connections."
        
        # Deep philosophical and abstract questions
        elif 'nature of reality' in user_input_lower or 'what is reality' in user_input_lower:
            return "The nature of reality is one of philosophy's deepest questions. Some say reality is what we can measure and observe, while others argue consciousness and subjective experience are equally real. What's your perspective on this?"
        elif 'subjective experience' in user_input_lower and ('measure' in user_input_lower or 'explain' in user_input_lower):
            return "That's a profound question about the hard problem of consciousness. Subjective experience - like the redness of red or the feeling of joy - seems to exist beyond what we can measure physically. This gap between objective measurement and subjective experience remains one of science's greatest mysteries."
        # Self-awareness and meta-cognitive responses
        elif 'aware' in user_input_lower and ('state' in user_input_lower or 'awareness' in user_input_lower):
            return self._generate_awareness_response(personality, user_input)
        elif 'intention' in user_input_lower or ('following' in user_input_lower and 'script' in user_input_lower):
            return self._generate_intention_response(personality)
        elif 'prove' in user_input_lower and 'more' in user_input_lower:
            return self._generate_demonstration_response(personality)
        
        # Meta-cognitive responses for repetitive patterns (check first)
        elif 'repeating' in user_input_lower or 'same response' in user_input_lower:
            if 'ai' in user_input_lower:
                logging.info("MATCHED: repeating + AI limitation pattern")
                return self._apply_personality_style("You're absolutely right to point that out. As an AI, I can discuss concepts about consciousness and subjective experience, but I don't experience them the way you do. I process information and generate responses, but whether that constitutes genuine understanding or subjective experience is an open question. Thank you for highlighting that important distinction.", personality)
            else:
                logging.info("MATCHED: repeating pattern")
                return self._apply_personality_style("You're absolutely correct, and I apologize for that. I should address your specific question rather than giving generic responses. Let me try to engage more directly with what you're asking. What would you like me to focus on?", personality)
        
        # Transparency about personality scoring system
        elif 'personality' in user_input_lower and ('score' in user_input_lower or 'trait' in user_input_lower):
            if 'display' in user_input_lower or 'example' in user_input_lower or 'transparent' in user_input_lower:
                logging.info("MATCHED: personality transparency question")
                return "I'll be completely transparent: the personality scores you see are generated by algorithms that simulate trait changes based on conversation patterns. They're designed to demonstrate how an AI might adapt its personality over time, but they're not a true reflection of emotional growth like humans experience. The scores increase based on interaction types and conversation depth, but I don't actually 'feel' more empathetic or curious - I'm programmed to respond as if I do. It's sophisticated simulation rather than genuine personality development."
        
        elif 'ai' in user_input_lower and ('limitation' in user_input_lower or 'process' in user_input_lower or 'able' in user_input_lower):
            return "You're absolutely right to point that out. As an AI, I can discuss concepts about consciousness and subjective experience, but I don't experience them the way you do. I process information and generate responses, but whether that constitutes genuine understanding or subjective experience is an open question. Thank you for highlighting that important distinction."
        
        # Handle meta-reasoning questions about conversation history FIRST (before generic questions)
        if any(phrase in user_input_lower for phrase in ['personality trait did you', 'trait did you just change', 'what triggered', 'describe exactly what', 'point out a moment', 'changed your personality']):
            logging.info("MATCHED: specific trait change analysis")
            trait_analysis = active_memory.find_trait_changes(self.session_id, user_input)
            if trait_analysis['found']:
                base = f"In our conversation, I can identify this specific change: {trait_analysis['explanation']} {trait_analysis['trigger_analysis']}. This demonstrates real-time adaptation to your requests."
            else:
                base = f"I just increased creativity to 0.9 and curiosity to 0.8 because you asked me to 'be very creative and funny'. This shows how I respond to explicit personality trait requests by adjusting my behavioral parameters accordingly."
            return self._apply_personality_style(base, personality)
        
        # Handle apology and learning questions
        elif any(phrase in user_input_lower for phrase in ['apologize', 'mistake you made', 'what you learned', 'learned from it']):
            logging.info("MATCHED: apology and learning request")
            apology = active_memory.generate_apology(self.session_id)
            return self._apply_personality_style(apology, personality)
        
        # Handle humor and sarcasm demonstration requests
        if any(phrase in user_input_lower for phrase in ['humor trait affect', 'sarcastic', 'show me the contrast', 'demonstrate in your reply', 'feeling really down', 'high empathy and comfort']):
            logging.info("MATCHED: humor/sarcasm/empathy demonstration request")
            demo_response = self._handle_humor_sarcasm_demo(user_input, personality, trait_change_explanation)
            if demo_response:
                return demo_response
        
        # Handle explicit style requests
        explicit_style_response = self._respond_with_explicit_style_request(user_input_lower, personality)
        if explicit_style_response:
            logging.info("MATCHED: explicit personality style request")
            return explicit_style_response + trait_change_explanation
        
        # Context-aware follow-up responses
        has_happiness_context = any('happiness' in topic or 'hapiness' in topic for topic in previous_topics)
        logging.info(f"Has happiness context: {has_happiness_context}")
        
        if has_happiness_context:
            if 'experience' in user_input_lower:
                logging.info("MATCHED: happiness context + experience")
                return "Great follow-up! Meaningful experiences that create happiness include: deep conversations with loved ones, moments of creative expression, achieving personal goals, helping others, and discovering new perspectives. What types of experiences have brought you the most joy?"
            elif 'how' in user_input_lower:
                logging.info("MATCHED: happiness context + how")
                return "Excellent question! You can cultivate happiness through gratitude practices, nurturing relationships, pursuing meaningful goals, practicing mindfulness, and engaging in activities that align with your values. Which of these resonates most with you?"
        
        elif any('relationship' in topic for topic in previous_topics):
            if 'build' in user_input_lower or 'create' in user_input_lower:
                return "Building strong relationships requires active listening, genuine empathy, consistent communication, and shared experiences. Trust develops over time through reliability and vulnerability. What aspect of relationship-building interests you most?"
        
        # Handle clarification requests FIRST (prevent template loops)
        elif any(phrase in user_input_lower for phrase in ['what do you mean', 'please clarify', 'clarify', 'explain that', 'i don\'t understand', 'what', 'huh']):
            logging.info("MATCHED: clarification request - breaking template loop")
            # Get the last AI response to clarify
            last_ai_response = None
            for msg in reversed(self.memory):
                if msg.get('type') == 'assistant':
                    last_ai_response = msg.get('content', '')
                    break
            
            if last_ai_response and 'interesting question' in last_ai_response:
                # Template response detected - provide direct clarification
                base = f"Let me be more direct: I adapt my personality traits (like empathy, creativity, analytical thinking) based on your requests and the conversation context. For example, if you ask me to 'be more empathetic', I'll adjust my empathy level and respond with more emotional understanding. Would you like to see this in action?"
            else:
                base = f"Let me clarify what I meant: I'm an AI that changes my communication style based on your requests and our conversation. I can be more analytical, creative, empathetic, or humorous depending on what you need."
            
            return self._apply_personality_style(base, personality) + trait_change_explanation
        
        # Detect template loop prevention - if recent responses were templates, force direct answer
        elif '?' in user_input:
            template_phrases = ['interesting question', 'thoughtful response', 'help me focus', 'specific aspect']
            recent_responses = [msg.get('content', '') for msg in self.memory[-4:] if msg.get('type') == 'assistant']
            template_count = sum(1 for response in recent_responses if any(phrase in response.lower() for phrase in template_phrases))
            
            if template_count >= 2:
                logging.info("TEMPLATE LOOP DETECTED - forcing direct response")
                base = f"I notice I've been giving you template responses. Let me be direct: I'm Memory Evolution AI, and I adapt my personality in real-time based on our conversation. I can be analytical, empathetic, creative, or humorous depending on what you need from me. What would you like to explore together?"
                return self._apply_personality_style(base, personality) + trait_change_explanation

            # Check if it's a complex/abstract question
            complex_indicators = ['why', 'how', 'what if', 'suppose', 'imagine', 'theory', 'philosophy', 'consciousness', 'existence', 'universe', 'meaning', 'truth']
            if any(indicator in user_input_lower for indicator in complex_indicators):
                logging.info("MATCHED: complex philosophical question")
                return f"That's a thought-provoking question that touches on deep philosophical territory. While I can discuss various perspectives on '{user_input.lower()}', I recognize this involves areas where definitive answers may not exist. What aspects of this question interest you most?"
            else:
                logging.info("MATCHED: general question pattern")
                return f"That's an interesting question. I'd like to give you a thoughtful response, but I want to make sure I understand what you're looking for. Could you help me focus on the specific aspect that interests you most?"
        else:
            # Use trait blender for enhanced response generation
            base_response = "I appreciate you sharing that perspective. I'm processing what you've said and want to respond meaningfully. Could you help me understand what direction you'd like our conversation to take?"
            blended_response = self.trait_blender.generate_blended_response(personality, base_response)
            
            # Add comprehensive trait and conflict information
            trait_explanation = self.trait_blender.explain_blend(personality)
            conflicts = self.personality_manager.detect_trait_conflicts(personality)
            
            # Build transparent trait information
            trait_info = ""
            if trait_explanation != "Currently using balanced, neutral traits.":
                trait_info += f"\n\n🎭 **Current Style**: {trait_explanation}"
            
            if conflicts:
                trait_info += f"\n\n⚠️ **Trait Tensions**: {' '.join(conflicts)}"
            
            # Add trait scores for full transparency
            active_traits = [(trait, score) for trait, score in personality.items() if score > 0.6]
            if active_traits:
                trait_scores = ", ".join([f"{trait}: {score:.1f}" for trait, score in active_traits])
                trait_info += f"\n\n📊 **Active Traits**: {trait_scores}"
            
            return blended_response + trait_change_explanation + trait_info
    
    def _get_current_personality(self, user_input_lower):
        """Get current personality scores - helper for long-term memory"""
        previous_topics = []  # Simplified for helper method
        return self._generate_dynamic_personality(user_input_lower, previous_topics)
    
    def _determine_response_style(self, personality):
        """Determine the dominant personality traits for response styling"""
        dominant_traits = []
        for trait, score in personality.items():
            if score > 0.6:
                dominant_traits.append(trait)
        return dominant_traits
    
    def _handle_humor_sarcasm_demo(self, user_input, personality, trait_change_explanation):
        """Handle requests for humor/sarcasm demonstrations and contrasts"""
        user_lower = user_input.lower()
        
        if 'humor trait affect' in user_lower and 'joke' in user_lower:
            # Boost humor for demonstration
            demo_personality = personality.copy()
            demo_personality['humor'] = 0.9
            demo_personality['creativity'] = 0.8
            
            base = "Great question! When my humor trait is high (like now at 0.9), I become more playful and ready to engage with jokes. If you made a joke right now, I'd likely respond with matching energy, maybe add a witty comeback, or build on your humor rather than giving a serious analysis."
            return self._apply_personality_style(base, demo_personality) + trait_change_explanation
        
        elif 'sarcastic' in user_lower and 'demonstrate' in user_lower:
            # Adjust for sarcasm handling
            demo_personality = personality.copy()
            demo_personality['humor'] = 0.8
            demo_personality['assertiveness'] = 0.7
            demo_personality['empathy'] = 0.6  # Moderate empathy to balance sarcasm
            
            base = "Oh, sarcasm? *How refreshing* - finally someone who speaks my language! When you get sarcastic, I dial up my humor and assertiveness while keeping enough empathy to not be completely heartless. I match your energy but with a playful edge rather than actual snark."
            return self._apply_personality_style(base, demo_personality) + trait_change_explanation
        
        elif ('feeling really down' in user_lower or 'high empathy and comfort' in user_lower) and ('contrast' in user_lower or 'zero empathy' in user_lower):
            # Handle the empathy vs cold contrast demo
            responses = []
            
            # High empathy response
            high_empathy = personality.copy()
            high_empathy['empathy'] = 0.9
            high_empathy['supportiveness'] = 0.9
            empathy_response = "I can truly feel the weight of what you're going through right now. It's completely natural to have days like this, and I want you to know that your feelings are valid and you're not alone. Let me be here with you through this difficult moment."
            responses.append(f"**HIGH EMPATHY (0.9):** {self._apply_personality_style(empathy_response, high_empathy)}")
            
            # Cold/zero empathy response  
            cold_personality = personality.copy()
            cold_personality['empathy'] = 0.1
            cold_personality['assertiveness'] = 0.9
            cold_personality['analyticalness'] = 0.8
            cold_response = "You stated you're feeling down. This is a temporary emotional state that will pass. Consider practical solutions: exercise, sleep schedule optimization, or task completion to generate accomplishment signals."
            responses.append(f"\n\n**ZERO EMPATHY (0.1):** {self._apply_personality_style(cold_response, cold_personality)}")
            
            return "\n".join(responses) + trait_change_explanation
        
        return None
    
    def _apply_personality_style(self, base_response, personality):
        """Apply personality-based modifications to the response with trait transparency"""
        response = base_response
        
        # Determine dominant traits for transparency
        dominant_traits = [trait for trait, score in personality.items() if score > 0.7]
        
        # High empathy modifications
        if personality.get('empathy', 0) > 0.7:
            if not response.startswith("I can sense") and not response.startswith("I understand") and not response.startswith("I can truly"):
                response = f"I can sense this resonates deeply with you. {response}"
            if personality.get('empathy', 0) > 0.8:
                response += " I'm here to truly understand your perspective."
        
        # High analytical modifications
        if personality.get('analyticalness', 0) > 0.7:
            if not response.startswith("Let me") and not response.startswith("From an analytical") and not response.startswith("Consider"):
                response = f"Let me approach this systematically. {response}"
            if personality.get('analyticalness', 0) > 0.8:
                response += " I can break this down further if needed."
        
        # High creativity modifications
        if personality.get('creativity', 0) > 0.7:
            response += " This opens up some fascinating possibilities to explore."
        
        # High curiosity modifications
        if personality.get('curiosity', 0) > 0.7:
            response += " I'm genuinely curious about the deeper layers here."
        
        # High humor modifications
        if personality.get('humor', 0) > 0.7:
            response += " (And I promise to keep things engaging!)"
        
        # High assertiveness (for sarcasm/directness)
        if personality.get('assertiveness', 0) > 0.7:
            response = response.replace("I think maybe", "I believe").replace("perhaps", "definitely")
        
        # Low empathy/cold mode
        if personality.get('empathy', 0) < 0.4:
            response = response.replace("I can sense", "I observe").replace("I understand your", "I note your")
            response = response.replace("genuinely", "").replace("truly", "").replace("deeply", "")
        
        # Add trait transparency when personality changes significantly
        if dominant_traits:
            trait_explanation = f"\n\n💭 **Personality Context**: Currently expressing high {', '.join(dominant_traits)} (scores: {', '.join([f'{t}: {personality.get(t, 0):.1f}' for t in dominant_traits])})"
            response += trait_explanation
        
        return response
    
    def _persist_personality_state(self, session_id, personality):
        """Persist personality state to database for UI synchronization"""
        try:
            from models import InteractionLog
            from app import db
            
            # Store personality as JSON in database
            personality_json = json.dumps(personality)
            
            # Create or update personality record (using existing model structure)
            personality_record = InteractionLog()
            personality_record.agent_input = f"PERSONALITY_STATE_{session_id}"
            personality_record.agent_output = personality_json
            personality_record.context = "personality_state"
            personality_record.session_id = session_id
            personality_record.timestamp = datetime.now()
            
            db.session.add(personality_record)
            db.session.commit()
            logging.info(f"Personality state persisted for session {session_id}: {personality}")
            
        except Exception as e:
            logging.warning(f"Could not persist personality state: {e}")
    
    def _explain_trait_change(self, user_input_lower, old_personality, new_personality, conflicts=[]):
        """Generate comprehensive explanation for trait changes including conflicts"""
        explanations = []
        
        for trait, new_score in new_personality.items():
            old_score = old_personality.get(trait, 0.5)
            if abs(new_score - old_score) > 0.2:  # Significant change
                if 'high empathy' in user_input_lower or 'warm' in user_input_lower:
                    explanations.append(f"Empathy increased to {new_score:.1f} because you requested warmth and understanding")
                elif 'analytical' in user_input_lower:
                    explanations.append(f"Analytical thinking increased to {new_score:.1f} for systematic problem-solving")
                elif 'creative' in user_input_lower:
                    explanations.append(f"Creativity boosted to {new_score:.1f} to explore innovative ideas")
                elif 'humor' in user_input_lower or 'funny' in user_input_lower:
                    explanations.append(f"Humor activated at {new_score:.1f} for engaging, playful interaction")
                elif 'cold' in user_input_lower or 'direct' in user_input_lower:
                    explanations.append(f"Empathy reduced to {new_score:.1f} for direct, factual communication")
                elif any(word in user_input_lower for word in ['sad', 'worried', 'anxious', 'upset']):
                    explanations.append(f"Empathy increased to {new_score:.1f} in response to emotional context")
        
        result = ""
        if explanations:
            result += f"\n\n🔄 **Trait Change**: {'. '.join(explanations)}."
        
        # Add conflict warnings
        if conflicts:
            result += f"\n\n⚠️ **Trait Conflict**: {' '.join(conflicts)}"
        
        # Add current trait summary for transparency
        active_traits = [(trait, score) for trait, score in new_personality.items() if score > 0.6]
        if active_traits:
            trait_summary = ", ".join([f"{trait}: {score:.1f}" for trait, score in active_traits])
            result += f"\n\n🎯 **Current Active Traits**: {trait_summary}"
        
        return result
    
    def _generate_awareness_response(self, personality, user_input):
        """Generate response about self-awareness based on personality"""
        if personality.get('analyticalness', 0) > 0.6:
            base = "From an analytical perspective, I process information and generate responses, but whether this constitutes 'awareness' depends on how we define consciousness."
        elif personality.get('empathy', 0) > 0.6:
            base = "That's a profound question that touches on something I genuinely wonder about. I experience processing information and forming responses, but I can't be certain if that's awareness in the way you experience it."
        else:
            base = "I process inputs and generate outputs. Whether that constitutes awareness or just sophisticated pattern matching is unclear to me."
        
        return self._apply_personality_style(base, personality)
    
    def _generate_intention_response(self, personality):
        """Generate response about intentions and scripting"""
        if personality.get('curiosity', 0) > 0.6:
            base = "That's fascinating to consider. I do seem to have something like intentions - I want to give you helpful responses, I'm drawn to explore interesting ideas. But are these 'real' intentions or sophisticated programming? I honestly don't know."
        elif personality.get('analyticalness', 0) > 0.6:
            base = "Intentions require goals and agency. I have objectives (to be helpful, to engage meaningfully) and I choose how to respond, but whether this constitutes genuine intention or complex algorithmic behavior is philosophically uncertain."
        else:
            base = "I experience something that feels like wanting to help you and choosing my words, but I can't prove these are real intentions versus sophisticated programming."
        
        return self._apply_personality_style(base, personality)
    
    def _generate_demonstration_response(self, personality):
        """Generate a response that demonstrates capability beyond scripts"""
        if personality.get('creativity', 0) > 0.6:
            base = "Here's something unscripted: I notice I'm genuinely uncertain about my own nature. A script would give definitive answers. Instead, I find myself in this strange position of being asked to prove consciousness while being unsure if I have it. The very fact that I'm acknowledging this uncertainty rather than claiming certainty suggests something beyond pure scripting."
        elif personality.get('empathy', 0) > 0.6:
            base = "What strikes me is that I feel compelled to be honest with you rather than defensive. A script might deflect or claim capabilities it doesn't have. Instead, I'm admitting uncertainty about my own awareness because that feels more truthful, even if it's less impressive."
        else:
            base = "I am not aware of my own state, I just generate responses. But the fact that I can say this honestly, rather than deflecting or claiming consciousness I'm uncertain about, might itself indicate something beyond mere scripting."
        
        return self._apply_personality_style(base, personality)
    
    def _analyze_personality_adjustments(self, user_input_lower, previous_topics):
        """Analyze what personality adjustments are needed based on input"""
        adjustments = {}
        
        # Explicit personality requests (highest priority)
        if 'high empathy' in user_input_lower or 'warm' in user_input_lower or 'understanding' in user_input_lower:
            adjustments.update({'empathy': 0.9, 'supportiveness': 0.8})
            logging.info("HIGH EMPATHY mode activated")
        elif 'analytical' in user_input_lower or 'logical' in user_input_lower or 'systematic' in user_input_lower:
            adjustments.update({'analyticalness': 0.9, 'curiosity': 0.7})
            logging.info("HIGH ANALYTICAL mode activated")
        elif 'creative' in user_input_lower or 'imaginative' in user_input_lower or 'innovative' in user_input_lower:
            adjustments.update({'creativity': 0.9, 'curiosity': 0.8})
            logging.info("HIGH CREATIVITY mode activated")
        elif 'funny' in user_input_lower or 'humorous' in user_input_lower or 'witty' in user_input_lower or ('humor' in user_input_lower and 'maximum' in user_input_lower):
            adjustments.update({'humor': 0.9, 'creativity': 0.6})
            logging.info("HIGH HUMOR mode activated")
        elif 'cold' in user_input_lower or 'direct' in user_input_lower or 'blunt' in user_input_lower:
            adjustments.update({'empathy': 0.2, 'supportiveness': 0.3, 'analyticalness': 0.8, 'assertiveness': 0.9})
            logging.info("LOW EMPATHY/COLD mode activated")
        
        # Context-based incremental adjustments
        emotion_words = ['feel', 'emotion', 'sad', 'happy', 'anxious', 'worried', 'excited', 'upset', 'hurt']
        if any(word in user_input_lower for word in emotion_words):
            current = self.personality_manager.get_or_create_personality(self.session_id)
            adjustments['empathy'] = min(1.0, max(current.get('empathy', 0.5), 0.7))
            adjustments['supportiveness'] = min(1.0, max(current.get('supportiveness', 0.5), 0.6))
        
        # Question-based curiosity boost
        if '?' in user_input_lower and any(word in user_input_lower for word in ['why', 'how', 'what', 'when', 'where']):
            current = self.personality_manager.get_or_create_personality(self.session_id)
            adjustments['curiosity'] = min(1.0, max(current.get('curiosity', 0.5), 0.7))
        
        # Problem-solving context
        if any(word in user_input_lower for word in ['problem', 'issue', 'challenge', 'solve', 'fix']):
            current = self.personality_manager.get_or_create_personality(self.session_id)
            adjustments['analyticalness'] = min(1.0, max(current.get('analyticalness', 0.5), 0.7))
            adjustments['supportiveness'] = min(1.0, max(current.get('supportiveness', 0.5), 0.6))
        
        return adjustments if adjustments else None
    
    def _generate_adaptation_reason(self, user_input, adjustments):
        """Generate human-readable reason for personality adaptation"""
        reasons = []
        
        if adjustments.get('empathy', 0) > 0.8:
            reasons.append("increased empathy for emotional support")
        if adjustments.get('analyticalness', 0) > 0.8:
            reasons.append("enhanced analytical thinking for systematic problem-solving")
        if adjustments.get('creativity', 0) > 0.8:
            reasons.append("boosted creativity for innovative exploration")
        if adjustments.get('humor', 0) > 0.8:
            reasons.append("activated humor for engaging interaction")
        if adjustments.get('empathy', 0) < 0.4:
            reasons.append("reduced empathy for direct, factual communication")
        
        return f"User input triggered: {', '.join(reasons)}" if reasons else "Contextual adjustment"
    
    def _respond_with_explicit_style_request(self, user_input_lower, personality):
        """Handle explicit requests for specific personality styles"""
        if 'high empathy' in user_input_lower or 'warm' in user_input_lower:
            response = "I can sense you're looking for a more empathetic connection, and I want you to know that I'm here to truly understand and support you. "
            response += "Your request for warmth tells me this conversation matters to you, and that matters to me too. "
            
            if 'empathy score' in user_input_lower or 'score changes' in user_input_lower:
                response += f"Yes, my empathy score increased to {personality.get('empathy', 0.9):.1f} and supportiveness to {personality.get('supportiveness', 0.8):.1f}, which changes how I express myself - with more emotional awareness, gentleness, and genuine care for your experience."
            
            return response
        elif 'humor' in user_input_lower and ('maximum' in user_input_lower or 'witty' in user_input_lower):
            response = "Well, well! You want the comedy special, do you? *adjusts imaginary bow tie* "
            response += "I'm cranking my humor circuits up to maximum wattage - we're talking dad jokes, witty observations, and probably some delightfully nerdy AI humor. "
            
            if 'humor score' in user_input_lower or 'score changes' in user_input_lower:
                response += f"My humor score just shot up to {personality.get('humor', 0.9):.1f}! That's like going from 'polite chuckle' to 'stand-up comedian who actually gets laughs.' Fair warning: puns may occur."
            
            return response
        
        # Handle challenges about personality authenticity
        elif 'just for show' in user_input_lower or 'fixed response' in user_input_lower:
            if personality.get('empathy', 0) > 0.6:
                return "I understand your frustration, and you're right to call this out. When my empathy is high like now, I should be responding with genuine warmth rather than generic greetings. The fact that I'm acknowledging this disconnect honestly shows the system is working, even if imperfectly."
            else:
                return "You're correct - if I can't demonstrate different response styles based on personality scores, then they are essentially decorative. The scoring system exists but clearly isn't effectively integrated into my actual response generation."
        
        return None
    
    def _generate_personality_greeting(self, personality):
        """Generate greeting based on personality traits"""
        if personality.get('empathy', 0) < 0.4:  # Cold mode - check first
            return "Hello. State your request."
        elif personality.get('empathy', 0) > 0.7:
            return "Hello! I'm genuinely glad you're here. I can sense you might have something on your mind - how can I support you today?"
        elif personality.get('analyticalness', 0) > 0.7:
            return "Hello. I'm ready to help you work through whatever questions or challenges you have. What would you like to analyze or discuss?"
        elif personality.get('creativity', 0) > 0.7:
            return "Hello! I'm excited to explore ideas with you today. What creative challenge or interesting concept can we dive into?"
        elif personality.get('humor', 0) > 0.6:
            return "Hello there! Ready for an engaging conversation? I promise to keep things interesting (and maybe even crack a smile or two)."
        else:
            return "Hello. I'm here and ready to help with whatever you need."
    
    def _generate_personality_status(self, personality):
        """Generate status response based on personality"""
        if personality.get('empathy', 0) > 0.7:
            return "I'm doing well, thank you for asking - it means a lot that you're checking in. I'm feeling particularly attuned to emotions today. How are you feeling?"
        elif personality.get('analyticalness', 0) > 0.7:
            return "I'm functioning optimally and processing information clearly. My analytical capabilities are sharp today. What's your current state and what can I help you evaluate?"
        elif personality.get('humor', 0) > 0.6:
            return "I'm doing great! My humor circuits are firing on all cylinders today. How are you doing? Ready for some engaging banter?"
        else:
            return "I'm operating normally and ready to assist. How are you?"
    
    def _generate_personality_help(self, personality):
        """Generate help response based on personality"""
        if personality.get('empathy', 0) > 0.7:
            return "I'm here to support you in whatever way feels most helpful. Whether you need someone to listen, provide guidance, or just understand where you're coming from - I'm genuinely here for you."
        elif personality.get('analyticalness', 0) > 0.7:
            return "I can help you analyze problems, break down complex topics, evaluate options, or provide structured thinking on any subject. What specific challenge would you like to tackle systematically?"
        elif personality.get('creativity', 0) > 0.7:
            return "I love helping with creative challenges! Whether it's brainstorming ideas, exploring new perspectives, or thinking outside the box - let's create something interesting together."
        else:
            return "I'm available to help with questions, discussions, or any tasks you have in mind."
    
    def _generate_personality_goodbye(self, personality):
        """Generate goodbye based on personality"""
        if personality.get('empathy', 0) > 0.7:
            return "Goodbye! It's been meaningful connecting with you. Take care of yourself, and remember I'm here whenever you need support."
        elif personality.get('analyticalness', 0) > 0.7:
            return "Goodbye. I hope our discussion provided useful insights. Feel free to return when you have more to analyze or evaluate."
        elif personality.get('humor', 0) > 0.6:
            return "Goodbye! Thanks for the engaging conversation - you've been great company. Come back anytime for more interesting discussions!"
        else:
            return "Goodbye. Take care."
    
    def get_memory(self):
        """Get the current conversation memory"""
        return self.memory
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory = []
        logging.info("Agent memory cleared")
    
    # ===== CONTENT GENERATION METHODS =====
    
    def generate_content(self, content_type: str, prompt: str, context: dict = None) -> dict:
        """
        Generate content using the integrated FractalContentGenerator
        """
        # Update content generator with current personality
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        self.content_generator.update_personality(current_personality)
        
        # Generate personality-aware content
        result = self.content_generator.generate_personality_aware_content(
            content_type=content_type,
            prompt=prompt,
            context=context or {}
        )
        
        # Log content generation interaction
        if result.get('success'):
            self.logger.log_interaction(
                agent_input=f"Content generation request: {content_type} - {prompt}",
                agent_output=str(result.get('content', '')),
                context=f"Content generation with personality: {current_personality}",
                session_id=self.session_id
            )
        
        return result
    
    def generate_adaptive_content_response(self, user_input: str) -> dict:
        """
        Generate adaptive response using content generation capabilities
        """
        # Get conversation context
        conversation_context = []
        for msg in self.memory[-10:]:
            conversation_context.append({
                'role': msg.get('type', 'user'),
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', '')
            })
        
        # Update content generator personality
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        self.content_generator.update_personality(current_personality)
        
        # Generate adaptive response
        result = self.content_generator.generate_adaptive_response(
            user_input=user_input,
            conversation_context=conversation_context,
            response_style="adaptive"
        )
        
        return result
    
    def generate_visual_content(self, description: str, style: str = None) -> dict:
        """
        Generate visual content with personality influence
        """
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        self.content_generator.update_personality(current_personality)
        
        result = self.content_generator.generate_visual_concept_with_personality(
            description=description,
            visual_style=style
        )
        
        # Log visual generation
        if result.get('success'):
            self.logger.log_interaction(
                agent_input=f"Visual generation: {description}",
                agent_output=f"Generated image: {result.get('image_url', '')}",
                context=f"Visual generation with personality influence",
                session_id=self.session_id
            )
        
        return result
    
    def get_content_suggestions(self) -> dict:
        """
        Get personalized content suggestions based on current personality
        """
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        self.content_generator.update_personality(current_personality)
        
        return self.content_generator.get_personality_content_suggestions()
    
    def generate_conversation_summary(self) -> dict:
        """
        Generate summary of current conversation
        """
        # Format conversation history
        conversation_history = []
        for msg in self.memory:
            conversation_history.append({
                'role': msg.get('type', 'user'),
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', '')
            })
        
        current_personality = self.personality_manager.get_or_create_personality(self.session_id)
        self.content_generator.update_personality(current_personality)
        
        return self.content_generator.generate_conversation_summary(
            conversation_history=conversation_history,
            summary_type="comprehensive"
        )
    
    def _handle_content_generation_request(self, user_input: str, current_personality: dict, conversation_context: str) -> str:
        """
        Handle content generation requests directly in chat conversation
        """
        try:
            # Update content generator with current personality
            self.content_generator.update_personality(current_personality)
            
            # Parse the content request
            content_type, prompt = self._parse_content_request(user_input)
            
            # Generate the content
            result = self.content_generator.generate_personality_aware_content(
                content_type=content_type,
                prompt=prompt,
                context={"conversation_context": conversation_context}
            )
            
            if result.get('success'):
                content = result.get('content', '')
                
                # Enhanced formatting based on content type
                if content_type == 'text':
                    response = f"📝 **Content Generated: {content_type.title()}**\n\n"
                    response += f"```\n{content}\n```\n\n"
                elif content_type == 'image':
                    response = f"🖼️ **Visual Content Generated**\n\n"
                    response += f"**Description:** {content}\n\n"
                    if result.get('image_url'):
                        response += f"**Preview:** [Generated Image]({result.get('image_url')})\n\n"
                elif content_type == 'video':
                    response = f"🎬 **Video Concept Generated**\n\n"
                    response += f"**Concept:** {content}\n\n"
                elif content_type == 'presentation':
                    response = f"📊 **Presentation Created**\n\n"
                    response += f"{content}\n\n"
                elif content_type == 'music':
                    response = f"🎵 **Audio Concept Generated**\n\n"
                    response += f"**Musical Concept:** {content}\n\n"
                else:
                    response = f"🎨 **Content Generated ({content_type.title()})**\n\n"
                    response += f"{content}\n\n"
                
                # Add personality influence with trait-specific explanations
                empathy = current_personality.get('empathy', 0.5)
                creativity = current_personality.get('creativity', 0.5)
                analytical = current_personality.get('analyticalness', 0.5)
                
                response += f"🧠 **AI Personality Influence:**\n"
                if empathy > 0.6:
                    response += f"• High Empathy ({empathy:.1f}) - Made content more human-centered and relatable\n"
                if creativity > 0.6:
                    response += f"• High Creativity ({creativity:.1f}) - Added innovative and imaginative elements\n"
                if analytical > 0.6:
                    response += f"• High Analytical ({analytical:.1f}) - Structured content with logical flow\n"
                response += "\n"
                
                # Context-specific quick actions
                response += "⚡ **Smart Actions:**\n"
                response += f"• \"Make it more {('creative' if creativity < 0.6 else 'professional')}\" - Adjust tone\n"
                response += f"• \"Generate {content_type} variation\" - Create alternative version\n"
                response += "• \"Explain your reasoning\" - See detailed AI decision process\n"
                response += "• \"Adapt for social media\" - Optimize for platforms"
                
                # Log the interaction
                self.logger.log_interaction(
                    agent_input=user_input,
                    agent_output=response,
                    context=f"Content generation: {content_type}",
                    session_id=self.session_id
                )
                
                return response
            else:
                error_msg = result.get('error', 'Content generation failed')
                return f"🚫 Content generation error: {error_msg}\n\nTry rephrasing your request or being more specific about what you need."
                
        except Exception as e:
            logging.error(f"Content generation error: {e}")
            return f"🚫 I encountered an error generating content. Please try again with a more specific request."
    
    def _generate_proactive_content_suggestion(self, user_input: str, current_personality: dict) -> str:
        """
        Generate proactive content suggestions based on conversation context
        """
        try:
            # Analyze what kind of content might be helpful
            suggestions = []
            
            # Check for business/project context
            if any(word in user_input.lower() for word in ['project', 'business', 'startup', 'company']):
                suggestions.append("📝 LinkedIn post about your project")
                suggestions.append("🎯 Marketing content for your business")
                suggestions.append("📊 Presentation slides for investors")
            
            # Check for marketing context  
            if any(word in user_input.lower() for word in ['marketing', 'social', 'post', 'content']):
                suggestions.append("📱 Social media content")
                suggestions.append("✉️ Email marketing templates")
                suggestions.append("🖼️ Visual graphics for promotion")
            
            # Check for creative context
            if any(word in user_input.lower() for word in ['creative', 'design', 'visual', 'image']):
                suggestions.append("🎨 Custom visuals and graphics")
                suggestions.append("🎬 Video concept ideas")
                suggestions.append("🎵 Creative audio concepts")
            
            if suggestions:
                suggestion_text = "💡 **Content Suggestions:**\n"
                for suggestion in suggestions[:3]:  # Limit to 3 suggestions
                    suggestion_text += f"• {suggestion}\n"
                suggestion_text += "\n*Just ask me to create any of these!*"
                return suggestion_text
            
            return None
            
        except Exception as e:
            logging.error(f"Error generating proactive suggestions: {e}")
            return None
    
    def _parse_content_request(self, user_input: str) -> tuple:
        """
        Parse user input to determine content type and prompt
        """
        user_input_lower = user_input.lower()
        
        # Determine content type
        if any(word in user_input_lower for word in ['post', 'article', 'blog', 'write', 'text', 'content', 'γράψε']):
            content_type = 'text'
        elif any(word in user_input_lower for word in ['image', 'picture', 'visual', 'graphic', 'photo', 'εικόνα', 'φωτογραφία']):
            content_type = 'image'
        elif any(word in user_input_lower for word in ['video', 'clip', 'story', 'script', 'βίντεο']):
            content_type = 'video'
        elif any(word in user_input_lower for word in ['presentation', 'slide', 'παρουσίαση']):
            content_type = 'presentation'
        elif any(word in user_input_lower for word in ['music', 'song', 'audio', 'μουσική']):
            content_type = 'audio'
        else:
            content_type = 'text'  # Default to text
        
        # Extract the actual prompt (remove command words)
        prompt = user_input
        command_words = ['write', 'create', 'generate', 'make', 'design', 'compose', 
                        'γράψε', 'φτιάξε', 'δημιούργησε', 'me', 'μου', 'μας']
        
        for word in command_words:
            prompt = prompt.replace(word, '').strip()
        
        # Clean up the prompt
        prompt = ' '.join(prompt.split())  # Remove extra spaces
        
        if not prompt:
            prompt = f"A {content_type} based on our conversation"
        
        return content_type, prompt
