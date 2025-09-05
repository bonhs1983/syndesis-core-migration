"""
Production-Ready Pipeline Integration
Connects all components: memory, traits, triggers, and response generation
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .trait_response_engine import trait_response_engine
from .dynamic_trait_triggers import dynamic_trait_triggers
from .contextual_memory_recall import contextual_memory_recall
from .persistent_personality import PersistentPersonalityManager


class ProductionPipeline:
    """
    Complete production pipeline that processes user input through all systems
    """
    
    def __init__(self):
        self.personality_manager = PersistentPersonalityManager()
        self.response_engine = trait_response_engine
        self.trait_triggers = dynamic_trait_triggers
        self.memory_system = contextual_memory_recall
        
        # Pipeline configuration
        self.enable_memory_recall = True
        self.enable_dynamic_traits = True
        self.enable_explainability = True
        self.log_all_interactions = True
        
    def process_user_input(self, session_id: str, user_input: str, 
                          context: Dict = None) -> Dict[str, Any]:
        """
        Complete pipeline processing: input → memory → traits → response
        """
        try:
            pipeline_start = datetime.now()
            
            # Step 1: Input validation and logging
            if not user_input or not user_input.strip():
                return self._generate_error_response("Empty input provided")
            
            if self.log_all_interactions:
                logging.info(f"Pipeline processing for session {session_id}: {user_input[:100]}...")
            
            # Step 2: Get current personality state
            try:
                current_traits = self.personality_manager.get_personality(session_id)
            except Exception as e:
                logging.error(f"Error getting personality: {e}")
                current_traits = self._get_default_traits()
            
            # Step 3: Analyze for dynamic trait triggers
            trigger_analysis = {}
            if self.enable_dynamic_traits:
                try:
                    trigger_analysis = self.trait_triggers.analyze_input_triggers(user_input, current_traits)
                    
                    # Apply dynamic adjustments if suggested
                    if trigger_analysis.get('adjustments'):
                        adjusted_traits = self.trait_triggers.apply_dynamic_adjustments(
                            current_traits,
                            trigger_analysis['adjustments'],
                            f"Dynamic trigger: {trigger_analysis.get('analysis', 'input analysis')}"
                        )
                        
                        # Update personality with new traits
                        try:
                            self.personality_manager.update_personality(session_id, adjusted_traits)
                            current_traits = adjusted_traits
                        except Exception as e:
                            logging.error(f"Error updating personality: {e}")
                            
                except Exception as e:
                    logging.error(f"Error in trait trigger analysis: {e}")
                    trigger_analysis = {'error': str(e)}
            
            # Step 4: Recall relevant memory context
            memory_context = {}
            if self.enable_memory_recall:
                try:
                    # Check for explicit recall requests
                    if self._is_explicit_recall_request(user_input):
                        memory_context = self.memory_system.handle_explicit_recall_request(session_id, user_input)
                    else:
                        memory_context = self.memory_system.recall_relevant_context(session_id, user_input)
                except Exception as e:
                    logging.error(f"Error in memory recall: {e}")
                    memory_context = {'error': str(e)}
            
            # Step 5: Generate trait-based response
            try:
                response_data = self.response_engine.generate_response(
                    user_input=user_input,
                    traits=current_traits,
                    context={
                        'memory_recall': memory_context.get('integration_text', ''),
                        'session_context': context or {},
                        'trigger_analysis': trigger_analysis
                    }
                )
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                response_data = self._generate_fallback_response(user_input)
            
            # Step 6: Store conversation in memory
            if self.enable_memory_recall:
                try:
                    ai_response = response_data.get('response', '')
                    self.memory_system.store_conversation_turn(
                        session_id=session_id,
                        user_input=user_input,
                        ai_response=ai_response,
                        context={
                            'traits_used': response_data.get('traits_used', []),
                            'style_info': response_data.get('style_info', {}),
                            'trigger_analysis': trigger_analysis,
                            'pipeline_timestamp': pipeline_start.isoformat()
                        }
                    )
                except Exception as e:
                    logging.error(f"Error storing conversation: {e}")
            
            # Step 7: Generate explainability if requested
            explanation = {}
            if self.enable_explainability:
                try:
                    explanation = self._generate_pipeline_explanation(
                        trigger_analysis, memory_context, response_data, current_traits
                    )
                except Exception as e:
                    logging.error(f"Error generating explanation: {e}")
                    explanation = {'error': str(e)}
            
            # Step 8: Compile final response
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            
            final_response = {
                'response': response_data.get('response', self._get_safe_fallback()),
                'session_id': session_id,
                'current_traits': current_traits,
                'style_info': response_data.get('style_info', {}),
                'traits_used': response_data.get('traits_used', []),
                'trigger_analysis': trigger_analysis,
                'memory_context': memory_context,
                'explanation': explanation,
                'pipeline_info': {
                    'processing_time_seconds': pipeline_duration,
                    'timestamp': pipeline_start.isoformat(),
                    'components_used': {
                        'dynamic_traits': self.enable_dynamic_traits and bool(trigger_analysis.get('adjustments')),
                        'memory_recall': self.enable_memory_recall and bool(memory_context.get('relevant_memories')),
                        'trait_blending': response_data.get('style_info', {}).get('blend', False),
                        'fallback_used': response_data.get('fallback_used', False)
                    }
                },
                'success': True
            }
            
            if self.log_all_interactions:
                logging.info(f"Pipeline completed for {session_id} in {pipeline_duration:.3f}s")
            
            return final_response
            
        except Exception as e:
            logging.error(f"Critical error in production pipeline: {e}")
            return self._generate_error_response(f"Pipeline error: {str(e)}")
    
    def explain_response_reasoning(self, session_id: str, response_data: Dict) -> str:
        """
        Generate detailed explanation of response reasoning (explainability mode)
        """
        try:
            if not response_data.get('success'):
                return "Unable to explain response due to processing error."
            
            explanation_parts = []
            
            # Trait influence explanation
            if response_data.get('traits_used'):
                trait_explanation = self.response_engine.explain_response_choice(response_data)
                explanation_parts.append(f"Personality influence: {trait_explanation}")
            
            # Dynamic trigger explanation
            trigger_analysis = response_data.get('trigger_analysis', {})
            if trigger_analysis.get('adjustments'):
                trigger_explanation = trigger_analysis.get('analysis', 'Traits were dynamically adjusted')
                explanation_parts.append(f"Dynamic adaptation: {trigger_explanation}")
            
            # Memory integration explanation
            memory_context = response_data.get('memory_context', {})
            if memory_context.get('relevant_memories'):
                memory_explanation = memory_context.get('explanation', 'Previous conversation context was used')
                explanation_parts.append(f"Memory integration: {memory_explanation}")
            
            # Response style explanation
            style_info = response_data.get('style_info', {})
            if style_info.get('blend'):
                explanation_parts.append(f"Style blending: Combined {style_info.get('primary_trait')} with {style_info.get('secondary_trait')}")
            elif style_info.get('primary_trait'):
                explanation_parts.append(f"Style focus: Emphasized {style_info.get('primary_trait')} communication")
            
            if explanation_parts:
                return "Here's how I crafted my response: " + "; ".join(explanation_parts) + "."
            else:
                return "I used a balanced approach without strong trait influences or memory context."
                
        except Exception as e:
            logging.error(f"Error explaining response reasoning: {e}")
            return "I'm not sure how I arrived at that response."
    
    def get_pipeline_health_status(self) -> Dict[str, Any]:
        """Get health status of all pipeline components"""
        try:
            health_status = {
                'overall_status': 'healthy',
                'components': {},
                'statistics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Test personality manager
            try:
                test_traits = self.personality_manager.get_personality('health_check')
                health_status['components']['personality_manager'] = {
                    'status': 'healthy',
                    'test_result': 'Successfully retrieved traits'
                }
            except Exception as e:
                health_status['components']['personality_manager'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            # Test response engine
            try:
                test_response = self.response_engine.generate_response(
                    "health check", {'empathy': 0.5}, {}
                )
                health_status['components']['response_engine'] = {
                    'status': 'healthy',
                    'template_stats': self.response_engine.get_template_variety_stats()
                }
            except Exception as e:
                health_status['components']['response_engine'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            # Test trait triggers
            try:
                test_triggers = self.trait_triggers.analyze_input_triggers(
                    "health check", {'empathy': 0.5}
                )
                health_status['components']['trait_triggers'] = {
                    'status': 'healthy',
                    'trigger_stats': self.trait_triggers.get_trigger_statistics()
                }
            except Exception as e:
                health_status['components']['trait_triggers'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            # Test memory system
            try:
                test_memory = self.memory_system.recall_relevant_context('health_check', 'test')
                health_status['components']['memory_system'] = {
                    'status': 'healthy',
                    'test_result': 'Memory recall functioning'
                }
            except Exception as e:
                health_status['components']['memory_system'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            return {
                'overall_status': 'critical_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _is_explicit_recall_request(self, user_input: str) -> bool:
        """Check if user is explicitly asking for memory recall"""
        recall_phrases = [
            'what did i say', 'did i mention', 'i said', 'earlier i told you',
            'remember when', 'you remember', 'recall', 'what was i talking about'
        ]
        input_lower = user_input.lower()
        return any(phrase in input_lower for phrase in recall_phrases)
    
    def _generate_pipeline_explanation(self, trigger_analysis: Dict, memory_context: Dict,
                                     response_data: Dict, current_traits: Dict) -> Dict[str, Any]:
        """Generate comprehensive pipeline explanation"""
        try:
            explanation = {
                'trait_state': {
                    'current_values': current_traits,
                    'dominant_traits': [trait for trait, value in current_traits.items() if value > 0.7],
                    'balanced_traits': [trait for trait, value in current_traits.items() if 0.4 <= value <= 0.7]
                },
                'processing_steps': [],
                'decision_points': [],
                'adaptations_made': []
            }
            
            # Document processing steps
            explanation['processing_steps'].append("1. Analyzed user input for emotional and topical content")
            
            if trigger_analysis.get('adjustments'):
                explanation['processing_steps'].append("2. Applied dynamic trait adjustments based on input triggers")
                explanation['adaptations_made'].extend([
                    f"{trait}: {adjustment:+.2f}" for trait, adjustment in trigger_analysis['adjustments'].items()
                ])
            
            if memory_context.get('relevant_memories'):
                explanation['processing_steps'].append("3. Retrieved relevant conversation context from memory")
                explanation['decision_points'].append(f"Used {len(memory_context['relevant_memories'])} memory references")
            
            explanation['processing_steps'].append("4. Generated response based on active personality traits")
            
            if response_data.get('style_info', {}).get('blend'):
                explanation['decision_points'].append("Used trait blending for nuanced response")
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error generating pipeline explanation: {e}")
            return {'error': str(e)}
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate safe error response"""
        return {
            'response': "I'm having trouble processing your request right now, but I'm here to help. Could you try rephrasing your question?",
            'success': False,
            'error': error_message,
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Generate fallback response when trait engine fails"""
        fallback_responses = [
            "I'm here to help you with whatever you need.",
            "Thank you for sharing that with me. How can I assist you?",
            "I appreciate you bringing this to my attention. What would you like to explore?",
            "Let me help you with that to the best of my ability."
        ]
        
        import random
        return {
            'response': random.choice(fallback_responses),
            'style_info': {'style': 'fallback'},
            'traits_used': [],
            'explanation': 'Used fallback response due to processing error',
            'fallback_used': True
        }
    
    def _get_default_traits(self) -> Dict[str, float]:
        """Get default trait values"""
        return {
            'empathy': 0.5,
            'curiosity': 0.5,
            'analyticalness': 0.5,
            'creativity': 0.5,
            'humor': 0.4,
            'supportiveness': 0.5,
            'assertiveness': 0.5
        }
    
    def _get_safe_fallback(self) -> str:
        """Get absolutely safe fallback response"""
        return "I'm here to help you."


# Global pipeline instance
production_pipeline = ProductionPipeline()