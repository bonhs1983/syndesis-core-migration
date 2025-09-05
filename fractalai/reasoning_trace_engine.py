"""
Reasoning Trace Engine - Generates step-by-step breakdown and fractal graph analysis
"""
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class ReasoningTraceEngine:
    """
    Generates detailed reasoning traces and fractal graph outputs for response generation
    """
    
    def __init__(self):
        self.trace_templates = self._initialize_trace_templates()
        self.fractal_symbols = self._initialize_fractal_symbols()
        
    def _initialize_trace_templates(self) -> Dict[str, List[str]]:
        """Initialize reasoning step templates"""
        return {
            'input_analysis': [
                "User input analyzed: '{input}'",
                "Emotional tone detected: {emotion}",
                "Key topics extracted: {topics}",
                "Intent classification: {intent}"
            ],
            'memory_processing': [
                "Memory search initiated for session: {session_id}",
                "Found {count} relevant memories",
                "Memory match score: {score}",
                "Context integration: {context_type}"
            ],
            'trait_analysis': [
                "Current trait state: {traits}",
                "Trait triggers detected: {triggers}",
                "Dynamic adjustments: {adjustments}",
                "Dominant traits: {dominant}"
            ],
            'response_generation': [
                "Response style selected: {style}",
                "Template category: {category}",
                "Trait influence: {influence}",
                "Final response crafted with {method}"
            ],
            'explanation': [
                "Decision rationale: {rationale}",
                "Alternative paths considered: {alternatives}",
                "Confidence level: {confidence}",
                "Quality assurance: {qa_check}"
            ]
        }
    
    def _initialize_fractal_symbols(self) -> Dict[str, str]:
        """Initialize fractal symbolic representations"""
        return {
            'empathy': '♡',
            'humor': '☆',
            'analyticalness': '◇',
            'creativity': '◯',
            'curiosity': '?',
            'supportiveness': '▲',
            'assertiveness': '■',
            'memory': '◐',
            'trigger': '⚡',
            'response': '→',
            'blend': '◈',
            'fallback': '△'
        }
    
    def generate_reasoning_trace(self, processing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive reasoning trace for response processing
        """
        try:
            trace_steps = []
            fractal_graph = []
            
            # Step 1: Input Analysis
            input_step = self._trace_input_analysis(processing_data)
            trace_steps.append(input_step)
            fractal_graph.extend(input_step.get('fractal_nodes', []))
            
            # Step 2: Memory Processing
            memory_step = self._trace_memory_processing(processing_data)
            trace_steps.append(memory_step)
            fractal_graph.extend(memory_step.get('fractal_nodes', []))
            
            # Step 3: Trait Analysis
            trait_step = self._trace_trait_analysis(processing_data)
            trace_steps.append(trait_step)
            fractal_graph.extend(trait_step.get('fractal_nodes', []))
            
            # Step 4: Response Generation
            response_step = self._trace_response_generation(processing_data)
            trace_steps.append(response_step)
            fractal_graph.extend(response_step.get('fractal_nodes', []))
            
            # Step 5: Decision Explanation
            explanation_step = self._trace_explanation(processing_data)
            trace_steps.append(explanation_step)
            fractal_graph.extend(explanation_step.get('fractal_nodes', []))
            
            # Generate summary trace
            summary_trace = self._generate_summary_trace(trace_steps)
            
            # Generate fractal visualization
            fractal_visualization = self._generate_fractal_visualization(fractal_graph)
            
            return {
                'trace_steps': trace_steps,
                'summary_trace': summary_trace,
                'fractal_graph': fractal_graph,
                'fractal_visualization': fractal_visualization,
                'trace_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_steps': len(trace_steps),
                    'processing_time': processing_data.get('processing_time', 0),
                    'confidence_score': self._calculate_confidence_score(trace_steps)
                }
            }
            
        except Exception as e:
            logging.error(f"Error generating reasoning trace: {e}")
            return self._generate_fallback_trace(str(e))
    
    def _trace_input_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace input analysis step"""
        try:
            user_input = data.get('user_input', '')
            trigger_analysis = data.get('trigger_analysis', {})
            
            step = {
                'step_number': 1,
                'step_name': 'Input Analysis',
                'description': f"User states: '{user_input}'",
                'details': [],
                'fractal_nodes': []
            }
            
            # Emotional analysis
            emotional_context = trigger_analysis.get('emotional_context', {})
            if emotional_context:
                emotion = emotional_context.get('dominant_emotion', 'neutral')
                intensity = emotional_context.get('intensity', 0.0)
                step['details'].append(f"Emotional tone detected: {emotion} (intensity: {intensity:.2f})")
                step['fractal_nodes'].append({
                    'node': 'emotion_detection',
                    'symbol': '◐',
                    'value': emotion,
                    'intensity': intensity
                })
            
            # Topic extraction
            interaction_type = trigger_analysis.get('interaction_type', 'statement')
            step['details'].append(f"Interaction type: {interaction_type}")
            step['fractal_nodes'].append({
                'node': 'interaction_type',
                'symbol': '◯',
                'value': interaction_type
            })
            
            # Trigger detection
            triggers = trigger_analysis.get('triggers', {})
            if triggers:
                step['details'].append(f"Trait triggers detected: {list(triggers.keys())}")
                for trait, trigger_data in triggers.items():
                    symbol = self.fractal_symbols.get(trait, '?')
                    step['fractal_nodes'].append({
                        'node': f'trigger_{trait}',
                        'symbol': f'{symbol}⚡',
                        'strength': trigger_data.get('strength', 0.0),
                        'trait': trait
                    })
            
            return step
            
        except Exception as e:
            logging.error(f"Error tracing input analysis: {e}")
            return {'step_number': 1, 'step_name': 'Input Analysis', 'error': str(e)}
    
    def _trace_memory_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace memory processing step"""
        try:
            memory_context = data.get('memory_context', {})
            
            step = {
                'step_number': 2,
                'step_name': 'Memory Processing',
                'description': 'Searching conversation history for relevant context',
                'details': [],
                'fractal_nodes': []
            }
            
            if memory_context.get('relevant_memories'):
                memories = memory_context['relevant_memories']
                step['details'].append(f"Found {len(memories)} relevant memories")
                step['fractal_nodes'].append({
                    'node': 'memory_search',
                    'symbol': '◐',
                    'count': len(memories),
                    'type': 'contextual'
                })
                
                # Memory match details
                for i, memory in enumerate(memories[:3]):  # Show top 3
                    match_reason = memory.get('match_reason', 'unknown')
                    relevance = memory.get('relevance_score', 0.0)
                    step['details'].append(f"Memory {i+1}: {match_reason} (relevance: {relevance:.2f})")
                    step['fractal_nodes'].append({
                        'node': f'memory_match_{i+1}',
                        'symbol': '◐→',
                        'relevance': relevance,
                        'reason': match_reason
                    })
            
            elif memory_context.get('found_memories'):
                # Explicit recall
                memories = memory_context['found_memories']
                step['details'].append(f"Explicit recall: {len(memories)} memories found")
                step['fractal_nodes'].append({
                    'node': 'explicit_recall',
                    'symbol': '◐?',
                    'count': len(memories),
                    'type': 'explicit'
                })
            
            else:
                step['details'].append("No relevant memories found")
                step['fractal_nodes'].append({
                    'node': 'no_memory',
                    'symbol': '○',
                    'status': 'empty'
                })
            
            return step
            
        except Exception as e:
            logging.error(f"Error tracing memory processing: {e}")
            return {'step_number': 2, 'step_name': 'Memory Processing', 'error': str(e)}
    
    def _trace_trait_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace trait analysis step"""
        try:
            original_traits = data.get('original_traits', {})
            updated_traits = data.get('updated_traits', {})
            trigger_analysis = data.get('trigger_analysis', {})
            
            step = {
                'step_number': 3,
                'step_name': 'Personality Trait Analysis',
                'description': 'Analyzing and adjusting personality traits',
                'details': [],
                'fractal_nodes': []
            }
            
            # Current trait state
            dominant_traits = [trait for trait, value in updated_traits.items() if value > 0.7]
            if dominant_traits:
                step['details'].append(f"Dominant traits: {dominant_traits}")
                for trait in dominant_traits:
                    symbol = self.fractal_symbols.get(trait, '?')
                    step['fractal_nodes'].append({
                        'node': f'dominant_{trait}',
                        'symbol': f'{symbol}▲',
                        'value': updated_traits[trait],
                        'status': 'dominant'
                    })
            
            # Dynamic adjustments
            adjustments = trigger_analysis.get('adjustments', {})
            if adjustments:
                step['details'].append(f"Dynamic adjustments made: {list(adjustments.keys())}")
                for trait, adjustment in adjustments.items():
                    old_val = original_traits.get(trait, 0.5)
                    new_val = updated_traits.get(trait, old_val)
                    symbol = self.fractal_symbols.get(trait, '?')
                    
                    step['details'].append(f"{trait}: {old_val:.2f} → {new_val:.2f} ({adjustment:+.2f})")
                    step['fractal_nodes'].append({
                        'node': f'adjust_{trait}',
                        'symbol': f'{symbol}⚡→',
                        'old_value': old_val,
                        'new_value': new_val,
                        'adjustment': adjustment
                    })
            else:
                step['details'].append("No dynamic adjustments made")
                step['fractal_nodes'].append({
                    'node': 'no_adjustments',
                    'symbol': '◇',
                    'status': 'stable'
                })
            
            return step
            
        except Exception as e:
            logging.error(f"Error tracing trait analysis: {e}")
            return {'step_number': 3, 'step_name': 'Trait Analysis', 'error': str(e)}
    
    def _trace_response_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace response generation step"""
        try:
            style_info = data.get('style_info', {})
            traits_used = data.get('traits_used', [])
            response = data.get('response', '')
            
            step = {
                'step_number': 4,
                'step_name': 'Response Generation',
                'description': 'Crafting personality-driven response',
                'details': [],
                'fractal_nodes': []
            }
            
            # Response style
            style = style_info.get('style', 'balanced')
            step['details'].append(f"Response style selected: {style}")
            
            if style_info.get('blend'):
                primary = style_info.get('primary_trait')
                secondary = style_info.get('secondary_trait')
                step['details'].append(f"Trait blending: {primary} + {secondary}")
                step['fractal_nodes'].append({
                    'node': 'trait_blend',
                    'symbol': '◈',
                    'primary': primary,
                    'secondary': secondary,
                    'type': 'blended'
                })
            elif style_info.get('primary_trait'):
                primary = style_info.get('primary_trait')
                trait_value = style_info.get('trait_value', 0.0)
                step['details'].append(f"Primary trait influence: {primary} ({trait_value:.2f})")
                symbol = self.fractal_symbols.get(primary, '?')
                step['fractal_nodes'].append({
                    'node': 'primary_trait',
                    'symbol': f'{symbol}→',
                    'trait': primary,
                    'value': trait_value,
                    'type': 'single'
                })
            
            # Response crafting
            if data.get('fallback_used'):
                step['details'].append("Fallback response used for safety")
                step['fractal_nodes'].append({
                    'node': 'fallback',
                    'symbol': '△',
                    'reason': 'safety_fallback'
                })
            else:
                step['details'].append(f"Template-based response with {len(traits_used)} trait influences")
                step['fractal_nodes'].append({
                    'node': 'template_response',
                    'symbol': '→',
                    'traits_count': len(traits_used),
                    'length': len(response)
                })
            
            return step
            
        except Exception as e:
            logging.error(f"Error tracing response generation: {e}")
            return {'step_number': 4, 'step_name': 'Response Generation', 'error': str(e)}
    
    def _trace_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace explanation step"""
        try:
            explanation = data.get('explanation', '')
            
            step = {
                'step_number': 5,
                'step_name': 'Decision Explanation',
                'description': 'Generating reasoning explanation',
                'details': [],
                'fractal_nodes': []
            }
            
            if explanation:
                step['details'].append(f"Explanation generated: {explanation[:100]}...")
                step['fractal_nodes'].append({
                    'node': 'explanation',
                    'symbol': '◇?',
                    'length': len(explanation),
                    'type': 'detailed'
                })
            
            # Quality assessment
            processing_successful = data.get('demo_info', {}).get('processing_successful', True)
            if processing_successful:
                step['details'].append("Quality check: PASSED")
                step['fractal_nodes'].append({
                    'node': 'quality_check',
                    'symbol': '✓',
                    'status': 'passed'
                })
            else:
                step['details'].append("Quality check: ISSUES DETECTED")
                step['fractal_nodes'].append({
                    'node': 'quality_check',
                    'symbol': '⚠',
                    'status': 'warning'
                })
            
            return step
            
        except Exception as e:
            logging.error(f"Error tracing explanation: {e}")
            return {'step_number': 5, 'step_name': 'Decision Explanation', 'error': str(e)}
    
    def _generate_summary_trace(self, trace_steps: List[Dict]) -> str:
        """Generate human-readable summary trace"""
        try:
            summary_lines = ["Reasoning Trace:"]
            
            for step in trace_steps:
                if 'error' in step:
                    summary_lines.append(f"  - Step {step['step_number']}: {step['step_name']} (ERROR)")
                else:
                    summary_lines.append(f"  - Step {step['step_number']}: {step['step_name']}")
                    for detail in step.get('details', [])[:2]:  # Show top 2 details
                        summary_lines.append(f"    • {detail}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logging.error(f"Error generating summary trace: {e}")
            return "Trace generation failed"
    
    def _generate_fractal_visualization(self, fractal_graph: List[Dict]) -> str:
        """Generate fractal symbol visualization"""
        try:
            if not fractal_graph:
                return "○ → △"
            
            # Group nodes by type
            visualization_parts = []
            
            # Input analysis symbols
            input_symbols = [node['symbol'] for node in fractal_graph if 'emotion' in node.get('node', '') or 'trigger' in node.get('node', '')]
            if input_symbols:
                visualization_parts.append(" ".join(input_symbols[:3]))
            
            # Memory symbols
            memory_symbols = [node['symbol'] for node in fractal_graph if 'memory' in node.get('node', '')]
            if memory_symbols:
                visualization_parts.append(" ".join(memory_symbols[:2]))
            
            # Trait symbols
            trait_symbols = [node['symbol'] for node in fractal_graph if any(trait in node.get('node', '') for trait in ['adjust', 'dominant', 'primary', 'blend'])]
            if trait_symbols:
                visualization_parts.append(" ".join(trait_symbols[:3]))
            
            # Response symbol
            response_symbols = [node['symbol'] for node in fractal_graph if 'response' in node.get('node', '') or node.get('node') == 'template_response']
            if response_symbols:
                visualization_parts.append(response_symbols[0])
            
            return " → ".join(visualization_parts) if visualization_parts else "○ → △"
            
        except Exception as e:
            logging.error(f"Error generating fractal visualization: {e}")
            return "◇ ? △"
    
    def _calculate_confidence_score(self, trace_steps: List[Dict]) -> float:
        """Calculate confidence score for the reasoning trace"""
        try:
            total_score = 0.0
            step_count = 0
            
            for step in trace_steps:
                if 'error' in step:
                    total_score += 0.0
                else:
                    # Score based on detail completeness
                    detail_count = len(step.get('details', []))
                    node_count = len(step.get('fractal_nodes', []))
                    
                    step_score = min(1.0, (detail_count * 0.3) + (node_count * 0.2))
                    total_score += step_score
                
                step_count += 1
            
            return total_score / max(step_count, 1) if step_count > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_fallback_trace(self, error_message: str) -> Dict[str, Any]:
        """Generate fallback trace when main trace generation fails"""
        return {
            'trace_steps': [{
                'step_number': 1,
                'step_name': 'Trace Generation Error',
                'description': f'Error occurred: {error_message}',
                'details': ['Fallback trace generated'],
                'fractal_nodes': [{'node': 'error', 'symbol': '⚠', 'message': error_message}]
            }],
            'summary_trace': f"Trace Error: {error_message}",
            'fractal_graph': [{'node': 'error', 'symbol': '⚠'}],
            'fractal_visualization': "⚠ → △",
            'trace_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_steps': 1,
                'confidence_score': 0.0,
                'error': True
            }
        }
    
    def generate_trait_changelog(self, original_traits: Dict[str, float], 
                               updated_traits: Dict[str, float], 
                               reason: str = "") -> List[Dict[str, Any]]:
        """Generate trait change log entries"""
        try:
            changelog = []
            
            for trait, new_value in updated_traits.items():
                old_value = original_traits.get(trait, 0.5)
                change = new_value - old_value
                
                if abs(change) > 0.01:  # Significant change threshold
                    change_type = "increased" if change > 0 else "decreased"
                    symbol = self.fractal_symbols.get(trait, '?')
                    
                    changelog.append({
                        'trait': trait,
                        'symbol': symbol,
                        'old_value': old_value,
                        'new_value': new_value,
                        'change': change,
                        'change_type': change_type,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat(),
                        'message': f"{symbol} {trait.title()} {change_type} to {new_value:.2f} {reason}"
                    })
            
            return changelog
            
        except Exception as e:
            logging.error(f"Error generating trait changelog: {e}")
            return []


# Global instance
reasoning_trace_engine = ReasoningTraceEngine()