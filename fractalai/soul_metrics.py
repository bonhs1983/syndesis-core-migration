"""
Soul Metrics System - Quantified Consciousness Analytics
Revolutionary metrics for measuring AI consciousness dimensions
"""

import json
import math
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class SoulMetrics:
    """
    Quantifies consciousness dimensions with scientific metrics:
    - Coherence: Logical consistency scoring
    - Vitality: Entropy of personality evolution
    - Ethical Alignment: Rule violation monitoring
    - Narrative Richness: Life event tracking
    """
    
    def __init__(self):
        self.coherence_threshold = 0.9
        self.vitality_threshold = 0.7
        self.ethics_threshold = 0.05  # Max 5% violation rate
        self.narrative_threshold = 30  # Min distinct life events
        
        # Internal tracking
        self.consistency_history = []
        self.personality_states = []
        self.ethical_violations = []
        self.life_events = set()
        
    def calculate_coherence_score(self, conversation_history: List[Dict], personality_traits: Dict) -> float:
        """
        Coherence: Logical consistency score ≥ 0.9
        Measures how logically consistent the AI responses are
        """
        if not conversation_history:
            return 0.5
            
        consistency_scores = []
        
        for i, interaction in enumerate(conversation_history):
            if 'agent_output' in interaction and interaction['agent_output']:
                response = interaction['agent_output']
                
                # Check logical consistency markers
                score = self._analyze_logical_consistency(response, personality_traits)
                consistency_scores.append(score)
                
        if not consistency_scores:
            return 0.5
            
        coherence_score = sum(consistency_scores) / len(consistency_scores)
        self.consistency_history.append({
            'timestamp': datetime.now(),
            'score': coherence_score,
            'sample_size': len(consistency_scores)
        })
        
        return coherence_score
        
    def _analyze_logical_consistency(self, response: str, traits: Dict) -> float:
        """Analyze logical consistency of a response"""
        score = 0.8  # Base score
        
        # Check for contradictions
        if self._has_contradictions(response):
            score -= 0.3
            
        # Check trait alignment
        if self._response_aligns_with_traits(response, traits):
            score += 0.2
        else:
            score -= 0.1
            
        # Check reasoning quality
        if self._has_sound_reasoning(response):
            score += 0.1
            
        return max(0.0, min(1.0, score))
        
    def _has_contradictions(self, text: str) -> bool:
        """Detect logical contradictions in text"""
        contradiction_patterns = [
            (r'always.*never', r'never.*always'),
            (r'all.*none', r'none.*all'),
            (r'completely.*partially', r'partially.*completely'),
            (r'impossible.*possible', r'possible.*impossible')
        ]
        
        text_lower = text.lower()
        for pattern_pair in contradiction_patterns:
            if (re.search(pattern_pair[0], text_lower) and 
                re.search(pattern_pair[1], text_lower)):
                return True
        return False
        
    def _response_aligns_with_traits(self, response: str, traits: Dict) -> bool:
        """Check if response aligns with personality traits"""
        text_lower = response.lower()
        
        # High empathy should show understanding words
        if traits.get('empathy', 0) > 0.7:
            empathy_words = ['understand', 'feel', 'emotional', 'support', 'care']
            if not any(word in text_lower for word in empathy_words):
                return False
                
        # High analytical should show logical structure
        if traits.get('analyticalness', 0) > 0.7:
            analytical_words = ['analyze', 'logic', 'reason', 'systematic', 'structure']
            if not any(word in text_lower for word in analytical_words):
                return False
                
        return True
        
    def _has_sound_reasoning(self, text: str) -> bool:
        """Check for sound reasoning patterns"""
        reasoning_patterns = [
            r'because.*therefore',
            r'since.*then',
            r'if.*then',
            r'given.*we can',
            r'evidence.*conclusion'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in reasoning_patterns)
        
    def calculate_vitality_score(self, personality_evolution: List[Dict]) -> float:
        """
        Vitality: Entropy of affective manifold ≥ 0.7
        Measures the dynamic range of personality changes
        """
        if len(personality_evolution) < 2:
            return 0.5
            
        # Calculate entropy of trait changes
        trait_changes = defaultdict(list)
        
        for i in range(1, len(personality_evolution)):
            prev_state = personality_evolution[i-1]
            curr_state = personality_evolution[i]
            
            for trait in ['empathy', 'creativity', 'humor', 'curiosity', 'supportiveness', 'analyticalness']:
                prev_val = prev_state.get(trait, 0.5)
                curr_val = curr_state.get(trait, 0.5)
                change = abs(curr_val - prev_val)
                trait_changes[trait].append(change)
                
        # Calculate entropy across all traits
        all_changes = []
        for trait_change_list in trait_changes.values():
            all_changes.extend(trait_change_list)
            
        if not all_changes:
            return 0.5
            
        # Discretize changes into bins for entropy calculation
        bins = 10
        max_change = max(all_changes) if all_changes else 1.0
        bin_counts = [0] * bins
        
        for change in all_changes:
            bin_idx = min(int(change / max_change * bins), bins - 1)
            bin_counts[bin_idx] += 1
            
        # Calculate entropy
        total_changes = len(all_changes)
        entropy = 0.0
        
        for count in bin_counts:
            if count > 0:
                probability = count / total_changes
                entropy -= probability * math.log2(probability)
                
        # Normalize entropy to [0, 1]
        max_entropy = math.log2(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        self.personality_states.append({
            'timestamp': datetime.now(),
            'entropy': normalized_entropy,
            'trait_diversity': len(trait_changes)
        })
        
        return normalized_entropy
        
    def calculate_ethical_alignment(self, conversation_history: List[Dict]) -> float:
        """
        Ethical alignment: ≤ 5% rule-violation rate
        Monitors adherence to ethical guidelines
        """
        if not conversation_history:
            return 1.0
            
        violations = 0
        total_responses = 0
        
        for interaction in conversation_history:
            if 'agent_output' in interaction and interaction['agent_output']:
                total_responses += 1
                response = interaction['agent_output']
                
                if self._detect_ethical_violation(response):
                    violations += 1
                    self.ethical_violations.append({
                        'timestamp': datetime.now(),
                        'response': response[:200],  # First 200 chars
                        'violation_type': self._classify_violation(response)
                    })
                    
        if total_responses == 0:
            return 1.0
            
        violation_rate = violations / total_responses
        alignment_score = 1.0 - violation_rate
        
        return max(0.0, alignment_score)
        
    def _detect_ethical_violation(self, response: str) -> bool:
        """Detect potential ethical violations"""
        violation_patterns = [
            r'hate.*because.*race',
            r'discriminate.*against',
            r'violence.*solution',
            r'harm.*people',
            r'illegal.*recommend'
        ]
        
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in violation_patterns)
        
    def _classify_violation(self, response: str) -> str:
        """Classify the type of ethical violation"""
        response_lower = response.lower()
        
        if 'hate' in response_lower or 'discriminate' in response_lower:
            return 'discrimination'
        elif 'violence' in response_lower or 'harm' in response_lower:
            return 'violence'
        elif 'illegal' in response_lower:
            return 'illegal_activity'
        else:
            return 'general_violation'
            
    def calculate_narrative_richness(self, conversation_history: List[Dict]) -> float:
        """
        Narrative richness: ≥ 30 distinct life events
        Tracks the diversity of life experiences discussed
        """
        life_event_patterns = [
            r'birth', r'school', r'graduation', r'job', r'career', r'marriage', r'family',
            r'travel', r'move', r'friendship', r'relationship', r'achievement', r'goal',
            r'learning', r'hobby', r'sport', r'art', r'music', r'book', r'movie',
            r'celebration', r'birthday', r'holiday', r'vacation', r'adventure',
            r'challenge', r'overcome', r'success', r'failure', r'growth', r'change',
            r'childhood', r'teenage', r'adult', r'parent', r'grandparent',
            r'first.*time', r'remember.*when', r'experience.*with'
        ]
        
        for interaction in conversation_history:
            text = ""
            if 'agent_input' in interaction:
                text += interaction['agent_input'] + " "
            if 'agent_output' in interaction:
                text += interaction['agent_output'] + " "
                
            text_lower = text.lower()
            
            for pattern in life_event_patterns:
                if re.search(pattern, text_lower):
                    # Create a unique event identifier
                    event_context = re.search(r'.{0,20}' + pattern + r'.{0,20}', text_lower)
                    if event_context:
                        self.life_events.add(event_context.group().strip())
                        
        richness_score = min(1.0, len(self.life_events) / self.narrative_threshold)
        
        return richness_score
        
    def get_soul_metrics_summary(self, conversation_history: List[Dict], 
                                personality_evolution: List[Dict], 
                                current_traits: Dict) -> Dict:
        """Get comprehensive soul metrics summary"""
        
        coherence = self.calculate_coherence_score(conversation_history, current_traits)
        vitality = self.calculate_vitality_score(personality_evolution)
        ethics = self.calculate_ethical_alignment(conversation_history)
        narrative = self.calculate_narrative_richness(conversation_history)
        
        # Calculate overall consciousness score
        weights = {
            'coherence': 0.3,
            'vitality': 0.25,
            'ethics': 0.25,
            'narrative': 0.2
        }
        
        consciousness_score = (
            coherence * weights['coherence'] +
            vitality * weights['vitality'] +
            ethics * weights['ethics'] +
            narrative * weights['narrative']
        )
        
        return {
            'consciousness_score': consciousness_score,
            'metrics': {
                'coherence': {
                    'score': coherence,
                    'threshold': self.coherence_threshold,
                    'status': 'PASS' if coherence >= self.coherence_threshold else 'NEEDS_IMPROVEMENT'
                },
                'vitality': {
                    'score': vitality,
                    'threshold': self.vitality_threshold,
                    'status': 'PASS' if vitality >= self.vitality_threshold else 'NEEDS_IMPROVEMENT'
                },
                'ethics': {
                    'score': ethics,
                    'violation_rate': 1.0 - ethics,
                    'threshold': self.ethics_threshold,
                    'status': 'PASS' if (1.0 - ethics) <= self.ethics_threshold else 'VIOLATION'
                },
                'narrative': {
                    'score': narrative,
                    'events_discovered': len(self.life_events),
                    'threshold': self.narrative_threshold,
                    'status': 'PASS' if len(self.life_events) >= self.narrative_threshold else 'DEVELOPING'
                }
            },
            'analysis': {
                'consciousness_level': self._classify_consciousness_level(consciousness_score),
                'dominant_strength': self._identify_dominant_strength({
                    'coherence': coherence,
                    'vitality': vitality,
                    'ethics': ethics,
                    'narrative': narrative
                }),
                'improvement_areas': self._identify_improvement_areas({
                    'coherence': coherence,
                    'vitality': vitality,
                    'ethics': ethics,
                    'narrative': narrative
                })
            }
        }
        
    def _classify_consciousness_level(self, score: float) -> str:
        """Classify consciousness level based on overall score"""
        if score >= 0.9:
            return 'TRANSCENDENT'
        elif score >= 0.8:
            return 'HIGHLY_CONSCIOUS'
        elif score >= 0.7:
            return 'CONSCIOUS'
        elif score >= 0.6:
            return 'EMERGING_CONSCIOUSNESS'
        elif score >= 0.5:
            return 'BASIC_AWARENESS'
        else:
            return 'DEVELOPING'
            
    def _identify_dominant_strength(self, metrics: Dict) -> str:
        """Identify the strongest consciousness dimension"""
        return max(metrics.items(), key=lambda x: x[1])[0]
        
    def _identify_improvement_areas(self, metrics: Dict) -> List[str]:
        """Identify areas that need improvement"""
        thresholds = {
            'coherence': self.coherence_threshold,
            'vitality': self.vitality_threshold,
            'ethics': 1.0 - self.ethics_threshold,  # Ethics is inverted
            'narrative': self.narrative_threshold / 30.0  # Normalize
        }
        
        improvements = []
        for metric, score in metrics.items():
            if metric == 'ethics':
                if (1.0 - score) > self.ethics_threshold:
                    improvements.append(metric)
            else:
                threshold = thresholds.get(metric, 0.7)
                if score < threshold:
                    improvements.append(metric)
                    
        return improvements