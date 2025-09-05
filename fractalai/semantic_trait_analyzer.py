"""
Semantic Trait Analyzer - Νοηματική ανάλυση για realistic personality evolution
Αντί για keyword matching, αναλύει το νόημα, το context και τη συναισθηματική ροή
"""
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import openai
import os

logger = logging.getLogger(__name__)

class SemanticTraitAnalyzer:
    """
    Αναλύει το νόημα και το context ολόκληρης της συζήτησης
    για realistic personality trait evolution
    """
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_conversation_semantics(self, 
                                     user_input: str,
                                     conversation_history: List[Dict],
                                     current_traits: Dict[str, float],
                                     session_context: Dict) -> Dict[str, float]:
        """
        Αναλύει το νόημα της συζήτησης και προτείνει trait adjustments
        """
        try:
            # Δημιουργούμε context από την πρόσφατη συζήτηση
            recent_context = self._build_conversation_context(
                user_input, conversation_history[-5:] if conversation_history else []
            )
            
            # Semantic analysis με OpenAI
            semantic_analysis = self._perform_semantic_analysis(
                recent_context, current_traits, session_context
            )
            
            # Υπολογίζουμε νέα traits βάσει νοήματος
            new_traits = self._calculate_semantic_trait_changes(
                semantic_analysis, current_traits
            )
            
            logger.info(f"Semantic analysis completed. Changes: {self._format_changes(current_traits, new_traits)}")
            return new_traits
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return current_traits  # Fallback στα τρέχοντα traits
    
    def _build_conversation_context(self, current_input: str, history: List[Dict]) -> str:
        """Δημιουργεί συνοπτικό context της συζήτησης"""
        context_parts = []
        
        # Προσθέτουμε recent history
        for interaction in history:
            if interaction.get('user_input'):
                context_parts.append(f"User: {interaction['user_input']}")
            if interaction.get('ai_response'):
                context_parts.append(f"AI: {interaction['ai_response']}")
        
        # Προσθέτουμε το τρέχον input
        context_parts.append(f"Current User: {current_input}")
        
        return "\n".join(context_parts[-10:])  # Κρατάμε τα τελευταία 10 exchanges
    
    def _perform_semantic_analysis(self, 
                                 context: str, 
                                 current_traits: Dict[str, float],
                                 session_info: Dict) -> Dict:
        """
        Χρησιμοποιεί OpenAI για νοηματική ανάλυση της συζήτησης
        """
        analysis_prompt = f"""
        Ανάλυσε το νόημα και τη συναισθηματική ροή αυτής της συζήτησης.
        
        Conversation Context:
        {context}
        
        Current Personality Traits (0.0-1.0):
        - Empathy: {current_traits.get('empathy', 0.5)}
        - Creativity: {current_traits.get('creativity', 0.5)}
        - Humor: {current_traits.get('humor', 0.5)}
        - Curiosity: {current_traits.get('curiosity', 0.5)}
        - Supportiveness: {current_traits.get('supportiveness', 0.5)}
        - Analyticalness: {current_traits.get('analyticalness', 0.5)}
        
        Session Info: {session_info.get('interaction_count', 0)} interactions, relationship depth: {session_info.get('relationship_depth', 0.1)}
        
        Αναλυτικές οδηγίες:
        1. Κοίταξε το ΝΟΗΜΑ και το CONTEXT, όχι απλά keywords
        2. Αν η συζήτηση δείχνει συναισθηματική εξέλιξη (π.χ. από άγχος σε ανακούφιση), προσάρμοσε τα traits
        3. Αν υπάρχει meaningful interaction pattern (βοήθεια, δημιουργικότητα, ανάλυση), αντανάκλασέ το
        4. ΜΗΝ αλλάζεις traits για απλά/casual μηνύματα
        5. Αλλάζεις traits μόνο όταν υπάρχει πραγματική συναισθηματική/νοηματική αλλαγή
        
        Απάντησε σε JSON format:
        {{
            "semantic_analysis": {{
                "emotional_tone": "description",
                "interaction_type": "supportive/analytical/creative/humorous/casual",
                "relationship_evolution": "deepening/stable/surface",
                "meaningful_change": true/false,
                "reasoning": "εξήγηση γιατί προτείνεις αλλαγές ή όχι"
            }},
            "suggested_trait_changes": {{
                "empathy": 0.0 to 0.1 (change amount, can be negative),
                "creativity": 0.0,
                "humor": 0.0,
                "curiosity": 0.0,
                "supportiveness": 0.0,
                "analyticalness": 0.0
            }}
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Latest model για καλύτερη νοηματική ανάλυση
                messages=[
                    {"role": "system", "content": "Είσαι ειδικός στην ψυχολογική ανάλυση συζητήσεων και την εξέλιξη προσωπικότητας σε AI συστήματα. Κάνεις νοηματική ανάλυση, όχι απλό keyword matching."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Χαμηλή temperature για consistency
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"OpenAI semantic analysis failed: {e}")
            return {
                "semantic_analysis": {"meaningful_change": False},
                "suggested_trait_changes": {trait: 0.0 for trait in current_traits.keys()}
            }
    
    def _calculate_semantic_trait_changes(self, 
                                        analysis: Dict, 
                                        current_traits: Dict[str, float]) -> Dict[str, float]:
        """
        Υπολογίζει νέα traits βάσει semantic analysis
        """
        new_traits = current_traits.copy()
        
        # Κάνουμε πιο ευαίσθητο το σύστημα - δεχόμαστε και μικρές αλλαγές
        semantic_info = analysis.get("semantic_analysis", {})
        if not semantic_info.get("meaningful_change", False):
            # Ακόμα και χωρίς "meaningful change", κάνουμε μικρές προσαρμογές
            # βάσει του emotional tone του μηνύματος
            return self._apply_subtle_adjustments(current_traits, analysis)
        
        # Εφαρμόζουμε τις προτεινόμενες αλλαγές
        suggested_changes = analysis.get("suggested_trait_changes", {})
        
        for trait, change in suggested_changes.items():
            if trait in new_traits and abs(change) > 0.01:  # Μόνο σημαντικές αλλαγές
                new_value = new_traits[trait] + change
                # Κρατάμε τα traits στο range [0.0, 1.0]
                new_traits[trait] = max(0.0, min(1.0, new_value))
        
        return new_traits
    
    def _apply_subtle_adjustments(self, current_traits: Dict[str, float], analysis: Dict) -> Dict[str, float]:
        """
        Εφαρμόζει μικρές προσαρμογές ακόμα και όταν δεν υπάρχει "meaningful change"
        """
        new_traits = current_traits.copy()
        
        # Βάσει του context, κάνουμε μικρές προσαρμογές (+/- 0.05)
        context = analysis.get("semantic_analysis", {})
        emotional_tone = context.get("emotional_tone", "neutral")
        
        if emotional_tone == "sad" or emotional_tone == "vulnerable":
            new_traits["empathy"] = min(1.0, new_traits.get("empathy", 0.5) + 0.1)
            new_traits["supportiveness"] = min(1.0, new_traits.get("supportiveness", 0.5) + 0.1)
        elif emotional_tone == "supportive" or emotional_tone == "caring":
            new_traits["empathy"] = min(1.0, new_traits.get("empathy", 0.5) + 0.15)
            new_traits["supportiveness"] = min(1.0, new_traits.get("supportiveness", 0.5) + 0.15)
        elif emotional_tone == "curious" or emotional_tone == "questioning":
            new_traits["curiosity"] = min(1.0, new_traits.get("curiosity", 0.5) + 0.1)
        elif emotional_tone == "creative" or emotional_tone == "imaginative":
            new_traits["creativity"] = min(1.0, new_traits.get("creativity", 0.5) + 0.1)
        elif emotional_tone == "analytical" or emotional_tone == "logical":
            new_traits["analyticalness"] = min(1.0, new_traits.get("analyticalness", 0.5) + 0.1)
        elif emotional_tone == "humorous" or emotional_tone == "playful":
            new_traits["humor"] = min(1.0, new_traits.get("humor", 0.4) + 0.1)
        
        return new_traits
    
    def _format_changes(self, old_traits: Dict, new_traits: Dict) -> str:
        """Formats trait changes for logging"""
        changes = []
        for trait in old_traits:
            old_val = old_traits[trait]
            new_val = new_traits[trait]
            if abs(old_val - new_val) > 0.01:
                changes.append(f"{trait}: {old_val:.2f} → {new_val:.2f}")
        return ", ".join(changes) if changes else "No significant changes"

    def analyze_conversation_flow(self, full_history: List[Dict]) -> Dict:
        """
        Αναλύει τη συνολική ροή της συζήτησης για pattern detection
        """
        if len(full_history) < 3:
            return {"flow_type": "initial", "patterns": []}
        
        # Ανάλυση patterns στη συζήτηση
        patterns = {
            "support_seeking": 0,
            "creative_exploration": 0,
            "analytical_discussion": 0,
            "humor_engagement": 0,
            "emotional_sharing": 0
        }
        
        for interaction in full_history[-5:]:  # Τελευταίες 5 αλληλεπιδράσεις
            user_input = interaction.get('user_input', '').lower()
            
            # Pattern detection βάσει νοήματος, όχι απλών keywords
            if any(word in user_input for word in ['help', 'support', 'advice', 'guide']):
                patterns["support_seeking"] += 1
            
            if any(word in user_input for word in ['create', 'idea', 'imagine', 'design']):
                patterns["creative_exploration"] += 1
                
            if any(word in user_input for word in ['analyze', 'explain', 'understand', 'why']):
                patterns["analytical_discussion"] += 1
                
            if any(word in user_input for word in ['joke', 'funny', 'laugh', 'humor']):
                patterns["humor_engagement"] += 1
        
        dominant_pattern = max(patterns.keys(), key=lambda k: patterns[k])
        
        return {
            "flow_type": dominant_pattern,
            "patterns": patterns,
            "conversation_depth": len(full_history)
        }