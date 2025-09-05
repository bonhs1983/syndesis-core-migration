"""
Trait Contrast Engine - Demonstrates dramatic differences between personality modes
Allows users to request comparative responses showing high vs low trait examples
"""
import logging
from typing import Dict, List, Tuple


class TraitContrastEngine:
    """Generates contrasting responses to demonstrate trait differences"""
    
    def __init__(self):
        self.contrast_templates = {
            'empathy': {
                'high': [
                    "I completely understand how you're feeling right now. Your emotions are valid and important, and I'm here to support you through this. Let's work together to find a solution that feels right for you.",
                    "Καταλαβαίνω απόλυτα τι περνάς. Τα συναισθήματά σου είναι φυσιολογικά και σημαντικά, και είμαι εδώ να σε στηρίξω. Ας βρούμε μαζί μια λύση που θα σε κάνει να νιώθεις καλύτερα."
                ],
                'low': [
                    "I don't see why this would be emotionally significant. The facts are straightforward - here's the information you requested without unnecessary emotional context.",
                    "Δεν καταλαβαίνω γιατί σε απασχολεί συναισθηματικά αυτό. Τα γεγονότα είναι απλά - να η πληροφορία που ζήτησες χωρίς περιττό συναισθηματικό περιεχόμενο."
                ]
            },
            'humor': {
                'high': [
                    "Haha, that's a great question! *chuckles* I'm excited to tackle this with you - and I promise to keep things entertaining while we figure it out together. Ready for some fun problem-solving?",
                    "Χαχα, τέλεια ερώτηση! *γελάει* Ανυπομονώ να το λύσουμε μαζί - και υπόσχομαι να κρατήσω τα πράγματα διασκεδαστικά ενώ το ψάχνουμε. Έτοιμος για λίγη πλάκα;"
                ],
                'low': [
                    "This is a standard inquiry that requires a factual response. I will provide the requested information without embellishment or attempts at entertainment.",
                    "Αυτή είναι μια τυπική ερώτηση που χρειάζεται πραγματική απάντηση. Θα σου δώσω την πληροφορία χωρίς περιτυλίγματα ή προσπάθειες διασκέδασης."
                ]
            },
            'creativity': {
                'high': [
                    "What a fascinating question! This opens up so many creative possibilities - I can imagine multiple innovative approaches we could explore. Let me think outside the box and share some imaginative perspectives with you.",
                    "Τι συναρπαστική ερώτηση! Αυτό ανοίγει τόσες δημιουργικές δυνατότητες - μπορώ να φανταστώ πολλαπλές καινοτόμες προσεγγίσεις. Ας σκεφτώ έξω από το κουτί και να μοιραστώ μερικές φαντασιόπληκτες οπτικές."
                ],
                'low': [
                    "I'll provide a conventional, standard response based on established practices. There's no need for creative interpretation - the straightforward approach is most appropriate here.",
                    "Θα δώσω μια συμβατική, τυπική απάντηση βασισμένη σε καθιερωμένες πρακτικές. Δεν χρειάζεται δημιουργική ερμηνεία - η απλή προσέγγιση είναι πιο κατάλληλη εδώ."
                ]
            },
            'analytical': {
                'high': [
                    "Let me analyze this systematically: 1) Core problem identification, 2) Evidence evaluation, 3) Logical conclusion derivation. I'll structure this response with precise reasoning and factual accuracy.",
                    "Ας το αναλύσω συστηματικά: 1) Εντοπισμός βασικού προβλήματος, 2) Αξιολόγηση στοιχείων, 3) Λογική εξαγωγή συμπεράσματος. Θα δομήσω την απάντηση με ακριβή συλλογισμό."
                ],
                'low': [
                    "I'll go with my gut feeling on this one. Sometimes intuition and emotional response are more valuable than cold analysis - let me share what feels right rather than what the data strictly suggests.",
                    "Θα πάω με το ένστικτό μου σε αυτό. Μερικές φορές η διαίσθηση και η συναισθηματική ανταπόκριση είναι πιο πολύτιμες από την ψυχρή ανάλυση."
                ]
            }
        }
    
    def generate_trait_contrast(self, user_input: str, trait: str, context: Dict = None) -> Dict[str, str]:
        """
        Generate high vs low contrast responses for a specific trait
        Returns dict with 'high' and 'low' response examples
        """
        if trait not in self.contrast_templates:
            return {
                'high': f"High {trait} response would be more intense in that trait dimension.",
                'low': f"Low {trait} response would be minimal in that trait expression.",
                'explanation': f"Contrast templates for '{trait}' not yet implemented."
            }
        
        templates = self.contrast_templates[trait]
        
        import random
        high_template = random.choice(templates['high'])
        low_template = random.choice(templates['low'])
        
        logging.info(f"TraitContrastEngine: Generated contrast for trait '{trait}'")
        
        return {
            'high': high_template,
            'low': low_template,
            'explanation': f"This demonstrates the dramatic difference between high {trait} ({trait} > 0.8) and low {trait} ({trait} < 0.3) responses to the same input.",
            'trait': trait,
            'user_input': user_input
        }
    
    def explain_trait_difference(self, trait: str, high_score: float, low_score: float) -> str:
        """Explain what causes the difference between high and low trait responses"""
        explanations = {
            'empathy': f"High empathy ({high_score:.1f}) focuses on emotional validation and support, while low empathy ({low_score:.1f}) prioritizes factual information without emotional consideration.",
            'humor': f"High humor ({high_score:.1f}) uses playful language and entertainment, while low humor ({low_score:.1f}) maintains serious, straightforward communication.",
            'creativity': f"High creativity ({high_score:.1f}) explores innovative and imaginative approaches, while low creativity ({low_score:.1f}) sticks to conventional, established methods.",
            'analytical': f"High analytical ({high_score:.1f}) uses systematic logic and structured reasoning, while low analytical ({low_score:.1f}) relies more on intuition and emotional response."
        }
        
        return explanations.get(trait, f"High {trait} shows strong expression of that trait, while low {trait} shows minimal expression.")
    
    def get_available_contrasts(self) -> List[str]:
        """Get list of traits that support contrast generation"""
        return list(self.contrast_templates.keys())
    
    def demonstrate_all_contrasts(self, user_input: str) -> Dict[str, Dict]:
        """Generate contrasts for all available traits"""
        all_contrasts = {}
        
        for trait in self.contrast_templates.keys():
            all_contrasts[trait] = self.generate_trait_contrast(user_input, trait)
        
        return all_contrasts