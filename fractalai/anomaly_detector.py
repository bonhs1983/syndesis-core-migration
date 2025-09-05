"""
Anomaly Detection & Proactive Insight Engine
Αναλύει patterns στις συζητήσεις και προβλέπει/προτείνει ενέργειες
"""
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import openai
import os

logger = logging.getLogger(__name__)

class ConversationAnomalyDetector:
    """
    Εντοπίζει anomalies και patterns στις συζητήσεις για proactive insights
    """
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Anomaly detection thresholds
        self.thresholds = {
            'sentiment_drop_threshold': -0.4,  # Σημαντική πτώση sentiment
            'repetitive_question_count': 3,    # Επαναλαμβανόμενες ερωτήσεις
            'session_gap_hours': 24,          # Μεγάλο χάσμα μεταξύ sessions
            'negative_streak_count': 4,       # Συνεχόμενα αρνητικά μηνύματα
            'topic_stagnation_count': 5,      # Κολλημένος στο ίδιο θέμα
            'trait_spike_threshold': 0.3      # Ξαφνική αλλαγή σε trait
        }
    
    def analyze_conversation_anomalies(self, user_sessions: List[Dict]) -> Dict[str, Any]:
        """
        Αναλύει όλες τις συζητήσεις ενός χρήστη για anomalies
        """
        if not user_sessions or len(user_sessions) < 2:
            return {"anomalies": [], "insights": [], "proactive_suggestions": []}
        
        anomalies = []
        insights = []
        proactive_suggestions = []
        
        # 1. Sentiment Analysis Over Time
        sentiment_anomalies = self._detect_sentiment_anomalies(user_sessions)
        anomalies.extend(sentiment_anomalies)
        
        # 2. Repetitive Pattern Detection
        repetitive_patterns = self._detect_repetitive_patterns(user_sessions)
        anomalies.extend(repetitive_patterns)
        
        # 3. Behavioral Change Detection
        behavioral_changes = self._detect_behavioral_changes(user_sessions)
        anomalies.extend(behavioral_changes)
        
        # 4. Topic Stagnation Detection
        topic_issues = self._detect_topic_stagnation(user_sessions)
        anomalies.extend(topic_issues)
        
        # 5. Session Gap Analysis
        session_gaps = self._detect_session_gaps(user_sessions)
        anomalies.extend(session_gaps)
        
        # Generate proactive insights and suggestions
        if anomalies:
            insights = self._generate_insights(anomalies, user_sessions)
            proactive_suggestions = self._generate_proactive_suggestions(anomalies, user_sessions)
        
        return {
            "anomalies": anomalies,
            "insights": insights, 
            "proactive_suggestions": proactive_suggestions,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _detect_sentiment_anomalies(self, sessions: List[Dict]) -> List[Dict]:
        """
        Εντοπίζει ξαφνικές αλλαγές στο sentiment
        """
        anomalies = []
        
        # Analyze sentiment trends
        sentiment_scores = []
        for session in sessions[-10:]:  # Τελευταία 10 sessions
            if session.get('agent_input') and session.get('agent_output'):
                # Simple sentiment analysis (θα μπορούσε να χρησιμοποιεί πιο advanced)
                user_text = session['agent_input'].lower()
                
                negative_words = ['stressed', 'sad', 'angry', 'frustrated', 'tired', 'overwhelmed', 'upset', 'worried', 'anxious', 'depressed', 'hopeless', 'confused', 'lost']
                positive_words = ['happy', 'excited', 'good', 'great', 'awesome', 'fantastic', 'wonderful', 'amazing', 'love', 'enjoy', 'pleased', 'satisfied', 'grateful']
                
                negative_count = sum(1 for word in negative_words if word in user_text)
                positive_count = sum(1 for word in positive_words if word in user_text)
                
                # Simple sentiment score: -1 to 1
                sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count + 1)
                sentiment_scores.append({
                    'timestamp': session.get('timestamp'),
                    'sentiment': sentiment,
                    'text': user_text[:100]  # First 100 chars for context
                })
        
        # Detect significant drops
        if len(sentiment_scores) >= 3:
            recent_avg = statistics.mean([s['sentiment'] for s in sentiment_scores[-3:]])
            overall_avg = statistics.mean([s['sentiment'] for s in sentiment_scores])
            
            if recent_avg - overall_avg < self.thresholds['sentiment_drop_threshold']:
                anomalies.append({
                    'type': 'sentiment_drop',
                    'severity': 'high',
                    'description': f'Significant sentiment drop detected (recent: {recent_avg:.2f} vs average: {overall_avg:.2f})',
                    'data': {
                        'recent_sentiment': recent_avg,
                        'overall_sentiment': overall_avg,
                        'drop_amount': recent_avg - overall_avg
                    }
                })
        
        # Detect negative streaks
        negative_streak = 0
        max_negative_streak = 0
        for score in sentiment_scores:
            if score['sentiment'] < -0.2:
                negative_streak += 1
                max_negative_streak = max(max_negative_streak, negative_streak)
            else:
                negative_streak = 0
        
        if max_negative_streak >= self.thresholds['negative_streak_count']:
            anomalies.append({
                'type': 'negative_streak',
                'severity': 'medium',
                'description': f'Extended negative sentiment streak detected ({max_negative_streak} consecutive interactions)',
                'data': {'streak_length': max_negative_streak}
            })
        
        return anomalies
    
    def _detect_repetitive_patterns(self, sessions: List[Dict]) -> List[Dict]:
        """
        Εντοπίζει επαναλαμβανόμενα patterns/ερωτήσεις
        """
        anomalies = []
        
        # Analyze question patterns
        user_inputs = [s.get('agent_input', '').lower() for s in sessions if s.get('agent_input')]
        
        # Detect similar questions
        question_similarity = defaultdict(int)
        keywords_counter = Counter()
        
        for input_text in user_inputs[-20:]:  # Τελευταία 20 inputs
            words = input_text.split()
            # Count question patterns
            if any(q in input_text for q in ['how', 'what', 'why', 'when', 'where', 'can you']):
                key_words = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'this', 'that', 'with', 'from']]
                if key_words:
                    pattern = ' '.join(sorted(key_words[:3]))  # Take top 3 keywords
                    question_similarity[pattern] += 1
            
            # Count overall keyword frequency
            for word in words:
                if len(word) > 3:
                    keywords_counter[word] += 1
        
        # Check for repetitive questions
        for pattern, count in question_similarity.items():
            if count >= self.thresholds['repetitive_question_count']:
                anomalies.append({
                    'type': 'repetitive_questions',
                    'severity': 'medium',
                    'description': f'User asking similar questions repeatedly: "{pattern}" ({count} times)',
                    'data': {
                        'pattern': pattern,
                        'frequency': count,
                        'suggestion': 'User might need deeper explanation or different approach'
                    }
                })
        
        return anomalies
    
    def _detect_behavioral_changes(self, sessions: List[Dict]) -> List[Dict]:
        """
        Εντοπίζει αλλαγές στη συμπεριφορά
        """
        anomalies = []
        
        if len(sessions) < 5:
            return anomalies
        
        # Analyze message length changes
        message_lengths = []
        for session in sessions:
            if session.get('agent_input'):
                length = len(session['agent_input'].split())
                message_lengths.append(length)
        
        if len(message_lengths) >= 5:
            recent_avg_length = statistics.mean(message_lengths[-5:])
            overall_avg_length = statistics.mean(message_lengths)
            
            # Detect sudden shift to very short messages (possible disengagement)
            if recent_avg_length < 3 and overall_avg_length > 8:
                anomalies.append({
                    'type': 'engagement_drop',
                    'severity': 'medium',
                    'description': 'User messages became significantly shorter, possible disengagement',
                    'data': {
                        'recent_avg_words': recent_avg_length,
                        'overall_avg_words': overall_avg_length
                    }
                })
            
            # Detect sudden shift to very long messages (possible frustration)
            elif recent_avg_length > overall_avg_length * 2 and recent_avg_length > 15:
                anomalies.append({
                    'type': 'verbosity_spike',
                    'severity': 'low',
                    'description': 'User messages became much longer, possible frustration or complex issue',
                    'data': {
                        'recent_avg_words': recent_avg_length,
                        'overall_avg_words': overall_avg_length
                    }
                })
        
        return anomalies
    
    def _detect_topic_stagnation(self, sessions: List[Dict]) -> List[Dict]:
        """
        Εντοπίζει κολλήματα στο ίδιο θέμα
        """
        anomalies = []
        
        # Extract topics from recent sessions
        recent_inputs = [s.get('agent_input', '') for s in sessions[-10:] if s.get('agent_input')]
        
        if len(recent_inputs) < 5:
            return anomalies
        
        # Simple topic extraction using keywords
        topics = []
        for input_text in recent_inputs:
            words = input_text.lower().split()
            # Focus on nouns and important keywords
            topic_words = [w for w in words if len(w) > 4 and w not in ['what', 'how', 'when', 'where', 'this', 'that', 'with', 'from', 'could', 'would', 'should']]
            if topic_words:
                topics.append(topic_words[0])  # Use first significant word as topic indicator
        
        # Check for topic stagnation
        if topics:
            topic_counter = Counter(topics)
            most_common_topic, frequency = topic_counter.most_common(1)[0]
            
            if frequency >= self.thresholds['topic_stagnation_count']:
                anomalies.append({
                    'type': 'topic_stagnation',
                    'severity': 'low',
                    'description': f'User seems focused on same topic repeatedly: "{most_common_topic}"',
                    'data': {
                        'topic': most_common_topic,
                        'frequency': frequency,
                        'total_interactions': len(recent_inputs)
                    }
                })
        
        return anomalies
    
    def _detect_session_gaps(self, sessions: List[Dict]) -> List[Dict]:
        """
        Εντοπίζει ασυνήθιστα χάσματα μεταξύ sessions
        """
        anomalies = []
        
        if len(sessions) < 2:
            return anomalies
        
        # Parse timestamps and find gaps
        timestamps = []
        for session in sessions:
            if session.get('timestamp'):
                try:
                    ts = datetime.fromisoformat(str(session['timestamp']).replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
        
        if len(timestamps) >= 2:
            timestamps.sort()
            gaps = []
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i-1]
                gaps.append(gap.total_seconds() / 3600)  # Convert to hours
            
            # Find unusual long gaps
            if gaps:
                avg_gap = statistics.mean(gaps)
                recent_gap = gaps[-1] if gaps else 0
                
                if recent_gap > self.thresholds['session_gap_hours'] and recent_gap > avg_gap * 3:
                    anomalies.append({
                        'type': 'long_absence',
                        'severity': 'low',
                        'description': f'User returned after unusually long absence ({recent_gap:.1f} hours)',
                        'data': {
                            'gap_hours': recent_gap,
                            'average_gap_hours': avg_gap
                        }
                    })
        
        return anomalies
    
    def _generate_insights(self, anomalies: List[Dict], sessions: List[Dict]) -> List[str]:
        """
        Δημιουργεί insights βάσει των anomalies
        """
        insights = []
        
        for anomaly in anomalies:
            if anomaly['type'] == 'sentiment_drop':
                insights.append("I've noticed a shift in your recent messages - you seem to be experiencing more challenges lately.")
            
            elif anomaly['type'] == 'repetitive_questions':
                pattern = anomaly['data']['pattern']
                insights.append(f"I see you've been asking about '{pattern}' several times - would you like me to approach this differently or provide more detailed guidance?")
            
            elif anomaly['type'] == 'engagement_drop':
                insights.append("Your messages have become shorter recently - if you're feeling overwhelmed or need a different approach, just let me know.")
            
            elif anomaly['type'] == 'topic_stagnation':
                topic = anomaly['data']['topic']
                insights.append(f"We've been focusing on '{topic}' for a while - would you like to explore related areas or take a different angle?")
            
            elif anomaly['type'] == 'long_absence':
                insights.append("Welcome back! It's been a while since our last conversation - anything new I should know about?")
        
        return insights
    
    def _generate_proactive_suggestions(self, anomalies: List[Dict], sessions: List[Dict]) -> List[str]:
        """
        Δημιουργεί proactive suggestions βάσει patterns
        """
        suggestions = []
        
        for anomaly in anomalies:
            if anomaly['type'] == 'sentiment_drop':
                suggestions.extend([
                    "Would you like to talk about what's been challenging lately?",
                    "Sometimes taking a step back helps - would you like some stress management techniques?",
                    "I'm here to listen if you need to vent or work through something difficult."
                ])
            
            elif anomaly['type'] == 'repetitive_questions':
                suggestions.extend([
                    "Let me try explaining this from a completely different perspective.",
                    "Would it help if I broke this down into smaller, more manageable steps?",
                    "Perhaps we could explore some practical examples or hands-on approaches?"
                ])
            
            elif anomaly['type'] == 'topic_stagnation':
                suggestions.extend([
                    "Would you like to explore how this connects to other areas of interest?",
                    "Maybe we could take a break from this topic and return to it with fresh perspective?",
                    "I could suggest some related resources or different approaches to consider."
                ])
            
            elif anomaly['type'] == 'engagement_drop':
                suggestions.extend([
                    "Would you prefer shorter, more focused responses?",
                    "If you're pressed for time, I can prioritize the most important points.",
                    "Let me know if you need a quick summary or want to continue this later."
                ])
        
        return list(set(suggestions))  # Remove duplicates
    
    def get_proactive_message(self, user_sessions: List[Dict]) -> Optional[str]:
        """
        Επιστρέφει proactive μήνυμα αν εντοπιστούν anomalies
        """
        analysis = self.analyze_conversation_anomalies(user_sessions)
        
        if analysis['anomalies']:
            # Prioritize by severity
            high_severity = [a for a in analysis['anomalies'] if a.get('severity') == 'high']
            medium_severity = [a for a in analysis['anomalies'] if a.get('severity') == 'medium']
            
            if high_severity and analysis['insights']:
                return analysis['insights'][0]  # Return first high-priority insight
            elif medium_severity and analysis['proactive_suggestions']:
                return analysis['proactive_suggestions'][0]  # Return first suggestion
        
        return None