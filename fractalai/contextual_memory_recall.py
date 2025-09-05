"""
Contextual Memory Recall System
Retrieves and integrates relevant conversation history based on context and topics
"""
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import re


class ContextualMemoryRecall:
    """
    Manages contextual memory recall for conversations
    """
    
    def __init__(self):
        self.conversation_memory = {}  # session_id -> conversation history
        self.topic_memory = {}  # session_id -> topic-based memory
        self.keyword_memory = {}  # session_id -> keyword-based index
        
    def store_conversation_turn(self, session_id: str, user_input: str, 
                              ai_response: str, context: Dict = None) -> None:
        """Store a conversation turn with contextual information"""
        try:
            if session_id not in self.conversation_memory:
                self.conversation_memory[session_id] = []
                self.topic_memory[session_id] = {}
                self.keyword_memory[session_id] = {}
            
            # Create conversation entry
            turn_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'ai_response': ai_response,
                'context': context or {},
                'topics': self._extract_topics(user_input),
                'keywords': self._extract_keywords(user_input),
                'turn_id': len(self.conversation_memory[session_id])
            }
            
            # Store in conversation history
            self.conversation_memory[session_id].append(turn_entry)
            
            # Update topic-based memory
            self._update_topic_memory(session_id, turn_entry)
            
            # Update keyword-based memory
            self._update_keyword_memory(session_id, turn_entry)
            
            # Trim old memories if needed (keep last 100 turns)
            if len(self.conversation_memory[session_id]) > 100:
                removed_turn = self.conversation_memory[session_id].pop(0)
                self._remove_from_indices(session_id, removed_turn)
            
        except Exception as e:
            logging.error(f"Error storing conversation turn: {e}")
    
    def recall_relevant_context(self, session_id: str, current_input: str, 
                              max_items: int = 3) -> Dict[str, Any]:
        """Recall relevant conversation context based on current input"""
        try:
            if session_id not in self.conversation_memory:
                return {'relevant_memories': [], 'explanation': 'No conversation history available'}
            
            # Extract topics and keywords from current input
            current_topics = self._extract_topics(current_input)
            current_keywords = self._extract_keywords(current_input)
            
            # Find relevant memories using multiple strategies
            topic_matches = self._find_topic_matches(session_id, current_topics)
            keyword_matches = self._find_keyword_matches(session_id, current_keywords)
            recency_matches = self._find_recent_context(session_id, current_input)
            
            # Combine and score matches
            all_matches = self._combine_and_score_matches(
                topic_matches, keyword_matches, recency_matches
            )
            
            # Select top matches
            relevant_memories = all_matches[:max_items]
            
            # Generate integration suggestions
            integration_text = self._generate_integration_text(relevant_memories, current_input)
            
            return {
                'relevant_memories': relevant_memories,
                'integration_text': integration_text,
                'explanation': self._explain_recall_reasoning(relevant_memories, current_topics, current_keywords),
                'recall_strategies_used': {
                    'topic_matches': len(topic_matches),
                    'keyword_matches': len(keyword_matches),
                    'recency_matches': len(recency_matches)
                }
            }
            
        except Exception as e:
            logging.error(f"Error recalling relevant context: {e}")
            return {
                'relevant_memories': [],
                'explanation': f'Error during memory recall: {str(e)}',
                'error': str(e)
            }
    
    def handle_explicit_recall_request(self, session_id: str, query: str) -> Dict[str, Any]:
        """Handle explicit requests like 'what did I say about X?'"""
        try:
            if session_id not in self.conversation_memory:
                return {
                    'found_memories': [],
                    'response': "I don't have any conversation history to search through.",
                    'query_type': 'explicit_recall'
                }
            
            # Parse the recall query
            query_info = self._parse_recall_query(query)
            
            # Search based on query type
            if query_info['type'] == 'topic_search':
                matches = self._search_by_topic(session_id, query_info['target'])
            elif query_info['type'] == 'keyword_search':
                matches = self._search_by_keywords(session_id, query_info['target'])
            elif query_info['type'] == 'time_based':
                matches = self._search_by_time(session_id, query_info['target'])
            else:
                matches = self._general_search(session_id, query_info['target'])
            
            # Generate response
            if matches:
                response = self._format_explicit_recall_response(matches, query_info)
            else:
                response = f"I couldn't find anything in our conversation about '{query_info['target']}'."
            
            return {
                'found_memories': matches,
                'response': response,
                'query_type': 'explicit_recall',
                'search_target': query_info['target'],
                'search_method': query_info['type']
            }
            
        except Exception as e:
            logging.error(f"Error handling explicit recall request: {e}")
            return {
                'found_memories': [],
                'response': f"I had trouble searching through our conversation: {str(e)}",
                'query_type': 'explicit_recall',
                'error': str(e)
            }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        try:
            text_lower = text.lower()
            
            # Predefined topic categories
            topic_patterns = {
                'work': ['work', 'job', 'career', 'office', 'boss', 'colleague', 'project', 'meeting'],
                'family': ['family', 'mother', 'father', 'mom', 'dad', 'sister', 'brother', 'child', 'kids'],
                'health': ['health', 'doctor', 'medicine', 'sick', 'pain', 'hospital', 'treatment'],
                'technology': ['computer', 'software', 'app', 'internet', 'website', 'tech', 'digital'],
                'education': ['school', 'university', 'study', 'learn', 'student', 'teacher', 'class'],
                'relationships': ['friend', 'partner', 'boyfriend', 'girlfriend', 'relationship', 'dating'],
                'hobbies': ['music', 'sports', 'reading', 'cooking', 'travel', 'movies', 'games'],
                'emotions': ['happy', 'sad', 'angry', 'excited', 'worried', 'stressed', 'love', 'hate']
            }
            
            detected_topics = []
            for topic, keywords in topic_patterns.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_topics.append(topic)
            
            # Also extract explicit noun phrases (simple approach)
            noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            detected_topics.extend(noun_phrases[:3])  # Limit to 3 noun phrases
            
            return list(set(detected_topics))  # Remove duplicates
            
        except Exception as e:
            logging.error(f"Error extracting topics: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        try:
            # Remove common stop words
            stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
            
            # Extract words, filter stop words, and keep meaningful ones
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            keywords = [word for word in words if word not in stop_words]
            
            # Return most frequent/important keywords (max 10)
            keyword_freq = {}
            for keyword in keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
            return [keyword for keyword, freq in sorted_keywords[:10]]
            
        except Exception as e:
            logging.error(f"Error extracting keywords: {e}")
            return []
    
    def _update_topic_memory(self, session_id: str, turn_entry: Dict) -> None:
        """Update topic-based memory index"""
        try:
            for topic in turn_entry['topics']:
                if topic not in self.topic_memory[session_id]:
                    self.topic_memory[session_id][topic] = []
                
                self.topic_memory[session_id][topic].append({
                    'turn_id': turn_entry['turn_id'],
                    'timestamp': turn_entry['timestamp'],
                    'relevance_score': 1.0  # Can be enhanced with more sophisticated scoring
                })
                
        except Exception as e:
            logging.error(f"Error updating topic memory: {e}")
    
    def _update_keyword_memory(self, session_id: str, turn_entry: Dict) -> None:
        """Update keyword-based memory index"""
        try:
            for keyword in turn_entry['keywords']:
                if keyword not in self.keyword_memory[session_id]:
                    self.keyword_memory[session_id][keyword] = []
                
                self.keyword_memory[session_id][keyword].append({
                    'turn_id': turn_entry['turn_id'],
                    'timestamp': turn_entry['timestamp'],
                    'frequency': turn_entry['keywords'].count(keyword)
                })
                
        except Exception as e:
            logging.error(f"Error updating keyword memory: {e}")
    
    def _find_topic_matches(self, session_id: str, current_topics: List[str]) -> List[Dict]:
        """Find conversation turns matching current topics"""
        try:
            matches = []
            topic_memory = self.topic_memory.get(session_id, {})
            
            for topic in current_topics:
                if topic in topic_memory:
                    for memory_entry in topic_memory[topic]:
                        turn_id = memory_entry['turn_id']
                        if turn_id < len(self.conversation_memory[session_id]):
                            turn = self.conversation_memory[session_id][turn_id]
                            matches.append({
                                'turn': turn,
                                'match_type': 'topic',
                                'match_reason': f"Topic: {topic}",
                                'relevance_score': memory_entry['relevance_score']
                            })
            
            return matches
            
        except Exception as e:
            logging.error(f"Error finding topic matches: {e}")
            return []
    
    def _find_keyword_matches(self, session_id: str, current_keywords: List[str]) -> List[Dict]:
        """Find conversation turns matching current keywords"""
        try:
            matches = []
            keyword_memory = self.keyword_memory.get(session_id, {})
            
            for keyword in current_keywords:
                if keyword in keyword_memory:
                    for memory_entry in keyword_memory[keyword]:
                        turn_id = memory_entry['turn_id']
                        if turn_id < len(self.conversation_memory[session_id]):
                            turn = self.conversation_memory[session_id][turn_id]
                            matches.append({
                                'turn': turn,
                                'match_type': 'keyword',
                                'match_reason': f"Keyword: {keyword}",
                                'relevance_score': memory_entry['frequency'] * 0.5
                            })
            
            return matches
            
        except Exception as e:
            logging.error(f"Error finding keyword matches: {e}")
            return []
    
    def _find_recent_context(self, session_id: str, current_input: str) -> List[Dict]:
        """Find recent relevant context"""
        try:
            matches = []
            conversation = self.conversation_memory.get(session_id, [])
            
            # Look at last 10 turns for recency-based relevance
            recent_turns = conversation[-10:] if len(conversation) > 10 else conversation
            
            for turn in recent_turns:
                # Simple relevance scoring based on recency and content similarity
                age_hours = (datetime.now() - datetime.fromisoformat(turn['timestamp'])).total_seconds() / 3600
                recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
                
                # Basic content similarity (shared words)
                current_words = set(current_input.lower().split())
                turn_words = set((turn['user_input'] + ' ' + turn['ai_response']).lower().split())
                similarity = len(current_words & turn_words) / max(len(current_words | turn_words), 1)
                
                relevance_score = (recency_score * 0.7) + (similarity * 0.3)
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    matches.append({
                        'turn': turn,
                        'match_type': 'recency',
                        'match_reason': f"Recent context (relevance: {relevance_score:.2f})",
                        'relevance_score': relevance_score
                    })
            
            return matches
            
        except Exception as e:
            logging.error(f"Error finding recent context: {e}")
            return []
    
    def _combine_and_score_matches(self, *match_lists) -> List[Dict]:
        """Combine matches from different strategies and score them"""
        try:
            all_matches = []
            turn_id_seen = set()
            
            # Combine all matches, avoiding duplicates
            for match_list in match_lists:
                for match in match_list:
                    turn_id = match['turn']['turn_id']
                    if turn_id not in turn_id_seen:
                        all_matches.append(match)
                        turn_id_seen.add(turn_id)
                    else:
                        # If we've seen this turn, boost its score
                        existing_match = next(m for m in all_matches if m['turn']['turn_id'] == turn_id)
                        existing_match['relevance_score'] += match['relevance_score'] * 0.5
                        existing_match['match_reason'] += f"; {match['match_reason']}"
            
            # Sort by relevance score
            all_matches.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return all_matches
            
        except Exception as e:
            logging.error(f"Error combining and scoring matches: {e}")
            return []
    
    def _generate_integration_text(self, relevant_memories: List[Dict], current_input: str) -> str:
        """Generate text to integrate memories into current response"""
        try:
            if not relevant_memories:
                return ""
            
            integration_phrases = []
            
            for memory in relevant_memories:
                turn = memory['turn']
                user_text = turn['user_input'][:100]  # Truncate for integration
                
                # Create contextual reference
                if memory['match_type'] == 'topic':
                    phrase = f"Earlier, you mentioned {user_text}"
                elif memory['match_type'] == 'keyword':
                    phrase = f"I remember you talking about {user_text}"
                else:
                    phrase = f"In our recent conversation about {user_text}"
                
                integration_phrases.append(phrase)
            
            if len(integration_phrases) == 1:
                return integration_phrases[0] + ". "
            elif len(integration_phrases) == 2:
                return f"{integration_phrases[0]}, and {integration_phrases[1]}. "
            else:
                return f"{integration_phrases[0]}, among other things we've discussed. "
                
        except Exception as e:
            logging.error(f"Error generating integration text: {e}")
            return ""
    
    def _explain_recall_reasoning(self, relevant_memories: List[Dict], 
                                current_topics: List[str], current_keywords: List[str]) -> str:
        """Explain why these memories were recalled"""
        try:
            if not relevant_memories:
                return "No relevant memories found for this conversation."
            
            reasons = []
            for memory in relevant_memories:
                reasons.append(memory['match_reason'])
            
            explanation = f"Recalled {len(relevant_memories)} relevant memories based on: {', '.join(reasons)}"
            
            if current_topics:
                explanation += f". Current topics detected: {', '.join(current_topics)}"
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error explaining recall reasoning: {e}")
            return "Unable to explain memory recall reasoning."
    
    def _parse_recall_query(self, query: str) -> Dict[str, Any]:
        """Parse explicit recall queries like 'what did I say about X?'"""
        try:
            query_lower = query.lower()
            
            # Detect query patterns
            if any(phrase in query_lower for phrase in ['what did i say', 'did i mention', 'i said']):
                query_type = 'topic_search'
            elif any(phrase in query_lower for phrase in ['when did', 'what time', 'earlier today']):
                query_type = 'time_based'
            elif any(phrase in query_lower for phrase in ['about', 'regarding', 'concerning']):
                query_type = 'keyword_search'
            else:
                query_type = 'general_search'
            
            # Extract target (what they're asking about)
            target_patterns = [
                r'about (.+?)(?:\?|$)',
                r'regarding (.+?)(?:\?|$)',
                r'concerning (.+?)(?:\?|$)',
                r'what did i say (.+?)(?:\?|$)',
                r'did i mention (.+?)(?:\?|$)'
            ]
            
            target = None
            for pattern in target_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    target = match.group(1).strip()
                    break
            
            if not target:
                # Fallback: extract meaningful words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
                stop_words = {'what', 'did', 'say', 'about', 'the', 'and', 'but', 'for'}
                target = ' '.join([w for w in words if w not in stop_words])
            
            return {
                'type': query_type,
                'target': target or query,
                'original_query': query
            }
            
        except Exception as e:
            logging.error(f"Error parsing recall query: {e}")
            return {'type': 'general_search', 'target': query, 'original_query': query}
    
    def _search_by_topic(self, session_id: str, topic: str) -> List[Dict]:
        """Search conversation by topic"""
        try:
            matches = []
            conversation = self.conversation_memory.get(session_id, [])
            topic_lower = topic.lower()
            
            for turn in conversation:
                turn_topics = [t.lower() for t in turn.get('topics', [])]
                if any(topic_lower in t or t in topic_lower for t in turn_topics):
                    matches.append({
                        'turn': turn,
                        'match_score': 1.0,
                        'match_detail': f"Topic match: {topic}"
                    })
            
            return matches[-5:]  # Return last 5 matches
            
        except Exception as e:
            logging.error(f"Error searching by topic: {e}")
            return []
    
    def _search_by_keywords(self, session_id: str, keywords: str) -> List[Dict]:
        """Search conversation by keywords"""
        try:
            matches = []
            conversation = self.conversation_memory.get(session_id, [])
            keyword_list = keywords.lower().split()
            
            for turn in conversation:
                text = (turn['user_input'] + ' ' + turn['ai_response']).lower()
                match_count = sum(1 for kw in keyword_list if kw in text)
                
                if match_count > 0:
                    matches.append({
                        'turn': turn,
                        'match_score': match_count / len(keyword_list),
                        'match_detail': f"Keyword matches: {match_count}/{len(keyword_list)}"
                    })
            
            # Sort by match score and return top matches
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            return matches[:5]
            
        except Exception as e:
            logging.error(f"Error searching by keywords: {e}")
            return []
    
    def _search_by_time(self, session_id: str, time_query: str) -> List[Dict]:
        """Search conversation by time reference"""
        try:
            matches = []
            conversation = self.conversation_memory.get(session_id, [])
            
            # Simple time-based search (can be enhanced)
            if 'recent' in time_query or 'earlier' in time_query:
                # Return recent conversations
                matches = [{'turn': turn, 'match_score': 1.0, 'match_detail': 'Recent conversation'} 
                          for turn in conversation[-5:]]
            
            return matches
            
        except Exception as e:
            logging.error(f"Error searching by time: {e}")
            return []
    
    def _general_search(self, session_id: str, query: str) -> List[Dict]:
        """General search through conversation"""
        try:
            return self._search_by_keywords(session_id, query)
            
        except Exception as e:
            logging.error(f"Error in general search: {e}")
            return []
    
    def _format_explicit_recall_response(self, matches: List[Dict], query_info: Dict) -> str:
        """Format response for explicit recall requests"""
        try:
            if not matches:
                return f"I couldn't find anything in our conversation about '{query_info['target']}'."
            
            if len(matches) == 1:
                turn = matches[0]['turn']
                return f"Yes, you mentioned: \"{turn['user_input'][:200]}...\""
            else:
                response = f"I found {len(matches)} times you mentioned something about '{query_info['target']}':\n\n"
                for i, match in enumerate(matches[:3], 1):
                    turn = match['turn']
                    response += f"{i}. \"{turn['user_input'][:150]}...\"\n"
                
                if len(matches) > 3:
                    response += f"\n...and {len(matches) - 3} more instances."
                
                return response
                
        except Exception as e:
            logging.error(f"Error formatting explicit recall response: {e}")
            return "I had trouble formatting the memory recall response."
    
    def _remove_from_indices(self, session_id: str, removed_turn: Dict) -> None:
        """Remove old turn from topic and keyword indices"""
        try:
            turn_id = removed_turn['turn_id']
            
            # Remove from topic memory
            for topic in removed_turn.get('topics', []):
                if topic in self.topic_memory[session_id]:
                    self.topic_memory[session_id][topic] = [
                        entry for entry in self.topic_memory[session_id][topic]
                        if entry['turn_id'] != turn_id
                    ]
            
            # Remove from keyword memory
            for keyword in removed_turn.get('keywords', []):
                if keyword in self.keyword_memory[session_id]:
                    self.keyword_memory[session_id][keyword] = [
                        entry for entry in self.keyword_memory[session_id][keyword]
                        if entry['turn_id'] != turn_id
                    ]
                    
        except Exception as e:
            logging.error(f"Error removing from indices: {e}")
    
    def get_memory_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get memory statistics for a session"""
        try:
            if session_id not in self.conversation_memory:
                return {'error': 'No memory found for session'}
            
            conversation = self.conversation_memory[session_id]
            topic_memory = self.topic_memory.get(session_id, {})
            keyword_memory = self.keyword_memory.get(session_id, {})
            
            return {
                'total_turns': len(conversation),
                'unique_topics': len(topic_memory),
                'unique_keywords': len(keyword_memory),
                'memory_span_hours': self._calculate_memory_span(conversation),
                'most_frequent_topics': self._get_most_frequent_topics(topic_memory),
                'most_frequent_keywords': self._get_most_frequent_keywords(keyword_memory)
            }
            
        except Exception as e:
            logging.error(f"Error getting memory statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_memory_span(self, conversation: List[Dict]) -> float:
        """Calculate time span of conversation memory"""
        try:
            if len(conversation) < 2:
                return 0.0
            
            oldest = datetime.fromisoformat(conversation[0]['timestamp'])
            newest = datetime.fromisoformat(conversation[-1]['timestamp'])
            
            return (newest - oldest).total_seconds() / 3600  # Return hours
            
        except Exception as e:
            logging.error(f"Error calculating memory span: {e}")
            return 0.0
    
    def _get_most_frequent_topics(self, topic_memory: Dict) -> List[Tuple[str, int]]:
        """Get most frequently mentioned topics"""
        try:
            topic_counts = [(topic, len(entries)) for topic, entries in topic_memory.items()]
            topic_counts.sort(key=lambda x: x[1], reverse=True)
            return topic_counts[:5]
            
        except Exception as e:
            logging.error(f"Error getting frequent topics: {e}")
            return []
    
    def _get_most_frequent_keywords(self, keyword_memory: Dict) -> List[Tuple[str, int]]:
        """Get most frequently used keywords"""
        try:
            keyword_counts = [(keyword, len(entries)) for keyword, entries in keyword_memory.items()]
            keyword_counts.sort(key=lambda x: x[1], reverse=True)
            return keyword_counts[:5]
            
        except Exception as e:
            logging.error(f"Error getting frequent keywords: {e}")
            return []


# Global instance for use across the application
contextual_memory_recall = ContextualMemoryRecall()