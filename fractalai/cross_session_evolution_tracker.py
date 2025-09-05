"""
Cross-Session Personality Evolution Tracker
Tracks how personality traits evolve across multiple sessions and time periods
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import statistics


class CrossSessionEvolutionTracker:
    """
    Tracks personality evolution across sessions with comprehensive analytics
    """
    
    def __init__(self, persistent_personality):
        self.persistent_personality = persistent_personality
        self.evolution_history = []
        self.session_snapshots = {}
        
    def track_personality_evolution(self, session_id: str, event_type: str = 'interaction') -> Dict:
        """
        Track personality state at this moment for evolution analysis
        """
        try:
            current_personality = self.persistent_personality.get_personality(session_id)
            
            # Create evolution snapshot
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'personality_state': current_personality.copy(),
                'event_type': event_type,
                'evolution_metrics': self._calculate_evolution_metrics(session_id, current_personality)
            }
            
            # Store snapshot
            self.evolution_history.append(snapshot)
            
            # Update session snapshots
            if session_id not in self.session_snapshots:
                self.session_snapshots[session_id] = []
            self.session_snapshots[session_id].append(snapshot)
            
            # Analyze evolution patterns
            evolution_analysis = self._analyze_current_evolution(session_id, current_personality)
            
            return {
                'success': True,
                'snapshot_created': True,
                'current_personality': current_personality,
                'evolution_metrics': snapshot['evolution_metrics'],
                'evolution_analysis': evolution_analysis,
                'timestamp': snapshot['timestamp']
            }
            
        except Exception as e:
            logging.error(f"Error tracking personality evolution: {e}")
            return {'error': f'Failed to track evolution: {str(e)}'}
    
    def get_cross_session_analytics(self, session_ids: List[str] = None, days_back: int = 30) -> Dict:
        """
        Get comprehensive cross-session personality analytics
        """
        try:
            # Filter evolution history
            cutoff_date = datetime.now() - timedelta(days=days_back)
            relevant_history = [
                snapshot for snapshot in self.evolution_history
                if (datetime.fromisoformat(snapshot['timestamp']) > cutoff_date and
                    (not session_ids or snapshot['session_id'] in session_ids))
            ]
            
            if not relevant_history:
                return {
                    'analysis_period': f'{days_back} days',
                    'sessions_analyzed': 0,
                    'total_snapshots': 0,
                    'evolution_summary': 'No data available for analysis'
                }
            
            # Group by session
            session_data = {}
            for snapshot in relevant_history:
                session_id = snapshot['session_id']
                if session_id not in session_data:
                    session_data[session_id] = []
                session_data[session_id].append(snapshot)
            
            # Calculate cross-session metrics
            cross_session_metrics = self._calculate_cross_session_metrics(session_data)
            
            # Analyze evolution patterns
            evolution_patterns = self._analyze_evolution_patterns(session_data)
            
            # Calculate personality stability
            stability_analysis = self._calculate_personality_stability(session_data)
            
            # Identify evolution trends
            evolution_trends = self._identify_evolution_trends(session_data)
            
            return {
                'analysis_period': f'{days_back} days',
                'sessions_analyzed': len(session_data),
                'total_snapshots': len(relevant_history),
                'cross_session_metrics': cross_session_metrics,
                'evolution_patterns': evolution_patterns,
                'stability_analysis': stability_analysis,
                'evolution_trends': evolution_trends,
                'session_summaries': self._generate_session_summaries(session_data)
            }
            
        except Exception as e:
            logging.error(f"Error getting cross-session analytics: {e}")
            return {'error': f'Failed to get analytics: {str(e)}'}
    
    def compare_personality_across_sessions(self, session_ids: List[str]) -> Dict:
        """
        Compare personality traits across different sessions
        """
        try:
            if len(session_ids) < 2:
                return {'error': 'At least 2 sessions required for comparison'}
            
            # Get latest personality for each session
            session_personalities = {}
            for session_id in session_ids:
                personality = self.persistent_personality.get_personality(session_id)
                session_personalities[session_id] = personality
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(session_personalities)
            
            # Identify differences and similarities
            differences = self._identify_personality_differences(session_personalities)
            similarities = self._identify_personality_similarities(session_personalities)
            
            # Generate evolution pathways
            evolution_pathways = self._analyze_evolution_pathways(session_ids)
            
            return {
                'sessions_compared': session_ids,
                'session_personalities': session_personalities,
                'comparison_metrics': comparison_metrics,
                'key_differences': differences,
                'similarities': similarities,
                'evolution_pathways': evolution_pathways,
                'relationship_analysis': self._analyze_session_relationships(session_personalities)
            }
            
        except Exception as e:
            logging.error(f"Error comparing personalities across sessions: {e}")
            return {'error': f'Failed to compare sessions: {str(e)}'}
    
    def get_personality_timeline(self, session_id: str, trait_focus: str = None) -> Dict:
        """
        Get detailed timeline of personality evolution for a specific session
        """
        try:
            session_snapshots = self.session_snapshots.get(session_id, [])
            
            if not session_snapshots:
                return {
                    'session_id': session_id,
                    'timeline_length': 0,
                    'message': 'No evolution data available for this session'
                }
            
            # Sort by timestamp
            session_snapshots.sort(key=lambda x: x['timestamp'])
            
            # Build timeline
            timeline = []
            for i, snapshot in enumerate(session_snapshots):
                timeline_point = {
                    'timestamp': snapshot['timestamp'],
                    'personality_state': snapshot['personality_state'],
                    'event_type': snapshot['event_type'],
                    'evolution_metrics': snapshot['evolution_metrics']
                }
                
                # Add change analysis if not first snapshot
                if i > 0:
                    previous_snapshot = session_snapshots[i-1]
                    changes = self._analyze_snapshot_changes(previous_snapshot, snapshot)
                    timeline_point['changes_from_previous'] = changes
                
                timeline.append(timeline_point)
            
            # Focus on specific trait if requested
            if trait_focus:
                trait_timeline = self._extract_trait_timeline(timeline, trait_focus)
                return {
                    'session_id': session_id,
                    'trait_focus': trait_focus,
                    'timeline_length': len(timeline),
                    'trait_timeline': trait_timeline,
                    'trait_evolution_summary': self._summarize_trait_evolution(trait_timeline, trait_focus)
                }
            
            # Calculate overall evolution summary
            evolution_summary = self._calculate_timeline_summary(timeline)
            
            return {
                'session_id': session_id,
                'timeline_length': len(timeline),
                'full_timeline': timeline,
                'evolution_summary': evolution_summary,
                'most_changed_traits': self._identify_most_changed_traits(timeline),
                'stability_periods': self._identify_stability_periods(timeline)
            }
            
        except Exception as e:
            logging.error(f"Error getting personality timeline: {e}")
            return {'error': f'Failed to get timeline: {str(e)}'}
    
    def predict_personality_trajectory(self, session_id: str, days_ahead: int = 7) -> Dict:
        """
        Predict how personality might evolve based on current trends
        """
        try:
            # Get recent evolution data
            recent_snapshots = self._get_recent_snapshots(session_id, days_back=14)
            
            if len(recent_snapshots) < 3:
                return {
                    'session_id': session_id,
                    'prediction_confidence': 'Low',
                    'message': 'Insufficient data for reliable prediction'
                }
            
            # Calculate evolution velocities
            trait_velocities = self._calculate_trait_velocities(recent_snapshots)
            
            # Generate predictions
            predictions = {}
            current_personality = self.persistent_personality.get_personality(session_id)
            
            for trait, velocity in trait_velocities.items():
                current_value = current_personality[trait]
                predicted_change = velocity * days_ahead
                predicted_value = max(0.0, min(1.0, current_value + predicted_change))
                
                predictions[trait] = {
                    'current_value': current_value,
                    'predicted_value': predicted_value,
                    'predicted_change': predicted_change,
                    'velocity': velocity,
                    'confidence': self._calculate_prediction_confidence(trait, recent_snapshots)
                }
            
            # Identify notable predicted changes
            notable_changes = self._identify_notable_predictions(predictions)
            
            # Calculate overall prediction confidence
            overall_confidence = self._calculate_overall_prediction_confidence(predictions)
            
            return {
                'session_id': session_id,
                'prediction_horizon': f'{days_ahead} days',
                'current_personality': current_personality,
                'predictions': predictions,
                'notable_predicted_changes': notable_changes,
                'overall_confidence': overall_confidence,
                'prediction_basis': f'{len(recent_snapshots)} recent data points'
            }
            
        except Exception as e:
            logging.error(f"Error predicting personality trajectory: {e}")
            return {'error': f'Failed to predict trajectory: {str(e)}'}
    
    def get_evolution_insights(self, session_id: str = None) -> Dict:
        """
        Get insights about personality evolution patterns
        """
        try:
            # Filter data by session if specified
            if session_id:
                relevant_snapshots = self.session_snapshots.get(session_id, [])
                scope = f'session {session_id}'
            else:
                relevant_snapshots = self.evolution_history
                scope = 'all sessions'
            
            if not relevant_snapshots:
                return {
                    'scope': scope,
                    'insights': ['No evolution data available']
                }
            
            # Generate various insights
            insights = []
            
            # Evolution velocity insights
            velocity_insights = self._generate_velocity_insights(relevant_snapshots)
            insights.extend(velocity_insights)
            
            # Stability insights
            stability_insights = self._generate_stability_insights(relevant_snapshots)
            insights.extend(stability_insights)
            
            # Pattern insights
            pattern_insights = self._generate_pattern_insights(relevant_snapshots)
            insights.extend(pattern_insights)
            
            # Trend insights
            trend_insights = self._generate_trend_insights(relevant_snapshots)
            insights.extend(trend_insights)
            
            return {
                'scope': scope,
                'total_snapshots_analyzed': len(relevant_snapshots),
                'insights': insights,
                'insight_categories': {
                    'velocity': len(velocity_insights),
                    'stability': len(stability_insights),
                    'patterns': len(pattern_insights),
                    'trends': len(trend_insights)
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting evolution insights: {e}")
            return {'error': f'Failed to get insights: {str(e)}'}
    
    def _calculate_evolution_metrics(self, session_id: str, current_personality: Dict[str, float]) -> Dict:
        """
        Calculate metrics for current evolution state
        """
        metrics = {}
        
        # Get previous snapshots for this session
        previous_snapshots = self.session_snapshots.get(session_id, [])
        
        if previous_snapshots:
            # Compare with most recent snapshot
            last_snapshot = previous_snapshots[-1]
            last_personality = last_snapshot['personality_state']
            
            # Calculate trait changes
            trait_changes = {}
            total_change = 0
            for trait, current_value in current_personality.items():
                if trait in last_personality:
                    change = abs(current_value - last_personality[trait])
                    trait_changes[trait] = change
                    total_change += change
            
            metrics['total_change_since_last'] = total_change
            metrics['most_changed_trait'] = max(trait_changes.keys(), key=trait_changes.get) if trait_changes else None
            metrics['stability_score'] = max(0, 1 - total_change)
            
            # Calculate evolution velocity
            time_diff = datetime.now() - datetime.fromisoformat(last_snapshot['timestamp'])
            hours_diff = time_diff.total_seconds() / 3600
            if hours_diff > 0:
                metrics['evolution_velocity'] = total_change / hours_diff
            else:
                metrics['evolution_velocity'] = 0
        else:
            # First snapshot for this session
            metrics['total_change_since_last'] = 0
            metrics['most_changed_trait'] = None
            metrics['stability_score'] = 1.0
            metrics['evolution_velocity'] = 0
        
        return metrics
    
    def _analyze_current_evolution(self, session_id: str, current_personality: Dict[str, float]) -> Dict:
        """
        Analyze current evolution state and patterns
        """
        analysis = {}
        
        # Get evolution history for this session
        session_snapshots = self.session_snapshots.get(session_id, [])
        
        if len(session_snapshots) >= 2:
            # Analyze evolution direction
            analysis['evolution_direction'] = self._calculate_evolution_direction(session_snapshots)
            
            # Check for convergence or divergence
            analysis['convergence_status'] = self._analyze_convergence(session_snapshots)
            
            # Identify evolution phase
            analysis['evolution_phase'] = self._identify_evolution_phase(session_snapshots)
        else:
            analysis['evolution_direction'] = 'Insufficient data'
            analysis['convergence_status'] = 'Unknown'
            analysis['evolution_phase'] = 'Initial'
        
        return analysis
    
    def _calculate_cross_session_metrics(self, session_data: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate metrics across all sessions
        """
        metrics = {}
        
        # Overall statistics
        total_snapshots = sum(len(snapshots) for snapshots in session_data.values())
        metrics['total_snapshots'] = total_snapshots
        metrics['average_snapshots_per_session'] = total_snapshots / len(session_data)
        
        # Evolution activity
        all_changes = []
        for session_snapshots in session_data.values():
            for snapshot in session_snapshots:
                evolution_metrics = snapshot.get('evolution_metrics', {})
                total_change = evolution_metrics.get('total_change_since_last', 0)
                if total_change > 0:
                    all_changes.append(total_change)
        
        if all_changes:
            metrics['average_evolution_rate'] = statistics.mean(all_changes)
            metrics['evolution_variability'] = statistics.stdev(all_changes) if len(all_changes) > 1 else 0
        else:
            metrics['average_evolution_rate'] = 0
            metrics['evolution_variability'] = 0
        
        # Session diversity
        session_personalities = []
        for session_id, snapshots in session_data.items():
            if snapshots:
                latest_personality = snapshots[-1]['personality_state']
                session_personalities.append(latest_personality)
        
        if len(session_personalities) > 1:
            metrics['cross_session_diversity'] = self._calculate_personality_diversity(session_personalities)
        else:
            metrics['cross_session_diversity'] = 0
        
        return metrics
    
    def _analyze_evolution_patterns(self, session_data: Dict[str, List[Dict]]) -> Dict:
        """
        Analyze patterns in personality evolution
        """
        patterns = {}
        
        # Common evolution directions
        evolution_directions = []
        for session_snapshots in session_data.values():
            if len(session_snapshots) >= 2:
                direction = self._calculate_evolution_direction(session_snapshots)
                evolution_directions.append(direction)
        
        if evolution_directions:
            patterns['common_evolution_patterns'] = self._identify_common_patterns(evolution_directions)
        
        # Trait-specific patterns
        trait_patterns = {}
        all_traits = set()
        for session_snapshots in session_data.values():
            for snapshot in session_snapshots:
                all_traits.update(snapshot['personality_state'].keys())
        
        for trait in all_traits:
            trait_evolution = self._extract_trait_evolution_across_sessions(session_data, trait)
            trait_patterns[trait] = self._analyze_trait_pattern(trait_evolution)
        
        patterns['trait_specific_patterns'] = trait_patterns
        
        return patterns
    
    def _calculate_personality_stability(self, session_data: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate personality stability metrics
        """
        stability = {}
        
        # Overall stability
        all_stability_scores = []
        for session_snapshots in session_data.values():
            for snapshot in session_snapshots:
                evolution_metrics = snapshot.get('evolution_metrics', {})
                stability_score = evolution_metrics.get('stability_score', 1.0)
                all_stability_scores.append(stability_score)
        
        if all_stability_scores:
            stability['overall_stability'] = statistics.mean(all_stability_scores)
            stability['stability_consistency'] = 1 - statistics.stdev(all_stability_scores) if len(all_stability_scores) > 1 else 1
        
        # Per-session stability
        session_stability = {}
        for session_id, snapshots in session_data.items():
            session_scores = [
                snapshot.get('evolution_metrics', {}).get('stability_score', 1.0)
                for snapshot in snapshots
            ]
            if session_scores:
                session_stability[session_id] = statistics.mean(session_scores)
        
        stability['per_session_stability'] = session_stability
        
        # Most/least stable sessions
        if session_stability:
            stability['most_stable_session'] = max(session_stability.keys(), key=session_stability.get)
            stability['least_stable_session'] = min(session_stability.keys(), key=session_stability.get)
        
        return stability
    
    def _identify_evolution_trends(self, session_data: Dict[str, List[Dict]]) -> Dict:
        """
        Identify trends in personality evolution
        """
        trends = {}
        
        # Trait increase/decrease trends
        trait_trends = {}
        all_traits = set()
        for session_snapshots in session_data.values():
            for snapshot in session_snapshots:
                all_traits.update(snapshot['personality_state'].keys())
        
        for trait in all_traits:
            trait_values = []
            timestamps = []
            
            for session_snapshots in session_data.values():
                for snapshot in session_snapshots:
                    trait_value = snapshot['personality_state'].get(trait, 0.5)
                    trait_values.append(trait_value)
                    timestamps.append(datetime.fromisoformat(snapshot['timestamp']))
            
            if len(trait_values) > 1:
                # Simple trend calculation
                trend_direction = self._calculate_simple_trend(trait_values)
                trait_trends[trait] = {
                    'direction': trend_direction,
                    'strength': abs(trait_values[-1] - trait_values[0]),
                    'consistency': self._calculate_trend_consistency(trait_values)
                }
        
        trends['trait_trends'] = trait_trends
        
        # Overall evolution trend
        trends['overall_trend'] = self._calculate_overall_evolution_trend(session_data)
        
        return trends
    
    def _generate_session_summaries(self, session_data: Dict[str, List[Dict]]) -> Dict:
        """
        Generate summaries for each session
        """
        summaries = {}
        
        for session_id, snapshots in session_data.items():
            if not snapshots:
                continue
            
            # Basic statistics
            first_snapshot = snapshots[0]
            last_snapshot = snapshots[-1]
            
            # Calculate total evolution
            total_evolution = sum(
                snapshot.get('evolution_metrics', {}).get('total_change_since_last', 0)
                for snapshot in snapshots[1:]  # Skip first snapshot
            )
            
            # Identify most active traits
            trait_activity = {}
            for snapshot in snapshots[1:]:
                evolution_metrics = snapshot.get('evolution_metrics', {})
                most_changed = evolution_metrics.get('most_changed_trait')
                if most_changed:
                    trait_activity[most_changed] = trait_activity.get(most_changed, 0) + 1
            
            most_active_trait = max(trait_activity.keys(), key=trait_activity.get) if trait_activity else None
            
            summaries[session_id] = {
                'total_snapshots': len(snapshots),
                'evolution_period': self._calculate_time_span(first_snapshot['timestamp'], last_snapshot['timestamp']),
                'total_evolution': total_evolution,
                'most_active_trait': most_active_trait,
                'evolution_velocity': total_evolution / max(1, len(snapshots) - 1),
                'stability_trend': self._calculate_session_stability_trend(snapshots)
            }
        
        return summaries
    
    def _calculate_comparison_metrics(self, session_personalities: Dict[str, Dict[str, float]]) -> Dict:
        """
        Calculate metrics for comparing personalities across sessions
        """
        metrics = {}
        
        session_ids = list(session_personalities.keys())
        personalities = list(session_personalities.values())
        
        if len(personalities) >= 2:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(personalities)):
                for j in range(i + 1, len(personalities)):
                    similarity = self._calculate_personality_similarity(personalities[i], personalities[j])
                    similarities.append({
                        'sessions': [session_ids[i], session_ids[j]],
                        'similarity': similarity
                    })
            
            metrics['pairwise_similarities'] = similarities
            metrics['average_similarity'] = statistics.mean([s['similarity'] for s in similarities])
            
            # Most/least similar pairs
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            metrics['most_similar_pair'] = similarities[0]
            metrics['least_similar_pair'] = similarities[-1]
        
        return metrics
    
    def _identify_personality_differences(self, session_personalities: Dict[str, Dict[str, float]]) -> Dict:
        """
        Identify key differences between session personalities
        """
        differences = {}
        
        # Calculate trait variance across sessions
        all_traits = set()
        for personality in session_personalities.values():
            all_traits.update(personality.keys())
        
        trait_variances = {}
        for trait in all_traits:
            values = [personality.get(trait, 0.5) for personality in session_personalities.values()]
            if len(values) > 1:
                trait_variances[trait] = statistics.variance(values)
        
        # Identify most variable traits
        if trait_variances:
            most_variable_trait = max(trait_variances.keys(), key=trait_variances.get)
            differences['most_variable_trait'] = {
                'trait': most_variable_trait,
                'variance': trait_variances[most_variable_trait],
                'values_by_session': {
                    session_id: personality.get(most_variable_trait, 0.5)
                    for session_id, personality in session_personalities.items()
                }
            }
        
        # Identify unique characteristics
        unique_characteristics = {}
        for session_id, personality in session_personalities.items():
            # Find traits that are notably high or low compared to other sessions
            distinctive_traits = []
            for trait, value in personality.items():
                other_values = [
                    other_personality.get(trait, 0.5)
                    for other_session_id, other_personality in session_personalities.items()
                    if other_session_id != session_id
                ]
                
                if other_values:
                    avg_other = statistics.mean(other_values)
                    if abs(value - avg_other) > 0.3:  # Significant difference
                        distinctive_traits.append({
                            'trait': trait,
                            'value': value,
                            'difference_from_others': value - avg_other
                        })
            
            if distinctive_traits:
                unique_characteristics[session_id] = distinctive_traits
        
        differences['unique_characteristics'] = unique_characteristics
        
        return differences
    
    def _identify_personality_similarities(self, session_personalities: Dict[str, Dict[str, float]]) -> Dict:
        """
        Identify similarities between session personalities
        """
        similarities = {}
        
        # Find common trait patterns
        common_patterns = []
        all_traits = set()
        for personality in session_personalities.values():
            all_traits.update(personality.keys())
        
        for trait in all_traits:
            values = [personality.get(trait, 0.5) for personality in session_personalities.values()]
            
            # Check if all sessions have similar values for this trait
            if len(values) > 1:
                trait_variance = statistics.variance(values)
                if trait_variance < 0.05:  # Low variance indicates similarity
                    common_patterns.append({
                        'trait': trait,
                        'common_value': statistics.mean(values),
                        'consistency': 1 - trait_variance
                    })
        
        similarities['common_trait_patterns'] = common_patterns
        
        # Identify personality archetypes
        archetypes = self._identify_personality_archetypes(session_personalities)
        similarities['shared_archetypes'] = archetypes
        
        return similarities
    
    def _analyze_evolution_pathways(self, session_ids: List[str]) -> Dict:
        """
        Analyze evolution pathways between sessions
        """
        pathways = {}
        
        # Get evolution history for each session
        for session_id in session_ids:
            session_snapshots = self.session_snapshots.get(session_id, [])
            if len(session_snapshots) >= 2:
                pathway = {
                    'initial_state': session_snapshots[0]['personality_state'],
                    'final_state': session_snapshots[-1]['personality_state'],
                    'evolution_trajectory': self._calculate_evolution_trajectory(session_snapshots),
                    'key_turning_points': self._identify_turning_points(session_snapshots)
                }
                pathways[session_id] = pathway
        
        return pathways
    
    def _analyze_session_relationships(self, session_personalities: Dict[str, Dict[str, float]]) -> Dict:
        """
        Analyze relationships between different sessions
        """
        relationships = {}
        
        # Calculate relationship matrix
        session_ids = list(session_personalities.keys())
        relationship_matrix = {}
        
        for i, session1 in enumerate(session_ids):
            relationship_matrix[session1] = {}
            for j, session2 in enumerate(session_ids):
                if i != j:
                    similarity = self._calculate_personality_similarity(
                        session_personalities[session1],
                        session_personalities[session2]
                    )
                    relationship_matrix[session1][session2] = similarity
        
        relationships['similarity_matrix'] = relationship_matrix
        
        # Identify clusters
        clusters = self._identify_personality_clusters(session_personalities)
        relationships['personality_clusters'] = clusters
        
        return relationships
    
    def _analyze_snapshot_changes(self, previous_snapshot: Dict, current_snapshot: Dict) -> Dict:
        """
        Analyze changes between two snapshots
        """
        changes = {}
        
        prev_personality = previous_snapshot['personality_state']
        curr_personality = current_snapshot['personality_state']
        
        trait_changes = {}
        for trait in curr_personality:
            if trait in prev_personality:
                change = curr_personality[trait] - prev_personality[trait]
                if abs(change) > 0.01:  # Only significant changes
                    trait_changes[trait] = {
                        'change': change,
                        'direction': 'increase' if change > 0 else 'decrease',
                        'magnitude': abs(change)
                    }
        
        changes['trait_changes'] = trait_changes
        changes['total_change'] = sum(data['magnitude'] for data in trait_changes.values())
        changes['most_changed_trait'] = max(trait_changes.keys(), key=lambda t: trait_changes[t]['magnitude']) if trait_changes else None
        
        # Time between snapshots
        time_diff = datetime.fromisoformat(current_snapshot['timestamp']) - datetime.fromisoformat(previous_snapshot['timestamp'])
        changes['time_elapsed'] = time_diff.total_seconds() / 3600  # Hours
        
        return changes
    
    def _extract_trait_timeline(self, timeline: List[Dict], trait: str) -> List[Dict]:
        """
        Extract timeline data for a specific trait
        """
        trait_timeline = []
        
        for point in timeline:
            trait_value = point['personality_state'].get(trait, 0.5)
            trait_point = {
                'timestamp': point['timestamp'],
                'trait_value': trait_value,
                'event_type': point['event_type']
            }
            
            # Add change information if available
            if 'changes_from_previous' in point:
                trait_changes = point['changes_from_previous'].get('trait_changes', {})
                if trait in trait_changes:
                    trait_point['change'] = trait_changes[trait]
            
            trait_timeline.append(trait_point)
        
        return trait_timeline
    
    def _summarize_trait_evolution(self, trait_timeline: List[Dict], trait: str) -> Dict:
        """
        Summarize evolution for a specific trait
        """
        if not trait_timeline:
            return {'message': 'No data available for trait'}
        
        values = [point['trait_value'] for point in trait_timeline]
        
        summary = {
            'initial_value': values[0],
            'final_value': values[-1],
            'total_change': values[-1] - values[0],
            'max_value': max(values),
            'min_value': min(values),
            'volatility': statistics.stdev(values) if len(values) > 1 else 0,
            'trend': self._calculate_simple_trend(values)
        }
        
        return summary
    
    def _calculate_timeline_summary(self, timeline: List[Dict]) -> Dict:
        """
        Calculate summary statistics for a timeline
        """
        if not timeline:
            return {'message': 'No timeline data available'}
        
        # Calculate total evolution
        total_changes = []
        for point in timeline:
            if 'changes_from_previous' in point:
                total_change = point['changes_from_previous'].get('total_change', 0)
                total_changes.append(total_change)
        
        summary = {
            'timeline_span': self._calculate_time_span(timeline[0]['timestamp'], timeline[-1]['timestamp']),
            'total_snapshots': len(timeline),
            'average_change_rate': statistics.mean(total_changes) if total_changes else 0,
            'most_active_period': self._identify_most_active_period(timeline),
            'stability_periods': len(self._identify_stability_periods(timeline))
        }
        
        return summary
    
    def _identify_most_changed_traits(self, timeline: List[Dict]) -> List[Dict]:
        """
        Identify traits that changed the most throughout the timeline
        """
        if len(timeline) < 2:
            return []
        
        # Calculate total change for each trait
        trait_changes = {}
        initial_state = timeline[0]['personality_state']
        final_state = timeline[-1]['personality_state']
        
        for trait in initial_state:
            if trait in final_state:
                total_change = abs(final_state[trait] - initial_state[trait])
                trait_changes[trait] = {
                    'trait': trait,
                    'total_change': total_change,
                    'initial_value': initial_state[trait],
                    'final_value': final_state[trait],
                    'direction': 'increase' if final_state[trait] > initial_state[trait] else 'decrease'
                }
        
        # Sort by total change
        most_changed = sorted(trait_changes.values(), key=lambda x: x['total_change'], reverse=True)
        
        return most_changed[:3]  # Top 3 most changed traits
    
    def _identify_stability_periods(self, timeline: List[Dict]) -> List[Dict]:
        """
        Identify periods of personality stability in the timeline
        """
        if len(timeline) < 3:
            return []
        
        stability_periods = []
        current_period_start = None
        stable_threshold = 0.05  # Low change threshold for stability
        
        for i in range(1, len(timeline)):
            point = timeline[i]
            if 'changes_from_previous' in point:
                total_change = point['changes_from_previous'].get('total_change', 0)
                
                if total_change < stable_threshold:
                    # Start of stable period
                    if current_period_start is None:
                        current_period_start = timeline[i-1]['timestamp']
                else:
                    # End of stable period
                    if current_period_start is not None:
                        stability_periods.append({
                            'start': current_period_start,
                            'end': timeline[i-1]['timestamp'],
                            'duration': self._calculate_time_span(current_period_start, timeline[i-1]['timestamp'])
                        })
                        current_period_start = None
        
        # Check if timeline ends in a stable period
        if current_period_start is not None:
            stability_periods.append({
                'start': current_period_start,
                'end': timeline[-1]['timestamp'],
                'duration': self._calculate_time_span(current_period_start, timeline[-1]['timestamp'])
            })
        
        return stability_periods
    
    def _get_recent_snapshots(self, session_id: str, days_back: int = 14) -> List[Dict]:
        """
        Get recent snapshots for a session
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        session_snapshots = self.session_snapshots.get(session_id, [])
        
        return [
            snapshot for snapshot in session_snapshots
            if datetime.fromisoformat(snapshot['timestamp']) > cutoff_date
        ]
    
    def _calculate_trait_velocities(self, snapshots: List[Dict]) -> Dict[str, float]:
        """
        Calculate velocity of change for each trait
        """
        if len(snapshots) < 2:
            return {}
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x['timestamp'])
        
        velocities = {}
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        time_diff = datetime.fromisoformat(last_snapshot['timestamp']) - datetime.fromisoformat(first_snapshot['timestamp'])
        days_diff = time_diff.total_seconds() / (24 * 3600)
        
        if days_diff > 0:
            for trait in first_snapshot['personality_state']:
                if trait in last_snapshot['personality_state']:
                    value_change = last_snapshot['personality_state'][trait] - first_snapshot['personality_state'][trait]
                    velocities[trait] = value_change / days_diff
        
        return velocities
    
    def _calculate_prediction_confidence(self, trait: str, snapshots: List[Dict]) -> float:
        """
        Calculate confidence for trait prediction
        """
        # Base confidence on data points and trend consistency
        data_points = len(snapshots)
        base_confidence = min(data_points / 10.0, 0.8)  # More data = higher confidence
        
        # Analyze trend consistency
        values = [snapshot['personality_state'].get(trait, 0.5) for snapshot in snapshots]
        trend_consistency = self._calculate_trend_consistency(values)
        
        # Combine factors
        confidence = base_confidence * trend_consistency
        
        return min(max(confidence, 0.1), 0.9)  # Keep between 0.1 and 0.9
    
    def _identify_notable_predictions(self, predictions: Dict) -> List[Dict]:
        """
        Identify notable predicted changes
        """
        notable = []
        
        for trait, prediction_data in predictions.items():
            predicted_change = abs(prediction_data['predicted_change'])
            confidence = prediction_data['confidence']
            
            # Notable if significant change with reasonable confidence
            if predicted_change > 0.2 and confidence > 0.5:
                notable.append({
                    'trait': trait,
                    'predicted_change': prediction_data['predicted_change'],
                    'confidence': confidence,
                    'significance': 'High' if predicted_change > 0.3 else 'Moderate'
                })
        
        # Sort by significance
        notable.sort(key=lambda x: abs(x['predicted_change']), reverse=True)
        
        return notable
    
    def _calculate_overall_prediction_confidence(self, predictions: Dict) -> str:
        """
        Calculate overall prediction confidence level
        """
        if not predictions:
            return 'Very Low'
        
        confidences = [data['confidence'] for data in predictions.values()]
        avg_confidence = statistics.mean(confidences)
        
        if avg_confidence > 0.8:
            return 'Very High'
        elif avg_confidence > 0.6:
            return 'High'
        elif avg_confidence > 0.4:
            return 'Moderate'
        elif avg_confidence > 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def _generate_velocity_insights(self, snapshots: List[Dict]) -> List[str]:
        """
        Generate insights about evolution velocity
        """
        insights = []
        
        if len(snapshots) < 2:
            return insights
        
        # Calculate overall velocity
        velocities = []
        for snapshot in snapshots[1:]:
            evolution_metrics = snapshot.get('evolution_metrics', {})
            velocity = evolution_metrics.get('evolution_velocity', 0)
            velocities.append(velocity)
        
        if velocities:
            avg_velocity = statistics.mean(velocities)
            
            if avg_velocity > 0.1:
                insights.append("Personality shows high evolution velocity with frequent changes")
            elif avg_velocity > 0.05:
                insights.append("Moderate evolution velocity indicates steady personality development")
            else:
                insights.append("Low evolution velocity suggests stable personality characteristics")
        
        return insights
    
    def _generate_stability_insights(self, snapshots: List[Dict]) -> List[str]:
        """
        Generate insights about personality stability
        """
        insights = []
        
        stability_scores = []
        for snapshot in snapshots:
            evolution_metrics = snapshot.get('evolution_metrics', {})
            stability = evolution_metrics.get('stability_score', 1.0)
            stability_scores.append(stability)
        
        if stability_scores:
            avg_stability = statistics.mean(stability_scores)
            
            if avg_stability > 0.8:
                insights.append("Personality demonstrates high stability with consistent traits")
            elif avg_stability > 0.6:
                insights.append("Moderate personality stability with occasional adjustments")
            else:
                insights.append("High personality variability indicates active adaptation")
        
        return insights
    
    def _generate_pattern_insights(self, snapshots: List[Dict]) -> List[str]:
        """
        Generate insights about evolution patterns
        """
        insights = []
        
        if len(snapshots) >= 3:
            # Analyze if there are cyclical patterns
            # This is a simplified pattern detection
            changes = []
            for snapshot in snapshots[1:]:
                evolution_metrics = snapshot.get('evolution_metrics', {})
                total_change = evolution_metrics.get('total_change_since_last', 0)
                changes.append(total_change)
            
            # Check for consistent increase/decrease
            increasing_periods = sum(1 for i in range(1, len(changes)) if changes[i] > changes[i-1])
            decreasing_periods = sum(1 for i in range(1, len(changes)) if changes[i] < changes[i-1])
            
            if increasing_periods > len(changes) * 0.7:
                insights.append("Personality shows increasing evolution activity over time")
            elif decreasing_periods > len(changes) * 0.7:
                insights.append("Personality evolution is slowing down and stabilizing")
        
        return insights
    
    def _generate_trend_insights(self, snapshots: List[Dict]) -> List[str]:
        """
        Generate insights about evolution trends
        """
        insights = []
        
        if len(snapshots) >= 2:
            # Compare first and last snapshots
            first_personality = snapshots[0]['personality_state']
            last_personality = snapshots[-1]['personality_state']
            
            # Find traits with significant changes
            significant_changes = []
            for trait in first_personality:
                if trait in last_personality:
                    change = last_personality[trait] - first_personality[trait]
                    if abs(change) > 0.2:
                        significant_changes.append((trait, change))
            
            if significant_changes:
                # Sort by magnitude
                significant_changes.sort(key=lambda x: abs(x[1]), reverse=True)
                top_change = significant_changes[0]
                
                direction = "increased" if top_change[1] > 0 else "decreased"
                insights.append(f"Most significant trend: {top_change[0]} has {direction} substantially")
        
        return insights
    
    # Utility methods
    def _calculate_evolution_direction(self, snapshots: List[Dict]) -> str:
        """Calculate overall evolution direction"""
        # Implementation would analyze the general direction of personality changes
        return "Stable"  # Simplified
    
    def _analyze_convergence(self, snapshots: List[Dict]) -> str:
        """Analyze convergence/divergence patterns"""
        return "Converging"  # Simplified
    
    def _identify_evolution_phase(self, snapshots: List[Dict]) -> str:
        """Identify current evolution phase"""
        return "Development"  # Simplified
    
    def _calculate_personality_diversity(self, personalities: List[Dict[str, float]]) -> float:
        """Calculate diversity score across personalities"""
        return 0.5  # Simplified
    
    def _identify_common_patterns(self, evolution_directions: List[str]) -> List[str]:
        """Identify common evolution patterns"""
        return ["Stable development"]  # Simplified
    
    def _extract_trait_evolution_across_sessions(self, session_data: Dict, trait: str) -> List[float]:
        """Extract trait evolution across all sessions"""
        return [0.5]  # Simplified
    
    def _analyze_trait_pattern(self, trait_evolution: List[float]) -> str:
        """Analyze pattern for a specific trait"""
        return "Stable"  # Simplified
    
    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return "Stable"
        return "Increasing" if values[-1] > values[0] else "Decreasing" if values[-1] < values[0] else "Stable"
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate how consistent a trend is"""
        if len(values) < 3:
            return 1.0
        
        # Simple consistency measure
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        same_direction = sum(1 for i in range(1, len(changes)) 
                           if (changes[i] > 0) == (changes[i-1] > 0))
        return same_direction / max(len(changes) - 1, 1)
    
    def _calculate_overall_evolution_trend(self, session_data: Dict) -> str:
        """Calculate overall evolution trend across all sessions"""
        return "Progressive development"  # Simplified
    
    def _calculate_time_span(self, start_time: str, end_time: str) -> str:
        """Calculate human-readable time span"""
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        diff = end - start
        
        if diff.days > 0:
            return f"{diff.days} days"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours"
        else:
            minutes = diff.seconds // 60
            return f"{minutes} minutes"
    
    def _calculate_session_stability_trend(self, snapshots: List[Dict]) -> str:
        """Calculate stability trend for a session"""
        if len(snapshots) < 2:
            return "Insufficient data"
        
        stability_scores = [
            snapshot.get('evolution_metrics', {}).get('stability_score', 1.0)
            for snapshot in snapshots
        ]
        
        return self._calculate_simple_trend(stability_scores)
    
    def _calculate_personality_similarity(self, personality1: Dict[str, float], 
                                        personality2: Dict[str, float]) -> float:
        """Calculate similarity between two personalities"""
        common_traits = set(personality1.keys()) & set(personality2.keys())
        
        if not common_traits:
            return 0.0
        
        total_difference = sum(
            abs(personality1[trait] - personality2[trait])
            for trait in common_traits
        )
        
        # Convert to similarity (0-1 scale)
        avg_difference = total_difference / len(common_traits)
        similarity = max(0, 1 - avg_difference)
        
        return similarity
    
    def _identify_personality_archetypes(self, session_personalities: Dict) -> List[str]:
        """Identify common personality archetypes"""
        return ["Balanced", "Analytical"]  # Simplified
    
    def _calculate_evolution_trajectory(self, snapshots: List[Dict]) -> str:
        """Calculate evolution trajectory"""
        return "Steady progression"  # Simplified
    
    def _identify_turning_points(self, snapshots: List[Dict]) -> List[Dict]:
        """Identify key turning points in evolution"""
        return []  # Simplified
    
    def _identify_personality_clusters(self, session_personalities: Dict) -> List[List[str]]:
        """Identify clusters of similar personalities"""
        return []  # Simplified
    
    def _identify_most_active_period(self, timeline: List[Dict]) -> str:
        """Identify the most active evolution period"""
        return "Recent period"  # Simplified