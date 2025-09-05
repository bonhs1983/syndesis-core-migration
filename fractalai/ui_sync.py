"""
Real-time UI synchronization system for personality changes
"""
import json
import logging
from typing import Dict, List, Any, Callable
from datetime import datetime

class UIPersonalitySync:
    """Handles real-time synchronization of personality changes with UI"""
    
    def __init__(self):
        self.active_sessions = {}  # Track active UI sessions
        self.change_callbacks = []  # Registered callbacks for UI updates
        self.sync_history = {}     # History of sync events
        
    def register_session(self, session_id: str, ui_context: Dict[str, Any] = None):
        """Register a UI session for personality sync"""
        self.active_sessions[session_id] = {
            'registered_at': datetime.now().isoformat(),
            'ui_context': ui_context or {},
            'last_sync': None,
            'sync_count': 0
        }
        logging.info(f"UI session registered for sync: {session_id}")
    
    def sync_personality_change(self, session_id: str, old_personality: Dict[str, float], 
                               new_personality: Dict[str, float], change_reason: str):
        """Sync personality changes to UI in real-time"""
        
        # Calculate significant changes
        significant_changes = {}
        for trait, new_value in new_personality.items():
            old_value = old_personality.get(trait, 0.5)
            if abs(new_value - old_value) > 0.15:  # Significant change threshold
                significant_changes[trait] = {
                    'from': old_value,
                    'to': new_value,
                    'change': new_value - old_value
                }
        
        # Create sync event
        sync_event = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'change_reason': change_reason,
            'personality_state': new_personality,
            'significant_changes': significant_changes,
            'sync_type': 'personality_update'
        }
        
        # Store in sync history
        if session_id not in self.sync_history:
            self.sync_history[session_id] = []
        self.sync_history[session_id].append(sync_event)
        
        # Update session tracking
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_sync'] = sync_event['timestamp']
            self.active_sessions[session_id]['sync_count'] += 1
        
        # Trigger all registered callbacks
        for callback in self.change_callbacks:
            try:
                callback(sync_event)
            except Exception as e:
                logging.error(f"Error in UI sync callback: {e}")
        
        # Log for transparency
        if significant_changes:
            changes_desc = ", ".join([f"{trait}: {change['from']:.1f}→{change['to']:.1f}" 
                                    for trait, change in significant_changes.items()])
            logging.info(f"UI SYNC: {session_id} - {changes_desc} ({change_reason})")
        
        return sync_event
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for personality change events"""
        self.change_callbacks.append(callback)
        logging.info(f"Registered UI sync callback: {callback.__name__}")
    
    def get_session_sync_status(self, session_id: str) -> Dict[str, Any]:
        """Get synchronization status for a session"""
        if session_id not in self.active_sessions:
            return {'status': 'not_registered', 'sync_count': 0}
        
        session = self.active_sessions[session_id]
        history = self.sync_history.get(session_id, [])
        
        return {
            'status': 'active',
            'registered_at': session['registered_at'],
            'last_sync': session['last_sync'],
            'sync_count': session['sync_count'],
            'recent_changes': history[-5:] if history else []  # Last 5 changes
        }
    
    def generate_ui_update_message(self, sync_event: Dict[str, Any]) -> str:
        """Generate human-readable message for UI display"""
        changes = sync_event.get('significant_changes', {})
        reason = sync_event.get('change_reason', 'User interaction')
        
        if not changes:
            return f"Personality stable - {reason}"
        
        change_descriptions = []
        for trait, change_info in changes.items():
            direction = "↗️" if change_info['change'] > 0 else "↘️"
            change_descriptions.append(f"{trait} {direction} {change_info['to']:.1f}")
        
        return f"🔄 Personality adapted: {', '.join(change_descriptions)} - {reason}"

# Global UI sync instance
ui_sync = UIPersonalitySync()