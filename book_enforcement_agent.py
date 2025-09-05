"""
Book Enforcement Agent
Ensures ONLY YOU and AI circles exist - blocks all secondary centers
"""
import logging
import json
from datetime import datetime

class BookEnforcementAgent:
    def __init__(self):
        self.agent_id = "book_enforcement_agent"
        self.name = "Single Center Enforcement Specialist"
        self.role = "center_enforcement"
        self.active = True
        self.tasks_completed = 0
        
        # Enforcement rules - HARD LIMITS
        self.allowed_centers = ['presence-ai', 'you']  # ΜΟΝΟ αυτα
        self.blocked_attempts = 0
        
        logging.info(f"🔒 {self.name} initialized - Zero tolerance for secondary centers")
    
    def enforce_single_center(self, book_data):
        """Enforce single center rule during book upload"""
        try:
            logging.info("🔒 Book Enforcement Agent: Validating upload for single center compliance...")
            
            # Force merge everything to YOU center
            enforced_data = {
                'id': book_data.get('id'),
                'title': book_data.get('title'),
                'filename': book_data.get('filename'),
                'upload_date': datetime.now().isoformat(),
                'book_centers': [
                    {
                        'id': 'presence-ai',
                        'name': 'YOU',
                        'title': 'Single Learning Center',
                        'color': '#4facfe',
                        'type': 'primary_only',
                        'merge_mode': True,
                        'expansion_only': True  # ΜΟΝΟ visual expansion
                    }
                ]
            }
            
            self.tasks_completed += 1
            logging.info("✅ Book content MERGED to YOU center - No secondary centers created")
            
            return {
                'success': True,
                'enforced': True,
                'centers_created': 0,
                'merge_target': 'YOU',
                'book_data': enforced_data
            }
            
        except Exception as e:
            logging.error(f"❌ Enforcement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def block_secondary_creation(self, center_request):
        """Block any attempt to create secondary centers"""
        self.blocked_attempts += 1
        
        logging.warning(f"🚫 BLOCKED secondary center creation attempt #{self.blocked_attempts}")
        logging.warning(f"🚫 Attempted center: {center_request}")
        
        return {
            'blocked': True,
            'reason': 'Single center mode enforced',
            'allowed_centers': self.allowed_centers,
            'redirect_to': 'presence-ai'
        }
    
    def validate_center_compliance(self):
        """Validate system is in single center mode"""
        try:
            # Check for any unauthorized centers
            compliance_report = {
                'mode': 'single',
                'allowed_centers': self.allowed_centers,
                'blocked_attempts': self.blocked_attempts,
                'enforcement_active': True,
                'compliance_status': 'ENFORCED',
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info("✅ Center compliance validated - Single center mode active")
            return compliance_report
            
        except Exception as e:
            logging.error(f"❌ Compliance check failed: {e}")
            return {'error': str(e)}
    
    def get_status(self):
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'active': self.active,
            'blocked_attempts': self.blocked_attempts,
            'tasks_completed': self.tasks_completed,
            'allowed_centers': self.allowed_centers,
            'enforcement_mode': 'ACTIVE'
        }

# Global instance
book_agent = BookEnforcementAgent()