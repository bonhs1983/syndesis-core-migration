"""
Multi-Agent Management System - Full Control Panel
Allows multiple agents to collaborate in the same workspace
"""
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify

class AgentManager:
    def __init__(self):
        self.active_agents = {}
        self.agent_configs = {}
        self.collaboration_log = []
    
    def register_agent(self, agent_id, config):
        """Register a new agent in the system"""
        self.active_agents[agent_id] = {
            'id': agent_id,
            'name': config.get('name', f'Agent_{agent_id}'),
            'role': config.get('role', 'assistant'),
            'permissions': config.get('permissions', ['read', 'write']),
            'active': True,
            'last_activity': datetime.now().isoformat(),
            'tasks_completed': 0
        }
        self.agent_configs[agent_id] = config
        
        self.log_activity(f"Agent {agent_id} registered with role: {config.get('role')}")
        return True
    
    def deactivate_agent(self, agent_id):
        """Deactivate an agent"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id]['active'] = False
            self.log_activity(f"Agent {agent_id} deactivated")
            return True
        return False
    
    def get_active_agents(self):
        """Get all active agents"""
        return {aid: info for aid, info in self.active_agents.items() 
                if info.get('active', False)}
    
    def delegate_task(self, from_agent, to_agent, task_description):
        """Delegate a task between agents"""
        if to_agent not in self.active_agents:
            return False, f"Agent {to_agent} not found"
        
        if not self.active_agents[to_agent]['active']:
            return False, f"Agent {to_agent} is not active"
        
        delegation = {
            'from': from_agent,
            'to': to_agent,
            'task': task_description,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        self.collaboration_log.append(delegation)
        self.log_activity(f"Task delegated from {from_agent} to {to_agent}: {task_description}")
        return True, "Task delegated successfully"
    
    def log_activity(self, message):
        """Log system activity"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        self.collaboration_log.append(log_entry)
        print(f"[AGENT_MANAGER] {message}")
    
    def get_system_status(self):
        """Get complete system status"""
        return {
            'active_agents': len(self.get_active_agents()),
            'total_agents': len(self.active_agents),
            'recent_activity': self.collaboration_log[-10:],  # Last 10 activities
            'system_health': 'operational'
        }

# Global agent manager instance
agent_manager = AgentManager()

def setup_agent_routes(app):
    """Add agent management routes to Flask app"""
    
    @app.route('/agents/register', methods=['POST'])
    def register_new_agent():
        """Register a new agent"""
        data = request.json
        agent_id = data.get('agent_id')
        config = data.get('config', {})
        
        if not agent_id:
            return jsonify({'success': False, 'error': 'agent_id required'}), 400
        
        success = agent_manager.register_agent(agent_id, config)
        return jsonify({'success': success, 'agent_id': agent_id})
    
    @app.route('/agents/active', methods=['GET'])
    def get_active_agents():
        """Get all active agents"""
        agents = agent_manager.get_active_agents()
        return jsonify({'active_agents': agents})
    
    @app.route('/agents/status', methods=['GET'])
    def get_system_status():
        """Get system status"""
        status = agent_manager.get_system_status()
        return jsonify(status)
    
    @app.route('/agents/delegate', methods=['POST'])
    def delegate_task():
        """Delegate task between agents"""
        data = request.json
        from_agent = data.get('from_agent')
        to_agent = data.get('to_agent')
        task = data.get('task')
        
        success, message = agent_manager.delegate_task(from_agent, to_agent, task)
        return jsonify({'success': success, 'message': message})
    
    @app.route('/agents/deactivate', methods=['POST'])
    def deactivate_agent():
        """Deactivate an agent"""
        data = request.json
        agent_id = data.get('agent_id')
        
        success = agent_manager.deactivate_agent(agent_id)
        message = f"Agent {agent_id} deactivated" if success else f"Agent {agent_id} not found"
        return jsonify({'success': success, 'message': message})

# Auto-register the current agent
def auto_register_current_agent():
    """Auto-register the current working agent"""
    current_config = {
        'name': 'Syndesis_Primary_Agent',
        'role': 'full_stack_developer',
        'permissions': ['read', 'write', 'admin'],
        'specialties': ['flask', 'hrv_analysis', 'consciousness_mapping', 'biometric_integration']
    }
    agent_manager.register_agent('syndesis_primary', current_config)

if __name__ == "__main__":
    # Register current agent automatically
    auto_register_current_agent()
    print("🤖 Multi-Agent Management System initialized")
    print("✅ Primary agent registered")
    print("🔗 Ready for collaboration")