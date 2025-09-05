/**
 * Event Bus Blocks - Prevents secondary center creation from events
 */

import { SINGLE_CENTER_CONFIG, LearningEvent } from './singleCenterConfig.js';
import { mergeLearningEvent } from './centerManager.js';

// Mock store for demonstration
let centerStore = {
    primary: {
        id: 'presence-ai',
        name: 'YOU / PRESENCE',
        color: '#4facfe',
        radius: 100,
        aura: { color: '#4facfe', intensity: 0.6 },
        rings: [],
        personaLayers: []
    }
};

/**
 * Event Bus - handles persona events with single center enforcement
 */
class SingleCenterEventBus {
    constructor() {
        this.listeners = {};
        this.setupEventBlocks();
    }
    
    setupEventBlocks() {
        // Block persona.center events that would create secondary centers
        this.on('persona.center', (event) => {
            if (SINGLE_CENTER_CONFIG.centers.mode === 'single') {
                const primaryId = SINGLE_CENTER_CONFIG.centers.primary_id;
                const primary = this.getCenter(primaryId);
                
                if (primary) {
                    // Convert persona event to learning event and merge
                    const learningEvent = new LearningEvent('persona.merge', {
                        persona: event.persona || {
                            name: event.name || 'Book Learning',
                            weight: event.weight || 0.5,
                            toneColor: event.color || '#4facfe'
                        },
                        bookId: event.bookId
                    });
                    
                    const merged = mergeLearningEvent(primary, learningEvent);
                    this.updateCenter(merged);
                    
                    console.log(`✅ [EVENT] Merged persona "${event.name}" into primary center`);
                } else {
                    console.warn('[EVENT] Primary center not found for persona merge');
                }
                
                return; // PREVENT secondary center creation
            }
        });
        
        // Block bookCenterActivated events
        this.on('bookCenterActivated', (event) => {
            if (SINGLE_CENTER_CONFIG.centers.mode === 'single') {
                console.log(`🔒 [EVENT] Blocked bookCenterActivated for "${event.name}"`);
                
                // Redirect to primary center enhancement
                this.emit('persona.center', {
                    name: event.name,
                    weight: 0.7,
                    color: event.color || '#4facfe',
                    bookId: event.bookId
                });
                
                return;
            }
        });
        
        // Block Two Centers Experience events
        this.on('twoCentersExperience', (event) => {
            if (!SINGLE_CENTER_CONFIG.features.twoCentersExperience) {
                console.warn(`🚫 [EVENT] Two Centers Experience is disabled:`, event);
                return;
            }
        });
    }
    
    on(eventType, handler) {
        if (!this.listeners[eventType]) {
            this.listeners[eventType] = [];
        }
        this.listeners[eventType].push(handler);
    }
    
    emit(eventType, data) {
        if (this.listeners[eventType]) {
            this.listeners[eventType].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`[EVENT] Error in ${eventType} handler:`, error);
                }
            });
        }
    }
    
    getCenter(id) {
        return id === 'presence-ai' ? centerStore.primary : null;
    }
    
    updateCenter(centerData) {
        if (centerData.id === 'presence-ai') {
            centerStore.primary = centerData;
            
            // Trigger visual update
            this.emit('centerUpdated', centerData);
        }
    }
}

// Global event bus instance
window.singleCenterEventBus = new SingleCenterEventBus();

// Hijack common center creation functions
window.originalCreateNode = window.createNode;
window.createNode = function(nodeData) {
    if (SINGLE_CENTER_CONFIG.centers.mode === 'single' && 
        nodeData.id !== SINGLE_CENTER_CONFIG.centers.primary_id) {
        console.warn('[HIJACK] Blocked createNode for secondary center:', nodeData);
        return null;
    }
    return window.originalCreateNode ? window.originalCreateNode(nodeData) : null;
};

window.originalAddCenter = window.addCenter;
window.addCenter = function(centerData) {
    if (SINGLE_CENTER_CONFIG.centers.mode === 'single' && 
        centerData.id !== SINGLE_CENTER_CONFIG.centers.primary_id) {
        console.warn('[HIJACK] Blocked addCenter for secondary center:', centerData);
        return null;
    }
    return window.originalAddCenter ? window.originalAddCenter(centerData) : null;
};

console.log('🚫 Event Bus Blocks active - Secondary center events blocked');