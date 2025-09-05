/**
 * Center Manager - SINGLE CENTER ENFORCEMENT
 * All secondary center creation is blocked here
 */

import { SINGLE_CENTER_CONFIG, PersonaLayer, LearningEvent } from './singleCenterConfig.js';

/**
 * GUARD FUNCTION: Reconcile centers - κόβει οτιδήποτε δεν είναι το primary
 */
export function reconcileCenters(incoming, cfg = SINGLE_CENTER_CONFIG) {
    const pid = cfg.centers.primary_id || 'presence-ai';
    
    if (cfg.centers.mode === 'single') {
        // Find primary center or fallback to first
        const primary = incoming.find(c => c.id === pid) || incoming[0];
        
        if (!primary) {
            console.warn('[SINGLE] No primary center found, creating default');
            return [{
                id: pid,
                role: 'primary',
                name: 'YOU / PRESENCE',
                color: '#4facfe',
                radius: 100,
                aura: { color: '#4facfe', intensity: 0.6 },
                rings: [],
                personaLayers: []
            }];
        }
        
        // Ensure only primary with correct role
        const result = [{ ...primary, role: 'primary', id: pid }];
        
        // Log blocked centers
        const blocked = incoming.filter(c => c.id !== pid);
        if (blocked.length > 0 && cfg.enforcement.logBlocked) {
            console.warn('[SINGLE] Blocked secondary centers:', blocked.map(c => c.id));
        }
        
        return result;
    }
    
    return incoming;
}

/**
 * Merge Learning Event - ενσωματώνει persona data στο primary center
 */
export function mergeLearningEvent(center, event) {
    const alpha = 0.25; // EMA smoothing factor
    const next = { ...center };
    
    if (event.type === 'persona.merge') {
        const persona = event.persona;
        const weight = Math.max(0, Math.min(1, persona.weight || 0.5));
        
        // Add persona layer
        next.personaLayers = upsertPersonaLayer(next.personaLayers || [], {
            source: 'book',
            label: persona.name || 'Book Learning',
            color: persona.toneColor || '#4facfe',
            weight: weight,
            timestamp: Date.now()
        });
        
        // Update aura
        next.aura = {
            color: persona.toneColor || next.aura.color,
            intensity: Math.max(next.aura.intensity || 0.6, weight)
        };
        
        // Add expansion rings
        next.rings = [...(next.rings || []), {
            timestamp: Date.now(),
            delta: 3 * weight,
            color: persona.toneColor || '#4facfe'
        }];
        
        // Expand radius (capped at 140)
        next.radius = Math.min((next.radius || 100) + 2 * weight, 140);
        
        console.log(`✅ [SINGLE] Merged persona "${persona.name}" into primary center`);
    }
    
    if (event.type === 'metric.update' && event.metrics?.coherence != null) {
        const coherence = event.metrics.coherence;
        next.aura = {
            ...next.aura,
            intensity: alpha * coherence + (1 - alpha) * (next.aura?.intensity || 0.6)
        };
    }
    
    return next;
}

/**
 * Upsert Persona Layer - προσθήκη ή ενημέρωση persona layer
 */
function upsertPersonaLayer(layers, newLayer) {
    const existing = layers.find(l => l.label === newLayer.label && l.source === newLayer.source);
    
    if (existing) {
        // Update existing layer
        return layers.map(l => 
            l === existing ? new PersonaLayer(newLayer) : l
        );
    } else {
        // Add new layer (keep only last 5)
        const updated = [...layers, new PersonaLayer(newLayer)];
        return updated.slice(-5); // Keep last 5 layers
    }
}

/**
 * BLOCK FUNCTION: Prevents secondary center creation
 */
export function blockSecondaryCenter(nodeData, cfg = SINGLE_CENTER_CONFIG) {
    if (cfg.centers.mode === 'single' && nodeData.id !== cfg.centers.primary_id) {
        if (cfg.enforcement.warnOnViolation) {
            console.warn('[SINGLE] Blocked secondary center creation:', nodeData);
        }
        return null; // Block creation
    }
    return nodeData;
}

/**
 * CLEANUP FUNCTION: Removes any existing secondary centers from DOM
 */
export function cleanupSecondaryNodes() {
    const secondarySelectors = [
        '.center-node[data-role="secondary"]',
        '.center-node[data-label="AWARENESS"]',
        '.center-space:not(.center-presence)', // All except presence
        '.center-awareness',
        '.center-compassion',
        '.center-quality',
        '.center-forgiveness'
    ];
    
    secondarySelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            console.log(`🗑️ [SINGLE] Removing secondary node:`, el);
            el.remove();
        });
    });
}

console.log('🛡️ Center Manager loaded - Single center enforcement active');