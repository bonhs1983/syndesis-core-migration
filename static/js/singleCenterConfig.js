/**
 * Single Center Configuration - HARD ENFORCEMENT
 * Prevents any secondary center creation
 */

// KILL SWITCH - Runtime configuration with no cache
export const SINGLE_CENTER_CONFIG = {
    centers: {
        mode: 'single',
        primary_id: 'presence-ai',
        secondary_enabled: false,
        max: 1,
    },
    features: {
        twoCentersExperience: false, // ΚΛΕΙΣΤΟ
        allowSecondaryNodes: false,
        enforceHardLimit: true,
    },
    // Runtime enforcement flags
    enforcement: {
        blockSecondaryCreation: true,
        removeExistingSecondary: true,
        logBlocked: true,
        warnOnViolation: true,
    }
};

/**
 * PersonaLayer - για visual feedback στο primary center
 */
export class PersonaLayer {
    constructor(data) {
        this.source = data.source || 'book';
        this.label = data.label || 'Unknown';
        this.color = data.color || '#4facfe';
        this.weight = Math.max(0, Math.min(1, data.weight || 0.5));
        this.timestamp = data.timestamp || Date.now();
    }

    // Visual impact calculation
    getVisualImpact() {
        const age = Date.now() - this.timestamp;
        const ageDecay = Math.exp(-age / (1000 * 60 * 30)); // 30 minute decay
        return this.weight * ageDecay;
    }
}

/**
 * LearningEvent - για merge events στο primary center
 */
export class LearningEvent {
    constructor(type, data) {
        this.type = type;
        this.timestamp = Date.now();
        this.data = data || {};
        
        if (type === 'persona.merge') {
            this.persona = data.persona;
            this.bookId = data.bookId;
        }
        
        if (type === 'metric.update') {
            this.metrics = data.metrics;
        }
    }
}

console.log('🔒 Single Center Config loaded - Secondary centers BLOCKED');