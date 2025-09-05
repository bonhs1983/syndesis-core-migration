/**
 * Single Center Renderer - Visual effects for the primary center only
 */

import { SINGLE_CENTER_CONFIG } from './singleCenterConfig.js';

/**
 * Render Single Center με visual enhancements
 */
export function renderSingleCenter(centerData) {
    const container = document.querySelector('.cosmos-container');
    if (!container) {
        console.error('[SINGLE] No cosmos container found');
        return;
    }
    
    // Remove any existing centers first
    container.querySelectorAll('.center-space').forEach(el => el.remove());
    
    // Create primary center element
    const centerEl = document.createElement('div');
    centerEl.className = 'center-space center-presence single-mode';
    centerEl.dataset.id = centerData.id;
    centerEl.dataset.role = 'primary';
    centerEl.innerHTML = '<span>YOU<br/>PRESENCE</span>';
    
    // Apply visual enhancements
    applyVisualEnhancements(centerEl, centerData);
    
    // Position (center of screen)
    centerEl.style.position = 'absolute';
    centerEl.style.left = '50%';
    centerEl.style.top = '50%';
    centerEl.style.transform = 'translate(-50%, -50%)';
    centerEl.style.zIndex = '100';
    
    container.appendChild(centerEl);
    
    // Add persona badges if any
    if (centerData.personaLayers && centerData.personaLayers.length > 0) {
        addPersonaBadges(centerEl, centerData.personaLayers);
    }
    
    console.log('🎨 [SINGLE] Primary center rendered with enhancements');
    return centerEl;
}

/**
 * Apply Visual Enhancements - aura, rings, radius
 */
function applyVisualEnhancements(element, centerData) {
    const radius = centerData.radius || 100;
    const aura = centerData.aura || { color: '#4facfe', intensity: 0.6 };
    const rings = centerData.rings || [];
    
    // Set base size
    element.style.width = `${radius}px`;
    element.style.height = `${radius}px`;
    
    // Aura effect
    const auraIntensity = Math.min(1, aura.intensity);
    const auraColor = aura.color || '#4facfe';
    element.style.boxShadow = `
        0 0 ${20 + auraIntensity * 30}px ${auraColor}${Math.floor(auraIntensity * 255).toString(16).padStart(2, '0')},
        inset 0 0 ${10 + auraIntensity * 20}px ${auraColor}${Math.floor(auraIntensity * 128).toString(16).padStart(2, '0')}
    `;
    
    // Animated pulse based on intensity
    if (auraIntensity > 0.7) {
        element.style.animation = 'singleCenterPulse 2s ease-in-out infinite';
    }
    
    // Expansion rings (temporary visual effects)
    addExpansionRings(element, rings);
}

/**
 * Add Expansion Rings - temporary visual feedback
 */
function addExpansionRings(centerElement, rings) {
    const activeRings = rings.filter(ring => {
        const age = Date.now() - ring.timestamp;
        return age < 3000; // 3 second duration
    });
    
    activeRings.forEach((ring, index) => {
        const age = Date.now() - ring.timestamp;
        const progress = age / 3000; // 0 to 1
        
        const ringEl = document.createElement('div');
        ringEl.className = 'expansion-ring';
        ringEl.style.position = 'absolute';
        ringEl.style.left = '50%';
        ringEl.style.top = '50%';
        ringEl.style.transform = 'translate(-50%, -50%)';
        ringEl.style.border = `2px solid ${ring.color}`;
        ringEl.style.borderRadius = '50%';
        ringEl.style.pointerEvents = 'none';
        ringEl.style.zIndex = '99';
        
        // Animate ring expansion
        const size = 100 + (ring.delta * 20 * progress);
        const opacity = 1 - progress;
        
        ringEl.style.width = `${size}px`;
        ringEl.style.height = `${size}px`;
        ringEl.style.opacity = opacity;
        
        centerElement.appendChild(ringEl);
        
        // Remove after animation
        setTimeout(() => {
            if (ringEl.parentNode) {
                ringEl.remove();
            }
        }, 3000 - age);
    });
}

/**
 * Add Persona Badges - visual indicators of book learning
 */
function addPersonaBadges(centerElement, personaLayers) {
    // Remove existing badges
    centerElement.querySelectorAll('.persona-badge').forEach(el => el.remove());
    
    const activeLayers = personaLayers
        .filter(layer => layer.getVisualImpact() > 0.1)
        .slice(-3); // Show last 3 active layers
    
    activeLayers.forEach((layer, index) => {
        const badge = document.createElement('div');
        badge.className = 'persona-badge';
        badge.textContent = layer.label.charAt(0); // First letter
        badge.title = `${layer.label} (${Math.round(layer.weight * 100)}%)`;
        
        // Position badges around the center
        const angle = (index / activeLayers.length) * 2 * Math.PI;
        const distance = 60; // Distance from center
        const x = Math.cos(angle) * distance;
        const y = Math.sin(angle) * distance;
        
        badge.style.position = 'absolute';
        badge.style.left = `calc(50% + ${x}px)`;
        badge.style.top = `calc(50% + ${y}px)`;
        badge.style.transform = 'translate(-50%, -50%)';
        badge.style.width = '24px';
        badge.style.height = '24px';
        badge.style.borderRadius = '50%';
        badge.style.backgroundColor = layer.color;
        badge.style.color = 'white';
        badge.style.fontSize = '12px';
        badge.style.fontWeight = 'bold';
        badge.style.display = 'flex';
        badge.style.alignItems = 'center';
        badge.style.justifyContent = 'center';
        badge.style.border = '2px solid rgba(255,255,255,0.3)';
        badge.style.zIndex = '101';
        badge.style.opacity = layer.getVisualImpact();
        
        centerElement.appendChild(badge);
    });
}

// CSS animations για single center mode
const singleCenterCSS = `
    @keyframes singleCenterPulse {
        0%, 100% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.05); }
    }
    
    .single-mode {
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .expansion-ring {
        animation: expandRing 3s ease-out forwards;
    }
    
    @keyframes expandRing {
        0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.8; }
        100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
    }
    
    .persona-badge {
        animation: badgeAppear 0.5s ease-out;
    }
    
    @keyframes badgeAppear {
        0% { transform: translate(-50%, -50%) scale(0); opacity: 0; }
        100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }
`;

// Inject CSS
const styleEl = document.createElement('style');
styleEl.textContent = singleCenterCSS;
document.head.appendChild(styleEl);

console.log('🎨 Single Center Renderer loaded');