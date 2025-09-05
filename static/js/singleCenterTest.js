/**
 * Single Center Unit Tests - Comprehensive validation
 */

import { SINGLE_CENTER_CONFIG, PersonaLayer, LearningEvent } from './singleCenterConfig.js';
import { reconcileCenters, mergeLearningEvent, blockSecondaryCenter } from './centerManager.js';

class SingleCenterTestSuite {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }
    
    // Test framework methods
    it(description, testFn) {
        this.tests.push({ description, testFn });
    }
    
    expect(actual) {
        return {
            toBe: (expected) => {
                if (actual !== expected) {
                    throw new Error(`Expected ${expected}, got ${actual}`);
                }
            },
            toHaveLength: (expected) => {
                if (!actual || actual.length !== expected) {
                    throw new Error(`Expected length ${expected}, got ${actual ? actual.length : 'undefined'}`);
                }
            },
            toContain: (expected) => {
                if (!actual || !actual.includes(expected)) {
                    throw new Error(`Expected array to contain ${expected}`);
                }
            }
        };
    }
    
    async runTests() {
        console.log('🧪 Starting Single Center Test Suite...');
        console.log('━'.repeat(60));
        
        for (const test of this.tests) {
            try {
                await test.testFn();
                console.log(`✅ ${test.description}`);
                this.passed++;
            } catch (error) {
                console.error(`❌ ${test.description}`);
                console.error(`   Error: ${error.message}`);
                this.failed++;
            }
        }
        
        console.log('━'.repeat(60));
        console.log(`📊 Test Results: ${this.passed} passed, ${this.failed} failed`);
        
        if (this.failed === 0) {
            console.log('🎉 ALL TESTS PASSED - Single Center enforcement working!');
            return true;
        } else {
            console.log('⚠️ Some tests failed - Review implementation');
            return false;
        }
    }
}

// Initialize test suite
const testSuite = new SingleCenterTestSuite();

// Test 1: Config validation
testSuite.it('enforces single center config', () => {
    testSuite.expect(SINGLE_CENTER_CONFIG.centers.mode).toBe('single');
    testSuite.expect(SINGLE_CENTER_CONFIG.centers.primary_id).toBe('presence-ai');
    testSuite.expect(SINGLE_CENTER_CONFIG.centers.secondary_enabled).toBe(false);
    testSuite.expect(SINGLE_CENTER_CONFIG.centers.max).toBe(1);
});

// Test 2: ReconcileCenters function
testSuite.it('reconcileCenters blocks secondary centers', () => {
    const cfg = { centers: { mode: 'single', primary_id: 'presence-ai' } };
    const primary = { id: 'presence-ai', name: 'YOU / PRESENCE' };
    const secondary = { id: 'awareness', role: 'secondary', name: 'AWARENESS' };
    const out = reconcileCenters([primary, secondary], cfg);
    
    testSuite.expect(out).toHaveLength(1);
    testSuite.expect(out[0].id).toBe('presence-ai');
    testSuite.expect(out[0].role).toBe('primary');
});

// Test 3: Block secondary center creation
testSuite.it('blockSecondaryCenter prevents non-primary creation', () => {
    const secondaryNode = { id: 'awareness', name: 'AWARENESS' };
    const result = blockSecondaryCenter(secondaryNode);
    testSuite.expect(result).toBe(null); // Should be blocked
    
    const primaryNode = { id: 'presence-ai', name: 'YOU / PRESENCE' };
    const primaryResult = blockSecondaryCenter(primaryNode);
    testSuite.expect(primaryResult).toBe(primaryNode); // Should pass through
});

// Test 4: PersonaLayer creation
testSuite.it('creates PersonaLayer with correct properties', () => {
    const layer = new PersonaLayer({
        source: 'book',
        label: 'Tolle',
        weight: 0.8,
        color: '#10b981'
    });
    
    testSuite.expect(layer.source).toBe('book');
    testSuite.expect(layer.label).toBe('Tolle');
    testSuite.expect(layer.weight).toBe(0.8);
    testSuite.expect(layer.color).toBe('#10b981');
});

// Test 5: LearningEvent creation
testSuite.it('creates LearningEvent with persona merge', () => {
    const persona = { name: 'Tolle', weight: 0.7, toneColor: '#10b981' };
    const event = new LearningEvent('persona.merge', { persona, bookId: 'test-123' });
    
    testSuite.expect(event.type).toBe('persona.merge');
    testSuite.expect(event.persona.name).toBe('Tolle');
    testSuite.expect(event.persona.weight).toBe(0.7);
});

// Test 6: Merge learning event functionality
testSuite.it('mergeLearningEvent updates primary center correctly', () => {
    const center = {
        id: 'presence-ai',
        radius: 100,
        aura: { color: '#4facfe', intensity: 0.6 },
        rings: [],
        personaLayers: []
    };
    
    const persona = { name: 'Test Author', weight: 0.5, toneColor: '#10b981' };
    const event = new LearningEvent('persona.merge', { persona });
    
    const merged = mergeLearningEvent(center, event);
    
    // Check aura was updated
    testSuite.expect(merged.aura.color).toBe('#10b981');
    testSuite.expect(merged.aura.intensity >= 0.5).toBe(true);
    
    // Check radius expanded
    testSuite.expect(merged.radius > 100).toBe(true);
    
    // Check rings were added
    testSuite.expect(merged.rings.length > 0).toBe(true);
    
    // Check persona layer was added
    testSuite.expect(merged.personaLayers.length).toBe(1);
    testSuite.expect(merged.personaLayers[0].label).toBe('Test Author');
});

// Test 7: Multiple persona merges
testSuite.it('handles multiple persona merges correctly', () => {
    let center = {
        id: 'presence-ai',
        radius: 100,
        aura: { color: '#4facfe', intensity: 0.6 },
        rings: [],
        personaLayers: []
    };
    
    // First persona
    const persona1 = { name: 'Tolle', weight: 0.7, toneColor: '#10b981' };
    const event1 = new LearningEvent('persona.merge', { persona: persona1 });
    center = mergeLearningEvent(center, event1);
    
    // Second persona
    const persona2 = { name: 'Osho', weight: 0.6, toneColor: '#f59e0b' };
    const event2 = new LearningEvent('persona.merge', { persona: persona2 });
    center = mergeLearningEvent(center, event2);
    
    // Check both personas are present
    testSuite.expect(center.personaLayers).toHaveLength(2);
    
    const authors = center.personaLayers.map(l => l.label);
    testSuite.expect(authors).toContain('Tolle');
    testSuite.expect(authors).toContain('Osho');
    
    // Check radius expanded more
    testSuite.expect(center.radius > 102).toBe(true); // At least 2 expansions
});

// Test 8: Metrics update
testSuite.it('handles metrics updates correctly', () => {
    const center = {
        id: 'presence-ai',
        aura: { color: '#4facfe', intensity: 0.6 }
    };
    
    const metrics = { coherence: 0, vitality: 0 };
    const event = new LearningEvent('metric.update', { metrics });
    
    const updated = mergeLearningEvent(center, event);
    
    // Check aura intensity was updated with EMA
    testSuite.expect(updated.aura.intensity > 0.6).toBe(true);
    testSuite.expect(updated.aura.intensity < 0.85).toBe(true); // EMA smoothing
});

// DOM Cleanup Test
testSuite.it('DOM cleanup functions are available', () => {
    testSuite.expect(typeof window.cleanupSecondaryNodes).toBe('function');
    testSuite.expect(typeof window.reconcileCenters).toBe('function');
    testSuite.expect(typeof window.mergeLearningEvent).toBe('function');
});

// Export test suite
window.singleCenterTestSuite = testSuite;

// Auto-run tests after page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        console.log('🔬 Running Single Center Tests...');
        testSuite.runTests().then(success => {
            if (success) {
                // Create visual indicator
                const indicator = document.createElement('div');
                indicator.innerHTML = '✅ TESTS PASSED';
                indicator.style.position = 'fixed';
                indicator.style.top = '10px';
                indicator.style.right = '10px';
                indicator.style.background = 'green';
                indicator.style.color = 'white';
                indicator.style.padding = '10px';
                indicator.style.borderRadius = '5px';
                indicator.style.zIndex = '9999';
                indicator.style.fontSize = '14px';
                indicator.style.fontWeight = 'bold';
                document.body.appendChild(indicator);
                
                setTimeout(() => indicator.remove(), 5000);
            }
        });
    }, 2000);
});

console.log('🧪 Single Center Test Suite loaded');