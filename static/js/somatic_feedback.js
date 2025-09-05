/**
 * Somatic Feedback - WebAudio + Haptic feedback for YOU motion
 * Provides immersive "feeling" of the movement and biometric state
 */

export class SomaticFeedback {
    constructor() {
        this.audioContext = null;
        this.isInitialized = false;
        this.isFrozen = false;
        
        // Audio nodes
        this.oscillator = null;
        this.gainNode = null;
        this.panNode = null;
        this.lfoOscillator = null;
        this.lfoGain = null;
        
        // State
        this.baseFrequency = 220; // A3
        this.currentIntensity = 0;
        this.currentRate = 0;
        
        console.log('[Somatic] Created');
    }
    
    async init() {
        if (this.isInitialized) return;
        
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create audio graph
            this.createAudioGraph();
            
            this.isInitialized = true;
            console.log('[Somatic] Audio initialized');
            
        } catch (e) {
            console.warn('[Somatic] Audio init failed:', e);
        }
    }
    
    createAudioGraph() {
        // Main oscillator (carrier)
        this.oscillator = this.audioContext.createOscillator();
        this.oscillator.type = 'sine';
        this.oscillator.frequency.setValueAtTime(this.baseFrequency, this.audioContext.currentTime);
        
        // Gain node (volume/intensity)
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        
        // Pan node (left/right movement)
        this.panNode = this.audioContext.createStereoPanner();
        this.panNode.pan.setValueAtTime(0, this.audioContext.currentTime);
        
        // LFO for breathing rhythm
        this.lfoOscillator = this.audioContext.createOscillator();
        this.lfoOscillator.type = 'sine';
        this.lfoOscillator.frequency.setValueAtTime(0.2, this.audioContext.currentTime); // 12 rpm default
        
        this.lfoGain = this.audioContext.createGain();
        this.lfoGain.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        
        // Connect audio graph
        this.oscillator.connect(this.gainNode);
        this.gainNode.connect(this.panNode);
        this.panNode.connect(this.audioContext.destination);
        
        // Connect LFO to main gain (breathing modulation)
        this.lfoOscillator.connect(this.lfoGain);
        this.lfoGain.connect(this.gainNode.gain);
        
        // Start oscillators
        this.oscillator.start();
        this.lfoOscillator.start();
    }
    
    updateFromFrame(frame, motionData = {}) {
        if (!this.isInitialized || this.isFrozen) return;
        
        const now = this.audioContext.currentTime;
        
        // Map vitality and coherence to intensity
        const vitality = frame.vitality / 100;
        const coherence = frame.coherence_pct / 100;
        const intensity = (vitality * 0.7 + coherence * 0.3) * 0.35; // Max 15% volume
        
        // Map breath rate to LFO frequency
        const breathHz = Math.max(0.05, Math.min(0.35, frame.breath_rpm / 60)); // Convert RPM to Hz, scaled
        
        // Map empathy to frequency modulation
        const empathyMod = (frame.empathy / 100) * 90; // ±50Hz modulation
        const targetFreq = this.baseFrequency + empathyMod;
        
        // Apply changes smoothly
        this.gainNode.gain.linearRampToValueAtTime(intensity, now + 0.1);
        this.oscillator.frequency.linearRampToValueAtTime(targetFreq, now + 0.2);
        this.lfoOscillator.frequency.linearRampToValueAtTime(breathHz, now + 0.3);
        
        // Pan based on motion angle
        if (motionData.angle !== undefined) {
            const panValue = Math.sin(motionData.angle) * 0.5; // -0.5 to 0.5
            this.panNode.pan.linearRampToValueAtTime(panValue, now + 0.1);
        }
        
        // Optional haptic feedback
        this.triggerHaptic(frame);
        
        this.currentIntensity = intensity;
        this.currentRate = breathHz;
    }
    
    triggerHaptic(frame) {
        // Only trigger on significant changes
        if (!navigator.vibrate) return;
        
        const coherence = frame.coherence_pct / 100;
        const vitality = frame.vitality / 100;
        
        // Gentle pulse on high coherence + vitality
        if (coherence > 0.7 && vitality > 0.7) {
            if (Math.random() < 0.1) { // 10% chance per frame
                navigator.vibrate([50]); // Short gentle pulse
            }
        }
    }
    
    freeze() {
        if (!this.isInitialized) return;
        
        this.isFrozen = true;
        
        // Fade out audio
        const now = this.audioContext.currentTime;
        this.gainNode.gain.linearRampToValueAtTime(0, now + 0.5);
        
        console.log('[Somatic] Frozen');
    }
    
    unfreeze() {
        this.isFrozen = false;
        console.log('[Somatic] Unfrozen');
    }
    
    stop() {
        if (!this.isInitialized) return;
        
        try {
            const now = this.audioContext.currentTime;
            this.gainNode.gain.linearRampToValueAtTime(0, now + 0.2);
            
            setTimeout(() => {
                if (this.oscillator) {
                    this.oscillator.stop();
                    this.oscillator = null;
                }
                if (this.lfoOscillator) {
                    this.lfoOscillator.stop();
                    this.lfoOscillator = null;
                }
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
            }, 300);
            
            this.isInitialized = false;
            console.log('[Somatic] Stopped');
            
        } catch (e) {
            console.warn('[Somatic] Stop error:', e);
        }
    }
    
    getState() {
        return {
            initialized: this.isInitialized,
            frozen: this.isFrozen,
            intensity: this.currentIntensity,
            rate: this.currentRate
        };
    }
}