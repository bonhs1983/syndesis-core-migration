/**
 * Panel Bridge - Event Bus & WebSocket Watchdog
 * Handles frame validation, timing, and freeze detection
 */

export class PanelBridge {
    constructor(wsUrl = null) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.lastFrameTs = 0;
        this.watchdogTimer = null;
        this.isLive = false;
        
        // Event bus
        this.events = new EventTarget();
        
        // Frame validator
        this.validator = new FrameValidator();
        
        // Start watchdog
        this.startWatchdog();
        
        // Initialize WebSocket if URL provided
        if (this.wsUrl) {
            this.initWebSocket();
        }
        
        console.log('[PanelBridge] Initialized', { wsUrl: this.wsUrl });
    }
    
    initWebSocket() {
        try {
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('[PanelBridge] WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleFrame(data);
                } catch (e) {
                    console.warn('[PanelBridge] Invalid WebSocket frame:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('[PanelBridge] WebSocket closed, reconnecting...');
                setTimeout(() => this.initWebSocket(), 2000);
            };
            
            this.ws.onerror = (error) => {
                console.warn('[PanelBridge] WebSocket error:', error);
            };
            
        } catch (e) {
            console.warn('[PanelBridge] WebSocket init failed:', e);
        }
    }
    
    emit(event, frame) {
        if (event === 'frame') {
            this.handleFrame(frame);
        }
    }
    
    handleFrame(rawFrame) {
        // Debug logging για frame reception
        console.log('[PanelBridge] Raw frame received:', {
            type: typeof rawFrame,
            keys: Object.keys(rawFrame || {}),
            hrv_ms: rawFrame?.hrv_ms,
            coherence_pct: rawFrame?.coherence_pct,
            ts: rawFrame?.ts
        });
        
        // Validate and clean frame
        const frame = this.validator.validate(rawFrame);
        
        if (!frame) {
            console.warn('[PanelBridge] Invalid frame rejected:', rawFrame);
            return;
        }
        
        // Debug τι validation έγινε
        console.log('[PanelBridge] Frame validated:', {
            hrv_ms: frame.hrv_ms,
            coherence_pct: frame.coherence_pct,
            focus: frame.focus,
            vitality: frame.vitality,
            ts: frame.ts
        });
        
        // Check if live signal
        const hasLiveSignal = frame.hrv_ms > 0 && frame.coherence_pct > 0;
        
        if (hasLiveSignal) {
            this.lastFrameTs = frame.ts;
            this.isLive = true;
            
            console.log('[PanelBridge] LIVE signal detected, emitting live-frame');
            
            // Emit live frame
            this.events.dispatchEvent(new CustomEvent('live-frame', { 
                detail: frame 
            }));
        } else {
            // Dead signal
            this.isLive = false;
            console.log('[PanelBridge] NO signal detected, emitting no-signal');
            this.events.dispatchEvent(new CustomEvent('no-signal'));
        }
    }
    
    startWatchdog() {
        this.watchdogTimer = setInterval(() => {
            const now = Date.now();
            const timeSinceLastFrame = now - this.lastFrameTs;
            
            // Debug logging για watchdog
            if (this.isLive) {
                console.log(`[PanelBridge] Watchdog check: ${timeSinceLastFrame}ms since last frame`);
            }
            
            if (timeSinceLastFrame > 2500 && this.isLive) {
                console.log('[PanelBridge] ⚠️ FREEZE TIMEOUT detected, freezing system');
                this.isLive = false;
                this.events.dispatchEvent(new CustomEvent('freeze'));
            }
        }, 500); // Check every 500ms
    }
    
    on(event, callback) {
        this.events.addEventListener(event, callback);
    }
    
    off(event, callback) {
        this.events.removeEventListener(event, callback);
    }
    
    destroy() {
        if (this.watchdogTimer) {
            clearInterval(this.watchdogTimer);
        }
        if (this.ws) {
            this.ws.close();
        }
    }
}

class FrameValidator {
    validate(frame) {
        if (!frame || typeof frame !== 'object') {
            return null;
        }
        const lower = {};
        for (const k in frame) {
            if (Object.hasOwn(frame, k)) {
                lower[k.toLowerCase()] = frame[k];
            }
        }
        // Accept both lower/camel and capitalized keys
        function pick(name, fallbackName, def=0) {
            if (lower[name] != null) return lower[name];
            if (frame[fallbackName] != null) return frame[fallbackName];
            if (frame[name] != null) return frame[name];
            return def;
        }
        // Normalize a metric that might be in 0–1 range to 0–100
        function toPct(v) {
            let x = Number(v);
            if (!isFinite(x)) x = 0;
            if (x <= 1 && x >= 0) x = x * 100;
            return Math.max(0, Math.min(100, x));
        }
        // Numeric helper
        function num(v, min, max) {
            let x = Number(v);
            if (!isFinite(x)) x = 0;
            if (min != null || max != null) {
                if (min == null) min = -Infinity;
                if (max == null) max = Infinity;
                x = Math.max(min, Math.min(max, x));
            }
            return x;
        }
        const ts = typeof frame.ts === 'number' ? frame.ts : Date.now();
        const validated = {
            ts,
            hrv_ms: num(pick('hrv_ms', 'HRV', 0), 0, 400),
            breath_rpm: num(pick('breath_rpm', 'Breath', 0), 0, 30),
            coherence_pct: toPct(pick('coherence_pct', 'Coherence', 0)),
            focus: toPct(pick('focus', 'Focus', 0)),
            vitality: toPct(pick('vitality', 'Vitality', 0)),
            empathy: toPct(pick('empathy', 'Empathy', 0)),
            creativity: toPct(pick('creativity', 'Creativity', 0)),
            curiosity: toPct(pick('curiosity', 'Curiosity', 0)),
            resilience: toPct(pick('resilience', 'Resilience', 0)),
        };
        return validated;
    
    }
    
    
    clip(value, min, max) {
        if (typeof value !== 'number' || isNaN(value)) {
            return 0;
        }
        return Math.max(min, Math.min(max, value));
    }
}

// Global panel instance
window.PANEL = null;