// 🧠 NEURAL CONSCIOUSNESS MAPPING - Biometric Motion Engine
// Revolutionary multidimensional YOU orb with real-time HRV consciousness mapping

(function() {
  'use strict';
  
  console.log('🧠 NEURAL CONSCIOUSNESS MAPPING - Starting...');
  
  // ===== CONSTANTS ===== //
  const UPDATE_RATE_MS = 800;     // Data polling rate
  const LIVE_CHECK_MS = 2000;     // Signal timeout - ΑΜΕΣΟ TIMEOUT!
  
  // ===== GLOBAL STATE ===== //
  let isLiveSignal = false;
  let lastLiveSignal = 0;
  let youEngine = null;
  let isEmotionDisplayManuallyHidden = false; // 🎯 Track user toggle preference
  
  // ===== NEURAL ENGINE CORE ===== //
  class NeuralConsciousnessEngine {
    constructor() {
      this.pos = {x: 0, y: 0};
      this.target = {x: 0, y: 0};
      this.freeze = true;
      this.arousalGain = 0.5;
      this.jitterLevel = 0.0;
      this.breathPhase = 0.0;
      this.initialized = false;
      
      console.log('🧠 Neural Consciousness Engine initialized');
    }
    
    setFreeze(flag) { 
      this.freeze = flag; 
      console.log(`🧠 ENGINE STATE: ${flag ? '🔒 FROZEN' : '🚀 NEURAL ACTIVE'}`);
    }
    
    update(data) {
      if (!data) return;
      
      console.log('🧠 NEURAL INPUT:', {
        coherence: data.coherence,
        vitality: data.vitality,
        rmssd: data.rmssd,
        empathy: data.empathy,
        creativity: data.creativity
      });
      
      // Extract normalized metrics
      const coherence = (data.coherence || 0) / 100;
      const vitality = (data.vitality || 0) / 100; 
      const ethics = (data.ethics || 0) / 100;
      const narrative = (data.narrative || 0) / 100;
      const rmssd = Math.min((data.rmssd || 0) / 50, 1);
      const empathy = (data.empathy || 0) / 100;
      const creativity = (data.creativity || 0) / 100;
      
      // 🧠 REVOLUTIONARY MULTIDIMENSIONAL CONSCIOUSNESS MAPPING
      // X-axis: Empathy/Compassion (left) ↔ Analytical/Focus (right)  
      // Y-axis: Χαρά/Αυτοπεποίθηση/Ευτυχία (up) ↔ Άγχος/Στρες (down)
      
      // 🚀 ULTRA-SENSITIVE MULTIDIMENSIONAL CONSCIOUSNESS MAPPING  
      // Adding base emotional variance from soul metrics for massive movement
      const emotionalVariance = (coherence - vitality + ethics - narrative) * 2.0;
      
      // LEFT ← Empathy/Compassion | Analytical/Focus → RIGHT
      const empathy_compassion = (empathy * 1.5 + vitality * 0.8 + rmssd * 1.2 + emotionalVariance * 0.5) / 2.0;
      const analytical_focus = (coherence * 1.8 + ethics * 1.0 + (1.0 - rmssd) * 0.8 + Math.abs(emotionalVariance) * 0.7) / 2.5;
      
      // 🎭 REVOLUTIONARY 24-EMOTION CONSCIOUSNESS MAPPING!
      let positive_emotions, negative_stress, empathy_creative_score, analytical_focus_score;
      
      if (window.latestEmotionalData && window.latestEmotionalData.live && window.latestEmotionalData.emotions) {
        // 🚀 USE ALL 24 EMOTIONS for ultra-precise movement calculation
        const emotions = window.latestEmotionalData.emotions;
        
        // 🌟 POSITIVE EMOTIONS (UP ↑): Joy, Confidence, Hope, Satisfaction, Content, Gratitude
        const joy = (emotions.joy || 50) / 100;
        const confidence = (emotions.confidence || 50) / 100;
        const hope = (emotions.hope || 50) / 100;
        const satisfaction = (emotions.satisfaction || 50) / 100;
        const contentment = (emotions.contentment || emotions.content || 50) / 100;
        const gratitude = (emotions.gratitude || 50) / 100;
        
        console.log('🌟 POSITIVE EMOTIONS RAW:', {joy: emotions.joy, confidence: emotions.confidence, hope: emotions.hope, gratitude: emotions.gratitude});
        
        positive_emotions = (joy * 2.5 + confidence * 2.0 + hope * 1.8 + 
                           satisfaction * 1.5 + contentment * 1.2 + gratitude * 1.0) / 6.0;
        
        // ⚡ NEGATIVE EMOTIONS (DOWN ↓): Sadness, Despair, Worry, Fear, Frustration, Anger
        const sadness = (emotions.sadness || 50) / 100;
        const despair = (emotions.despair || 50) / 100;
        const worry = (emotions.worry || 50) / 100;
        const fear = (emotions.fear || 50) / 100;
        const frustration = (emotions.frustration || 50) / 100;
        const anger = (emotions.anger || 50) / 100;
        
        console.log('⚡ NEGATIVE EMOTIONS RAW:', {sadness: emotions.sadness, worry: emotions.worry, fear: emotions.fear, anger: emotions.anger});
        
        negative_stress = (sadness * 1.8 + despair * 2.0 + worry * 1.5 + 
                          fear * 1.2 + frustration * 1.3 + anger * 1.1) / 6.0;
        
        // 💖 EMPATHY/CREATIVITY (LEFT ←): Empathy, Compassion, Love, Connection, Inspiration, Wonder
        const empathy = (emotions.empathy || 50) / 100;
        const compassion = (emotions.compassion || 50) / 100;
        const love = (emotions.love || 50) / 100;
        const connection = (emotions.connection || 50) / 100;
        const inspiration = (emotions.inspiration || 50) / 100;
        const wonder = (emotions.wonder || 50) / 100;
        
        empathy_creative_score = (empathy * 2.2 + compassion * 1.9 + love * 1.6 + 
                                connection * 1.4 + inspiration * 1.3 + wonder * 1.1) / 6.0;
        
        // 🎯 ANALYTICAL/FOCUS (RIGHT →): Concentration, Determination, Alertness, Clarity, Mindfulness, Serenity
        const concentration = (emotions.concentration || 50) / 100;
        const determination = (emotions.determination || 50) / 100;
        const alertness = (emotions.alertness || 50) / 100;
        const clarity = (emotions.clarity || 50) / 100;
        const mindfulness = (emotions.mindfulness || 50) / 100;
        const serenity = (emotions.serenity || 50) / 100;
        
        analytical_focus_score = (concentration * 2.2 + determination * 1.9 + alertness * 1.6 + 
                                clarity * 1.4 + mindfulness * 1.3 + serenity * 1.1) / 6.0;
        
        console.log('🎭 24-EMOTION CONSCIOUSNESS MAPPING:', {
          positive: (positive_emotions * 100).toFixed(1) + '%',
          negative: (negative_stress * 100).toFixed(1) + '%',
          empathy_creative: (empathy_creative_score * 100).toFixed(1) + '%',
          analytical_focus: (analytical_focus_score * 100).toFixed(1) + '%',
          dominant: window.latestEmotionalData.dominantEmotion
        });
        
      } else {
        // FALLBACK: Traditional biometric-based emotional calculation
        positive_emotions = (vitality * 1.5 + coherence * 1.2 + rmssd * 1.0 + creativity * 0.8 + emotionalVariance * 0.5) / 3.0;
        negative_stress = (ethics * 0.6 + (1.0 - coherence) * 1.8 + (1.0 - rmssd) * 1.2 + Math.abs(emotionalVariance) * 0.8) / 2.5;
        empathy_creative_score = empathy_compassion;
        analytical_focus_score = analytical_focus;
      }
      
      // 🎯 PRIORITY 1: PURE 1:1 LINEAR MAPPING - No Distortion!
      if (window.latestEmotionalData && window.latestEmotionalData.backendCoordinates && 
          typeof window.latestEmotionalData.backendCoordinates.x === 'number' &&
          typeof window.latestEmotionalData.backendCoordinates.y === 'number') {
        
        // 🎯 CORRECTED MAPPING: Backend coordinates (200-400 range) → Screen coordinates
        const center = getCenterPosition();
        
        // Backend coordinates are in 200-400 range, normalize to screen range
        const backendCenterX = 300; // Backend coordinate system center
        const backendCenterY = 300;
        const screenRangeX = center.x * 0.6; // 60% of screen width for movement
        const screenRangeY = center.y * 0.6; // 60% of screen height for movement
        
        // Convert backend coordinates to screen offsets
        const backendDeltaX = window.latestEmotionalData.backendCoordinates.x - backendCenterX;
        const backendDeltaY = window.latestEmotionalData.backendCoordinates.y - backendCenterY;
        
        // Scale to screen coordinates (backend range ~100px = screen range)  
        this.target.x = (backendDeltaX / 100) * screenRangeX;
        this.target.y = (backendDeltaY / 100) * screenRangeY;
        
        console.log('🎯 PURE 1:1 MAPPING - PROPORTIONAL MOVEMENT:', {
          backend_x: window.latestEmotionalData.backendCoordinates.x,
          backend_y: window.latestEmotionalData.backendCoordinates.y,
          center_x: center.x,
          center_y: center.y,
          orb_target_x: this.target.x.toFixed(1),
          orb_target_y: this.target.y.toFixed(1),
          backend_delta: `Δx=${backendDeltaX.toFixed(1)}, Δy=${backendDeltaY.toFixed(1)}`,
          screen_range: `${screenRangeX.toFixed(0)}x${screenRangeY.toFixed(0)}`,
          mapping: 'Backend-to-Screen corrected mapping!',
          movement_visible: Math.abs(this.target.x) > 50 || Math.abs(this.target.y) > 50 ? '✅ DRAMATIC' : Math.abs(this.target.x) > 20 || Math.abs(this.target.y) > 20 ? '⚠️ MODERATE' : '❌ SMALL'
        });
        
        // Clean arousal calculation (normalized 0-1)
        this.arousalGain = (coherence + vitality * 0.01) / 100;
        this.jitterLevel = Math.max(0, (100 - coherence - vitality) / 200);
        
        return; // PERFECT - Pure backend mapping complete!
      }
      
      // 🎯 FALLBACK: 24-EMOTION ULTRA-SENSITIVE CONSCIOUSNESS COORDINATES
      const empathy_deviation = empathy_creative_score - 0.5;
      const analytical_deviation = analytical_focus_score - 0.5;
      const emotion_balance = positive_emotions - negative_stress;
      
      // 🚀 HYPER-SENSITIVE MOVEMENT: ΕΛΑΧΙΣΤΑ thresholds για μεγάλη κίνηση!
      let x_consciousness = 0;
      let y_consciousness = 0;
      
      // 🎯 AMPLIFIED LINEAR MAPPING: Wider movement range (-400px to +400px)
      // X-AXIS: Empathy/Creative (left) ↔ Analytical/Focus (right)
      const x_balance = analytical_deviation - empathy_deviation;
      x_consciousness = x_balance * 400; // Amplified mapping: ±1.0 → ±400px for dramatic movement!
      
      // Y-AXIS: Positive emotions (up) ↔ Negative emotions (down)
      y_consciousness = emotion_balance * 400; // Amplified mapping: ±1.0 → ±400px for dramatic movement!
      
      console.log('🚀 QUANTUM-SENSITIVE CALCULATION:', {
        raw_emotion_balance: emotion_balance.toFixed(3),
        raw_x_balance: x_balance.toFixed(3),
        threshold_triggered_y: Math.abs(emotion_balance) > 0.005,
        threshold_triggered_x: Math.abs(x_balance) > 0.005,
        final_movement_x: x_consciousness.toFixed(1),
        final_movement_y: y_consciousness.toFixed(1),
        emotion_intensity: ((positive_emotions + negative_stress) * 50).toFixed(1) + '%'
      });
      
      console.log('🎯 24-EMOTION ULTRA-PRECISION MAPPING:', {
        empathy_creative: (empathy_creative_score * 100).toFixed(1) + '%',
        analytical_focus: (analytical_focus_score * 100).toFixed(1) + '%',
        emotion_balance: (emotion_balance * 100).toFixed(1) + '%',
        x_movement: x_consciousness.toFixed(1),
        y_movement: y_consciousness.toFixed(1)
      });
      
      // 🎭 EMOTIONAL INTENSITY AMPLIFIER - Use dominant emotion intensity
      let emotionalAmplifier = 1.0;
      if (window.latestEmotionalData && window.latestEmotionalData.live) {
        emotionalAmplifier = 1.0 + (window.latestEmotionalData.emotionalIntensity * 1.5);
      }
      
      // 🎯 ΣΥΓΧΡΟΝΙΣΜΟΣ: Χρήση consciousness coordinates από backend αν υπάρχουν
      if (data.consciousness && data.consciousness.x !== undefined && data.consciousness.y !== undefined) {
        // Backend έχει ήδη υπολογίσει τις σωστές συντεταγμένες - τις χρησιμοποιούμε!
        // Χρησιμοποιούμε το dynamic center position αντί hardcoded values
        const center = getCenterPosition();
        this.target.x = data.consciousness.x - center.x; // Dynamic center calculation
        this.target.y = data.consciousness.y - center.y; // Dynamic center calculation
        console.log('🎯 BACKEND CONSCIOUSNESS SYNC: {raw_x:', data.consciousness.x, ', raw_y:', data.consciousness.y, '}');
        
        console.log('🎯 BACKEND CONSCIOUSNESS SYNC:', {
          raw_x: data.consciousness.x.toFixed(1),
          raw_y: data.consciousness.y.toFixed(1),
          center_x: center.x,
          center_y: center.y,
          relative_x: this.target.x.toFixed(1),
          relative_y: this.target.y.toFixed(1)
        });
      } else {
        // FALLBACK: Direct linear frontend calculation (1:1 mapping)
        this.target.x = x_consciousness; // Direct pixel mapping - no amplification!
        this.target.y = y_consciousness; // Direct pixel mapping - no amplification!
        
        console.log('🎯 LINEAR MAPPING - Direct proportional movement:', {
          normalized_x: x_balance.toFixed(3),
          normalized_y: emotion_balance.toFixed(3),
          pixel_x: this.target.x.toFixed(1),
          pixel_y: this.target.y.toFixed(1),
          mapping: '3x AMPLIFIED proportional (dramatic range)'
        });
      }
      
      // No orbital motion - pure 1:1 mapping only!
      
      // Dynamic arousal from HRV and consciousness alignment
      this.arousalGain = (coherence + vitality * 0.5 + rmssd * 0.3) / 1.8;
      
      // Stress jitter from low coherence/HRV
      this.jitterLevel = Math.max(0, 0.6 - coherence - rmssd * 0.5);
      
      console.log('🧠 CONSCIOUSNESS COORDINATES:', {
        target: `(${this.target.x.toFixed(1)}, ${this.target.y.toFixed(1)})`,
        arousal: (this.arousalGain * 100).toFixed(1) + '%',
        stress: (this.jitterLevel * 100).toFixed(1) + '%'
      });
    }
    
    tick() {
      if (this.freeze) return;
      
      // 🚀 FASTER MOVEMENT for visible 1:1 proportional mapping!
      const easing = 0.6; // Much faster movement to show metric changes immediately
      this.pos.x += (this.target.x - this.pos.x) * easing;
      this.pos.y += (this.target.y - this.pos.y) * easing;
      
      // 🎯 DEBUG: Log actual movement for verification
      if (Math.abs(this.target.x - this.pos.x) > 1 || Math.abs(this.target.y - this.pos.y) > 1) {
        console.log('🎯 ORB MOVEMENT:', {
          target: `(${this.target.x.toFixed(1)}, ${this.target.y.toFixed(1)})`,
          current: `(${this.pos.x.toFixed(1)}, ${this.pos.y.toFixed(1)})`,
          delta: `(${(this.target.x - this.pos.x).toFixed(1)}, ${(this.target.y - this.pos.y).toFixed(1)})`,
          easing: (easing * 100) + '%'
        });
      }
      
      // Breathing synchronization
      this.breathPhase += 0.025;
      const breathOffset = Math.sin(this.breathPhase) * 4 * this.arousalGain;
      
      // Stress-induced jitter
      const jitterX = (Math.random() - 0.5) * this.jitterLevel * 15;
      const jitterY = (Math.random() - 0.5) * this.jitterLevel * 15;
      
      // Final position with micro-movements
      this.currentX = this.pos.x + breathOffset + jitterX;
      this.currentY = this.pos.y + breathOffset + jitterY;
      
      // 🎯 DEBUG: Track the complete movement chain
      console.log('🎯 TICK MOVEMENT CHAIN:', {
        pos: `(${this.pos.x.toFixed(1)}, ${this.pos.y.toFixed(1)})`,
        breath: breathOffset.toFixed(1),
        jitter: `(${jitterX.toFixed(1)}, ${jitterY.toFixed(1)})`,
        final_current: `(${this.currentX.toFixed(1)}, ${this.currentY.toFixed(1)})`
      });
      
      // Boundary constraint (keep within ±400px for wider range)
      const radius = Math.hypot(this.currentX, this.currentY);
      if (radius > 400) {
        const scale = 400 / radius;
        this.currentX *= scale;
        this.currentY *= scale;
      }
    }
    
    getEffects() {
      const time = Date.now() * 0.001;
      const intensity = 0.6 + 0.4 * this.arousalGain;
      
      // 🌈 PSYCHEDELIC PULSING based on HRV  
      const pulseScale = 1.0 + Math.sin(time * 4) * 0.3 * this.arousalGain;
      const breathPulse = 1.0 + Math.sin(time * 0.7) * 0.2;
      
      // 🎨 EMOTIONAL STATE COLOR MAPPING (not continuous)
      const biometricData = window.latestBiometricData;
      const baseHue = biometricData ? (
        (biometricData.coherence * 3 + biometricData.vitality * 2 + biometricData.ethics * 2.5) % 360
      ) : 180;
      const saturation = 80 + this.arousalGain * 20;
      
      // ✨ MULTI-LAYERED PSYCHEDELIC GLOW
      const innerGlow = `0 0 ${20 + this.arousalGain * 30}px rgba(255, 255, 255, 0.9)`;
      const middleGlow = `0 0 ${40 + this.arousalGain * 50}px hsl(${baseHue}, ${saturation}%, 70%)`;
      const outerGlow = `0 0 ${70 + this.arousalGain * 80}px hsl(${baseHue + 60}, ${saturation}%, 60%)`;
      const cosmicGlow = `0 0 ${120 + this.arousalGain * 150}px hsl(${baseHue + 120}, ${saturation + 10}%, 50%)`;
      
      // 🌀 FRACTAL ROTATION based on consciousness state  
      const rotation = (baseHue * 0.5) % 360;
      
      return {
        scale: pulseScale * breathPulse,
        opacity: 0.75 + 0.25 * intensity,
        boxShadow: `${innerGlow}, ${middleGlow}, ${outerGlow}, ${cosmicGlow}`,
        filter: `blur(${Math.max(0, 1 - this.arousalGain * 2)}px) 
                 hue-rotate(${baseHue}deg) 
                 saturate(${120 + this.arousalGain * 80}%) 
                 contrast(${110 + this.arousalGain * 40}%)
                 brightness(${100 + this.arousalGain * 20}%)`,
        transform: `rotate(${rotation * 0.5}deg)`,
        background: `radial-gradient(circle at 40% 30%, 
                    rgba(255, 255, 255, 0.95) 0%,
                    hsl(${baseHue}, ${saturation}%, 75%) 30%,
                    hsl(${baseHue + 90}, ${saturation}%, 60%) 60%,
                    hsl(${baseHue + 180}, ${saturation}%, 45%) 100%)`
      };
    }
  }
  
  // ===== 24-EMOTION VISUAL DISPLAY ===== //
  function createEmotionDisplay() {
    let display = document.getElementById('emotionDisplay');
    if (display) return display;
    
    display = document.createElement('div');
    display.id = 'emotionDisplay';
    display.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      width: 300px;
      height: 200px;
      background: rgba(0, 0, 0, 0.8);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 15px;
      backdrop-filter: blur(10px);
      padding: 15px;
      z-index: 999998;
      font-family: monospace;
      color: white;
      font-size: 11px;
      overflow-y: auto;
      display: none;
    `;
    
    document.body.appendChild(display);
    console.log('🎭 24-EMOTION Visual Display created');
    return display;
  }
  
  function updateEmotionDisplay(emotions) {
    const display = createEmotionDisplay();
    if (!emotions || Object.keys(emotions).length === 0) {
      display.style.display = 'none';
      return;
    }
    
    // 🎯 RESPECT USER TOGGLE CHOICE - Only show if not manually hidden
    if (!isEmotionDisplayManuallyHidden) {
      display.style.display = 'block';
    }
    
    // 🌈 EMOTION COLOR MAPPING
    const emotionColors = {
      // POSITIVE EMOTIONS - Warm colors
      joy: '#FFD700', confidence: '#FFA500', hope: '#FF69B4',
      satisfaction: '#00CED1', contentment: '#98FB98', gratitude: '#DDA0DD',
      
      // NEGATIVE EMOTIONS - Cool/Red colors  
      sadness: '#4682B4', despair: '#2F4F4F', worry: '#8B4513',
      fear: '#DC143C', frustration: '#B22222', anger: '#FF0000',
      
      // FOCUS EMOTIONS - Blue spectrum
      concentration: '#0000FF', determination: '#4169E1', alertness: '#6495ED',
      clarity: '#87CEEB', mindfulness: '#ADD8E6', serenity: '#E0E6FF',
      
      // EMPATHY EMOTIONS - Green/Purple spectrum
      empathy: '#9370DB', compassion: '#BA55D3', love: '#FF1493',
      connection: '#FF69B4', inspiration: '#DDA0DD', wonder: '#DA70D6'
    };
    
    let html = '<div style="text-align: center; margin-bottom: 10px; font-weight: bold;">🎭 24-EMOTION CONSCIOUSNESS</div>';
    
    // Group emotions by category
    const categories = {
      '🌟 POSITIVE': ['joy', 'confidence', 'hope', 'satisfaction', 'contentment', 'gratitude'],
      '⚡ NEGATIVE': ['sadness', 'despair', 'worry', 'fear', 'frustration', 'anger'], 
      '🎯 FOCUS': ['concentration', 'determination', 'alertness', 'clarity', 'mindfulness', 'serenity'],
      '💖 EMPATHY': ['empathy', 'compassion', 'love', 'connection', 'inspiration', 'wonder']
    };
    
    Object.entries(categories).forEach(([category, emotionList]) => {
      html += `<div style="margin: 8px 0; font-weight: bold;">${category}</div>`;
      html += '<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">';
      
      emotionList.forEach(emotion => {
        const value = emotions[emotion] || 50;
        const intensity = Math.max(0, Math.min(1, (value - 30) / 40)); // Map 30-70 to 0-1
        const color = emotionColors[emotion] || '#FFFFFF';
        const size = 8 + intensity * 12; // Size based on intensity
        const opacity = 0.3 + intensity * 0.7;
        
        html += `<div style="
          width: ${size}px;
          height: ${size}px;
          background: ${color};
          border-radius: 50%;
          opacity: ${opacity};
          box-shadow: 0 0 ${intensity * 6}px ${color};
          margin: 1px;
          position: relative;
        " title="${emotion}: ${value.toFixed(1)}%"></div>`;
      });
      
      html += '</div>';
    });
    
    display.innerHTML = html;
  }
  
  // ===== EXPERIMENTAL EMOTIONS FETCHER ===== //
  function fetchEmotionsData() {
    fetch('/neural/emotions')
      .then(response => response.json())
      .then(emotionData => {
        if (emotionData.live && emotionData.emotions) {
          window.latestEmotionalData = {
            emotions: emotionData.emotions,
            dominantEmotion: emotionData.dominant_emotion,
            emotionalIntensity: emotionData.emotional_intensity,
            backendCoordinates: emotionData.coordinates, // 🎯 BACKEND COORDINATES!
            live: true,
            disclaimer: emotionData.disclaimer
          };
          
          console.log('🎯 BACKEND COORDINATES RECEIVED:', emotionData.coordinates);
          
          // 🎭 UPDATE VISUAL EMOTION DISPLAY
          updateEmotionDisplay(emotionData.emotions);
          
          console.log('🎭 24-EMOTION SYSTEM UPDATE:', {
            dominant: emotionData.dominant_emotion,
            intensity: (emotionData.emotional_intensity * 100).toFixed(1) + '%',
            joy: emotionData.emotions.joy?.toFixed(1) || '50.0',
            sadness: emotionData.emotions.sadness?.toFixed(1) || '50.0',
            empathy: emotionData.emotions.empathy?.toFixed(1) || '50.0',
            focus: emotionData.emotions.concentration?.toFixed(1) || '50.0'
          });
        } else {
          window.latestEmotionalData = {
            live: false,
            emotions: {},
            dominantEmotion: 'neutral',
            emotionalIntensity: 0.5
          };
        }
      })
      .catch(err => {
        console.error('🔴 /neural/emotions ERROR:', {
          error: err.message || err,
          timestamp: new Date().toISOString(),
          url: '/neural/emotions'
        });
        window.latestEmotionalData = {
          live: false,
          emotions: {},
          dominantEmotion: 'neutral',
          emotionalIntensity: 0.5
        };
        
        // 🔄 RETRY after 3 seconds if it's a temporary issue
        setTimeout(() => {
          console.log('🔄 RETRYING /neural/emotions after error...');
          fetchEmotionsData();
        }, 3000);
      });
  }

  // ===== DIRECT API POLL FOR INSTANT DEATH DETECTION ===== //
  function fetchLiveData() {
    // 🚨 CRITICAL: Direct API call to get REAL live status
    fetch('/live-data')
      .then(response => response.json())
      .then(data => {
        const previousSignal = isLiveSignal;
        
        // 🎯 AUTHORITATIVE SOURCE: Backend decides live status
        isLiveSignal = data.hasLiveSignal === true;
        
        if (previousSignal !== isLiveSignal) {
          console.log(`🔄 BACKEND AUTHORITY: ${isLiveSignal ? '🟢 LIVE' : '💀 DEAD'}`);
        }
        
        // Store data globally for neural engine
        if (isLiveSignal) {
          window.latestBiometricData = {
            coherence: data.soul_metrics?.coherence || 0,
            vitality: data.soul_metrics?.vitality || 0,
            ethics: data.soul_metrics?.ethics || 0,
            narrative: data.soul_metrics?.narrative || 0,
            empathy: data.evolution?.empathy || 0,
            creativity: data.evolution?.creativity || 0,
            resilience: data.evolution?.resilience || 0,
            focus: data.evolution?.focus || 0,
            curiosity: data.evolution?.curiosity || 0,
            compassion: data.evolution?.compassion || 0,
            rmssd: data.hrv?.rmssd || 0,
            // 🎯 ΣΥΓΧΡΟΝΙΣΜΟΣ: Χρήση consciousness coordinates από backend
            consciousness_x: data.consciousness?.x || 0,
            consciousness_y: data.consciousness?.y || 0
          };
          
          lastLiveSignal = Date.now();
          console.log('🧠 LIVE DATA UPDATED:', window.latestBiometricData);
        } else {
          // 💀 INSTANT DEATH: Clear all data
          window.latestBiometricData = null;
          console.log('💀 DEAD SIGNAL: All biometric data cleared');
        }
      })
      .catch(err => {
        console.error('🔴 /live-data ERROR:', {
          error: err.message || err,
          timestamp: new Date().toISOString(),
          url: '/live-data'
        });
        isLiveSignal = false;
        window.latestBiometricData = null;
        
        // 🔄 RETRY after 3 seconds if it's a temporary issue
        setTimeout(() => {
          console.log('🔄 RETRYING /live-data after error...');
          fetchLiveData();
        }, 3000);
      });
  }
  
  // ===== YOU ORB MANAGEMENT ===== //
  // ===== BIDIRECTIONAL CONSCIOUSNESS CONTROL VARIABLES ===== //
  let isDragging = false;
  let dragStartPos = {x: 0, y: 0};
  let intentionOverride = {active: false, x: 0, y: 0, intensity: 0};
  
  function createYouOrb() {
    let orb = document.getElementById('youOrb');
    if (orb) return orb;
    
    orb = document.createElement('div');
    orb.id = 'youOrb';
    orb.className = 'user-state neural-consciousness';
    orb.style.cssText = `
      position: fixed;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, 
        rgba(255, 255, 255, 0.9) 0%,
        rgba(100, 200, 255, 0.8) 40%,
        rgba(50, 150, 255, 0.6) 100%);
      border: 3px solid rgba(255, 255, 255, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      color: white;
      text-shadow: 0 0 10px rgba(0,0,0,0.8);
      z-index: 999999;
      backdrop-filter: blur(5px);
      cursor: grab;
      user-select: none;
    `;
    orb.textContent = 'YOU';
    
    // 🎯 BIDIRECTIONAL CONSCIOUSNESS CONTROL - DRAG & DROP TRAINING
    orb.addEventListener('mousedown', startDrag);
    orb.addEventListener('touchstart', startDrag, {passive: false});
    
    document.addEventListener('mousemove', drag);
    document.addEventListener('touchmove', drag, {passive: false});
    
    document.addEventListener('mouseup', endDrag);
    document.addEventListener('touchend', endDrag);
    
    // 🎭 ADD EMOTION DISPLAY TOGGLE BUTTON
    const toggleBtn = document.createElement('button');
    toggleBtn.id = 'emotionToggle';
    toggleBtn.innerHTML = '🎭';
    toggleBtn.style.cssText = `
      position: fixed;
      top: 20px;
      right: 330px;
      width: 40px;
      height: 40px;
      background: rgba(0, 0, 0, 0.8);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      color: white;
      font-size: 18px;
      cursor: pointer;
      z-index: 999999;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    `;
    
    toggleBtn.addEventListener('click', () => {
      const display = document.getElementById('emotionDisplay');
      if (display) {
        const isVisible = display.style.display !== 'none';
        display.style.display = isVisible ? 'none' : 'block';
        toggleBtn.innerHTML = isVisible ? '🎭' : '❌';
        toggleBtn.style.transform = isVisible ? 'scale(1)' : 'scale(1.1)';
        // 🎯 Remember user preference
        isEmotionDisplayManuallyHidden = isVisible;
        console.log('🎭 Emotion display toggled:', isVisible ? 'HIDDEN' : 'VISIBLE');
      }
    });
    
    document.body.appendChild(orb);
    document.body.appendChild(toggleBtn);
    console.log('🧠 Neural YOU orb created with BIDIRECTIONAL CONTROL + 24-EMOTION DISPLAY');
    return orb;
  }
  
  // ===== BIDIRECTIONAL CONSCIOUSNESS CONTROL FUNCTIONS ===== //
  function startDrag(e) {
    e.preventDefault();
    isDragging = true;
    
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    dragStartPos.x = clientX;
    dragStartPos.y = clientY;
    
    const orb = document.getElementById('youOrb');
    orb.style.cursor = 'grabbing';
    orb.style.transform = 'scale(1.1)';
    
    console.log('🎯 CONSCIOUSNESS TRAINING - Drag started');
  }
  
  function drag(e) {
    if (!isDragging) return;
    e.preventDefault();
    
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    const center = getCenterPosition();
    const deltaX = clientX - dragStartPos.x;
    const deltaY = clientY - dragStartPos.y;
    
    // Convert drag position to consciousness coordinates (-1 to 1)
    const maxRange = 800; // Max pixels from center - EXTREME MOVEMENT for dramatic visibility
    intentionOverride.x = Math.max(-1, Math.min(1, deltaX / maxRange));
    intentionOverride.y = Math.max(-1, Math.min(1, deltaY / maxRange));
    intentionOverride.active = true;
    intentionOverride.intensity = Math.min(1, Math.hypot(deltaX, deltaY) / maxRange);
    
    // Update orb position immediately for responsive feedback
    const orb = document.getElementById('youOrb');
    orb.style.left = (center.x + deltaX) + 'px';
    orb.style.top = (center.y + deltaY) + 'px';
    
    console.log(`🎯 INTENTION OVERRIDE: x=${intentionOverride.x.toFixed(2)}, y=${intentionOverride.y.toFixed(2)}, intensity=${(intentionOverride.intensity*100).toFixed(1)}%`);
  }
  
  function endDrag(e) {
    if (!isDragging) return;
    
    isDragging = false;
    const orb = document.getElementById('youOrb');
    orb.style.cursor = 'grab';
    orb.style.transform = 'scale(1)';
    
    if (intentionOverride.intensity > 0.1) {
      // Send consciousness intention to backend for neural training
      sendConsciousnessIntention(intentionOverride);
      
      // Keep intention active for a few seconds to "hold" the position
      setTimeout(() => {
        intentionOverride.active = false;
        console.log('🎯 INTENTION TRAINING - Override released');
      }, 3000);
    } else {
      intentionOverride.active = false;
    }
    
    console.log('🎯 CONSCIOUSNESS TRAINING - Drag ended, intention recorded');
  }
  
  function sendConsciousnessIntention(intention) {
    // Map consciousness coordinates to trait intentions
    const traitIntention = {
      // X-axis: Empathy (left) ↔ Analytical/Focus (right)  
      empathy_boost: intention.x < 0 ? Math.abs(intention.x) * intention.intensity * 50 : 0,
      focus_boost: intention.x > 0 ? intention.x * intention.intensity * 50 : 0,
      // Y-axis: Θετικά Συναισθήματα (up) ↔ Άγχος/Στρες (down)
      happiness_boost: intention.y < 0 ? Math.abs(intention.y) * intention.intensity * 50 : 0,
      stress_reduction: intention.y > 0 ? intention.y * intention.intensity * 50 : 0,
      timestamp: Date.now()
    };
    
    fetch('/api/consciousness/intention', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(traitIntention)
    }).then(response => response.json())
      .then(data => console.log('🎯 INTENTION SENT:', data))
      .catch(err => console.log('🚫 Intention send failed:', err));
  }

  function getCenterPosition() {
    // 🎯 DYNAMIC CENTER: Calculate actual window center for proper orb positioning
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    
    const center = {
      x: Math.floor(windowWidth / 2),   // True window center X
      y: Math.floor(windowHeight / 2)   // True window center Y
    };
    
    // 🔍 DEBUG: Log center position for verification  
    console.log('🎯 DYNAMIC CENTER POSITION:', {
      window: `${windowWidth}x${windowHeight}`,
      center: `(${center.x}, ${center.y})`
    });
    
    return center;
  }
  
  // ===== NEURAL MOTION UPDATE ===== //
  function updateYouPosition() {
    const orb = createYouOrb();
    const center = getCenterPosition();
    
    if (!isLiveSignal) {
      // NO SIGNAL: INSTANT DEATH STATE 💀
      if (youEngine) youEngine.setFreeze(true);
      
      // 🎯 ΣΈΒΟΝΤΑΙ ΤΑ BACKEND COORDINATES ακόμα και στο death state
      // Αν υπάρχουν backend coordinates, τις χρησιμοποιούμε
      let deathX = center.x;
      let deathY = center.y;
      
      if (window.latestLiveData && 
          window.latestLiveData.consciousness && 
          window.latestLiveData.consciousness.x !== undefined && 
          window.latestLiveData.consciousness.y !== undefined) {
        deathX = window.latestLiveData.consciousness.x;
        deathY = window.latestLiveData.consciousness.y;
        console.log('💀 DEATH STATE με backend coordinates:', deathX, deathY);
      }
      
      orb.style.left = deathX + 'px';
      orb.style.top = deathY + 'px';
      orb.style.transform = 'scale(0.8)';  // Σμίκρυνση
      orb.style.opacity = '0.3';           // Πιο αχνό
      orb.style.filter = 'grayscale(1.0) blur(1px)'; // Πλήρες γκρι + blur
      orb.style.boxShadow = '0 0 5px rgba(255,0,0,0.2)'; // Κόκκινη αχνή λάμψη
      
      console.log(`💀 YOU ORB DEATH - SIGNAL LOST at: (${center.x}, ${center.y})`);
      return;
    }
    
    // LIVE SIGNAL: Neural consciousness mapping
    if (!youEngine) {
      youEngine = new NeuralConsciousnessEngine();
    }
    
    youEngine.setFreeze(false);
    youEngine.update(window.latestBiometricData);
    youEngine.tick();
    
    const effects = youEngine.getEffects();
    
    // Apply neural position with intention override
    let finalX = center.x + (youEngine.currentX || 0);
    let finalY = center.y + (youEngine.currentY || 0);
    
    // 🎯 BIDIRECTIONAL OVERRIDE - User intention takes precedence
    if (intentionOverride.active && !isDragging) {
      const intentionX = intentionOverride.x * 400 * intentionOverride.intensity;
      const intentionY = intentionOverride.y * 400 * intentionOverride.intensity;
      
      // Blend intention with neural data (stronger intention = more override)
      const blendFactor = intentionOverride.intensity * 0.7;
      finalX = center.x + (intentionX * blendFactor + (youEngine.currentX || 0) * (1 - blendFactor));
      finalY = center.y + (intentionY * blendFactor + (youEngine.currentY || 0) * (1 - blendFactor));
      
      console.log(`🎯 INTENTION BLEND: ${(blendFactor*100).toFixed(1)}% intention, ${((1-blendFactor)*100).toFixed(1)}% neural`);
    }
    
    orb.style.left = finalX + 'px';
    orb.style.top = finalY + 'px';
    orb.style.transform = `scale(${effects.scale}) ${effects.transform || ''}`;
    orb.style.opacity = effects.opacity;
    orb.style.boxShadow = effects.boxShadow;
    orb.style.filter = effects.filter;
    orb.style.background = effects.background || orb.style.background;
    orb.style.display = 'flex';
    orb.style.visibility = 'visible';
    
    // 🎯 ENHANCED DEBUG: Full rendering chain visibility
    console.log(`🧠 NEURAL CONSCIOUSNESS RENDERING:`, {
      center: `(${center.x}, ${center.y})`,
      engine_current: `(${(youEngine.currentX || 0).toFixed(1)}, ${(youEngine.currentY || 0).toFixed(1)})`,
      final_position: `(${finalX.toFixed(1)}, ${finalY.toFixed(1)})`,
      dom_style: `left:${finalX}px, top:${finalY}px`,
      coherence: (window.latestBiometricData?.coherence || 0).toFixed(1) + '%'
    });
  }
  
  // ===== MAIN LOOP ===== //
  function mainLoop() {
    fetchLiveData();
    fetchEmotionsData(); // 🎯 CRITICAL: Fetch emotions for backend coordinates!
    updateYouPosition();
    setTimeout(mainLoop, UPDATE_RATE_MS);
  }
  
  // ===== INITIALIZATION ===== //
  document.addEventListener('DOMContentLoaded', function () {
    // 🧹 CLEANUP: Remove ALL existing YOU orbs and user-state elements
    const existingOrbs = document.querySelectorAll('#youOrb, .user-state, #userState, #center1, #center2');
    existingOrbs.forEach(el => {
      if (el) {
        el.remove();
        console.log('🧹 Removed duplicate orb/element:', el.id || el.className);
      }
    });
    
    // Force hide via CSS for any remaining elements
    const style = document.createElement('style');
    style.textContent = `
      #userState, .user-state:not(.neural-consciousness), #center1, #center2 { 
        display: none !important; 
        visibility: hidden !important;
      }
    `;
    document.head.appendChild(style);
    
    console.log('🧠 NEURAL CONSCIOUSNESS MAPPING SYSTEM - INITIALIZING...');
    createYouOrb();
    
    // Start main loop
    setTimeout(() => {
      mainLoop();
      console.log('🚀 NEURAL CONSCIOUSNESS ENGINE - FULLY OPERATIONAL!');
    }, 1000);
  });
  
  // Handle window resize
  window.addEventListener('resize', () => {
    if (!isLiveSignal) {
      updateYouPosition();
    }
  });
  
})();