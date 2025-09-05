// ΠΟΛΥΔΙΑΣΤΑΤΟ YOU MOTION ENGINE - Neural Consciousness Mapping
// Φτιαγμένο βάσει συνομιλίας με ChatGPT για neurochemical mapping

console.log("🧠 NEURAL CONSCIOUSNESS MAPPING ENGINE - Loading...");

// ===== HELPER FUNCTIONS ===== //
function ema(prev, x, a=0.15){ return prev + a*(x-prev); }
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
const tanh=x=>Math.tanh(x);
const sigmoid=x=>1/(1+Math.exp(-x));

// ===== RUNNING STATISTICS FOR BASELINE ===== //
class RunningStats{
  constructor(){this.n=0;this.mean=0;this.M2=0;}
  add(x){this.n++; const d=x-this.mean; this.mean+=d/this.n; this.M2+=d*(x-this.mean);}
  std(){return this.n>1?Math.sqrt(this.M2/(this.n-1)):1;}
}

// ===== BASELINE NORMALIZATION SYSTEM ===== //
class Baseline {
  constructor(keys){ 
    this.keys=keys; 
    this.stats=Object.fromEntries(keys.map(k=>[k,new RunningStats()])); 
    this.ready=false; 
    this.samples=0; 
  }
  push(f){
    this.keys.forEach(k=>{ const v=f[k]; if(Number.isFinite(v)) this.stats[k].add(v); });
    this.samples++; 
    if(this.samples>=240) this.ready=true; // ~8s at 30fps
    console.log(`🧠 BASELINE LEARNING: ${this.samples}/240 samples (${this.ready ? 'READY' : 'learning...'})`);
  }
  z(f){
    const out={};
    this.keys.forEach(k=>{
      const s=this.stats[k]; 
      const std=Math.max(1e-6, s.std());
      out[k]=Number.isFinite(f[k])? (f[k]-s.mean)/std : 0;
    });
    return out;
  }
}

// ===== FEATURE EXTRACTION ===== //
function extractFeatures(data){
  const d = data || window.latestBiometricData || {};
  
  // Βασικές μετρικές από backend
  const features = {
    // Soul metrics
    coherence: d?.coherence ?? 0,
    vitality: d?.vitality ?? 0,
    ethics: d?.ethics ?? 0,
    narrative: d?.narrative ?? 0,
    // HRV metrics
    hrv_ms: d?.rmssd ?? 0,
    breath_rpm: d?.breath ?? 0,
    heart_rate: d?.heartRate ?? 0,
    // Synthetic evolution metrics (derived from base metrics)
    empathy: (d?.coherence ?? 0) * 0.6 + (d?.ethics ?? 0) * 0.4,
    creativity: (d?.vitality ?? 0) * 0.7 + (d?.narrative ?? 0) * 0.3,
    resilience: (d?.coherence ?? 0) * 0.5 + (d?.rmssd ?? 0) * 0.001,
    focus: (d?.coherence ?? 0) * 0.8 + (d?.breath ?? 0 < 15 ? 1 : -0.5),
    curiosity: (d?.vitality ?? 0) * 0.5 + Math.random() * 10 - 5,
    compassion: (d?.ethics ?? 0) * 0.7 + (d?.coherence ?? 0) * 0.3
  };
  
  console.log('🧠 FEATURE EXTRACTION:', {
    raw_input: d,
    extracted_features: features,
    baseline_ready: window.youEngine?.base?.ready
  });
  
  return features;
}

// ===== MULTIDIMENSIONAL YOU MOTION ENGINE ===== //
class YouMotionEngine{
  constructor({radius=160, ease=0.12}={}){
    this.keys=["coherence","vitality","ethics","narrative","empathy","creativity","resilience","focus","curiosity","compassion","hrv_ms","breath_rpm","heart_rate"];
    this.base=new Baseline(this.keys);
    this.radius=radius; 
    this.ease=ease;
    this.pos={x:0,y:0}; 
    this.target={x:0,y:0}; 
    this.freeze=true;
    this.arousalGain=0.0;
    this.jitterLevel=0.0;
    this.breathPhase=0.0;
    
    // MLP weights for neural mapping (16 hidden neurons)
    // Βασισμένο στη συνομιλία - neurochemical mapping
    this.W1 = [
      {coherence:+0.3, vitality:+0.2, ethics:+0.4, narrative:+0.1, focus:+0.5, resilience:+0.5, compassion:-0.3, hrv_ms:+0.1, breath_rpm:-0.2},
      {creativity:+0.6, curiosity:+0.6, vitality:+0.3, empathy:+0.2, focus:-0.2},
      {empathy:+0.4, compassion:+0.4, coherence:+0.2, resilience:+0.1},
      {resilience:+0.6, ethics:+0.3, focus:+0.3, breath_rpm:-0.2},
      {curiosity:+0.5, creativity:+0.4, vitality:+0.3, hrv_ms:+0.2},
      {focus:+0.6, ethics:+0.3, coherence:+0.3, compassion:-0.2},
      {vitality:+0.5, hrv_ms:+0.3, breath_rpm:+0.2},
      {narrative:+0.4, curiosity:+0.3, creativity:+0.3},
      {hrv_ms:+0.6, coherence:+0.4, breath_rpm:-0.3},
      {empathy:+0.3, creativity:+0.2, resilience:+0.2, focus:+0.2},
      {ethics:+0.5, compassion:-0.2, curiosity:+0.2},
      {hrv_ms:+0.4, vitality:+0.2, breath_rpm:-0.3},
      {coherence:+0.5, focus:+0.3, resilience:+0.2},
      {curiosity:+0.5, empathy:+0.2, compassion:+0.2},
      {vitality:+0.6, narrative:+0.3, creativity:+0.2},
      {ethics:+0.6, resilience:+0.4, coherence:+0.3}
    ];
    this.W2x=[+0.6,-0.2,+0.4,+0.5,-0.1,+0.6,+0.2,-0.1,+0.4,+0.3,+0.5,+0.2,+0.4,-0.1,+0.1,+0.6];
    this.W2y=[-0.1,+0.7,+0.2,-0.2,+0.6,-0.1,+0.1,+0.4,+0.3,+0.2,-0.2,+0.1,+0.2,+0.5,+0.6,+0.1];
    
    console.log('🧠 YOU MOTION ENGINE INITIALIZED:', {
      radius: this.radius,
      ease: this.ease,
      features: this.keys.length,
      neural_hidden: this.W1.length
    });
  }
  
  setFreeze(flag){ 
    this.freeze=flag; 
    console.log(`🧠 YOU ENGINE STATE: ${flag ? '🔒 FROZEN' : '🔴 LIVE PROCESSING'}`);
  }
  
  // Update with live biometric data
  update(raw){
    const f = extractFeatures(raw);
    
    if(!this.base.ready){ 
      this.base.push(f); 
      console.log(`🧠 LEARNING BASELINE: ${this.base.samples}/240 samples`);
      return; 
    }
    
    const z = this.base.z(f); // Z-scores around baseline (0 = baseline)
    console.log('🧠 Z-SCORE NORMALIZATION:', {
      input_coherence: f.coherence,
      z_coherence: z.coherence?.toFixed(2),
      input_hrv: f.hrv_ms,
      z_hrv: z.hrv_ms?.toFixed(2)
    });
    
    // ---- MLP NEURAL NETWORK (16 hidden neurons) ----
    const h = new Array(16).fill(0).map((_,i)=>{
      let s = 0; 
      for(const [k,w] of Object.entries(this.W1[i])) {
        s += w * (z[k] ?? 0);
      }
      return Math.max(0, s); // ReLU activation
    });
    
    let x = 0, y = 0; 
    for(let i = 0; i < 16; i++){ 
      x += this.W2x[i] * h[i]; 
      y += this.W2y[i] * h[i]; 
    }
    x = tanh(x); // -1 to +1
    y = tanh(y); // -1 to +1
    
    console.log('🧠 NEURAL NETWORK OUTPUT:', {
      hidden_activations: h.slice(0,4).map(v => v.toFixed(2)), 
      raw_x: x.toFixed(3), 
      raw_y: y.toFixed(3)
    });
    
    // AROUSAL από φυσιολογικά δεδομένα (επηρεάζει την ακτίνα)
    const arousal = sigmoid( 
      0.5 * (z.vitality ?? 0) + 
      0.4 * (z.hrv_ms ?? 0) - 
      0.4 * Math.abs(z.breath_rpm ?? 0) + 
      0.3 * (z.coherence ?? 0) 
    );
    this.arousalGain = ema(this.arousalGain, arousal, 0.08);
    
    // JITTER από stress (χαμηλό HRV, χαμηλό coherence)
    const stress = Math.max(0, 
      0.6 - (z.hrv_ms ?? 0) * 0.1 - (z.coherence ?? 0) * 0.1
    );
    this.jitterLevel = ema(this.jitterLevel, stress, 0.1);
    
    // ΥΠΟΛΟΓΙΣΜΟΣ TARGET POSITION
    const mag = Math.hypot(x, y) || 1e-6;
    const R = this.radius * (0.55 + 0.45 * this.arousalGain); // 55-100% της ακτίνας
    
    this.target.x = (x / mag) * R;
    this.target.y = (y / mag) * R;
    
    console.log('🧠 CONSCIOUSNESS STATE:', {
      arousal_level: (this.arousalGain * 100).toFixed(1) + '%',
      stress_jitter: (this.jitterLevel * 100).toFixed(1) + '%',
      movement_radius: R.toFixed(1) + 'px',
      target_position: `(${this.target.x.toFixed(1)}, ${this.target.y.toFixed(1)})`
    });
  }
  
  // Animation tick
  tick(){
    if(this.freeze) return;
    
    // Ομαλή κίνηση προς το target
    this.pos.x = ema(this.pos.x, this.target.x, this.ease);
    this.pos.y = ema(this.pos.y, this.target.y, this.ease);
    
    // Αναπνευστικός συγχρονισμός
    this.breathPhase += 0.02; // ~0.6 Hz breathing
    const breathOffset = Math.sin(this.breathPhase) * 3 * this.arousalGain;
    
    // Stress jitter
    const jitterX = (Math.random() - 0.5) * this.jitterLevel * 12;
    const jitterY = (Math.random() - 0.5) * this.jitterLevel * 12;
    
    // Τελική θέση με εφέ
    this.currentX = this.pos.x + breathOffset + jitterX;
    this.currentY = this.pos.y + breathOffset + jitterY;
    
    // Boundary safety
    const r = Math.hypot(this.currentX, this.currentY);
    if(r > this.radius){ 
      const k = this.radius / r; 
      this.currentX *= k; 
      this.currentY *= k; 
    }
  }
  
  // Get CSS transform
  getTransform(centerX, centerY){ 
    if (this.freeze || !this.currentX || !this.currentY) {
      return `translate3d(${centerX}px, ${centerY}px, 0)`;
    }
    return `translate3d(${(centerX + this.currentX).toFixed(1)}px, ${(centerY + this.currentY).toFixed(1)}px, 0)`; 
  }
  
  // Get visual effects based on consciousness state
  getEffects(){
    const intensity = 0.5 + 0.5 * this.arousalGain;
    const glowSize = 15 + 30 * intensity;
    const opacity = 0.7 + 0.3 * intensity;
    
    // HRV-based color mapping
    const r = Math.round(255 * (1 - this.arousalGain * 0.5));
    const g = Math.round(180 + 75 * this.arousalGain);
    const b = 255;
    
    return {
      opacity: opacity,
      boxShadow: `0 0 ${glowSize}px rgba(${r}, ${g}, ${b}, 0.8)`,
      filter: this.jitterLevel > 0.4 ? 'saturate(0.6) contrast(1.2)' : 'none',
      transform_scale: 1 + this.arousalGain * 0.2
    };
  }
}

// ===== GLOBAL INITIALIZATION ===== //
window.neuralMotionReady = false;

// Export για χρήση από άλλα scripts
window.YouMotionEngine = YouMotionEngine;
window.extractFeatures = extractFeatures;

console.log('🧠 NEURAL CONSCIOUSNESS MAPPING - Ready for integration!');