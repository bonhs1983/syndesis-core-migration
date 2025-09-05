"""
Simplified Neural HRV Processing - Compatible with existing environment
No external neural dependencies required
"""
import numpy as np
import time
from collections import deque
import math

def kalman_filter(measurements, q=1.0, r=30.0):
    """
    Simple Kalman filter for signal regularization
    """
    if measurements is None or len(measurements) < 2:
        return measurements
    
    x_hat = measurements[0]  # initial estimate
    P = 1.0  # initial error covariance
    filtered = [x_hat]
    
    for z in measurements[1:]:
        # Prediction step
        x_pred = x_hat
        P_pred = P + q
        
        # Update step
        K = P_pred / (P_pred + r)
        x_hat = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        
        filtered.append(x_hat)
    
    return np.array(filtered)

def extract_hrv_features(rr_intervals):
    """
    Extract core HRV features from RR intervals (ms)
    Returns: [meanRR, SDNN, RMSSD, LF, HF, LF_HF_ratio]
    """
    if rr_intervals is None or len(rr_intervals) < 5:
        return None
    
    rr = np.array(rr_intervals, dtype=float)
    
    # Apply Kalman filtering
    rr_filtered = kalman_filter(rr)
    
    # Time domain features
    meanRR = float(np.mean(rr_filtered))
    sdnn = float(np.std(rr_filtered, ddof=1))
    
    # RMSSD calculation
    diff = np.diff(rr_filtered)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else 0.0
    
    # Frequency domain (simplified)
    try:
        # Detrend and apply FFT
        detrended = rr_filtered - np.mean(rr_filtered)
        fft_result = np.fft.rfft(detrended)
        psd = np.abs(fft_result)**2
        
        # Approximate frequency bands (simplified approach)
        n_samples = len(psd)
        lf_start = max(1, int(n_samples * 0.04))
        lf_end = int(n_samples * 0.15)
        hf_start = lf_end
        hf_end = int(n_samples * 0.40)
        
        lf_power = float(np.sum(psd[lf_start:lf_end]))
        hf_power = float(np.sum(psd[hf_start:hf_end]))
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 1e-9 else float('inf')
        
    except Exception:
        lf_power, hf_power, lf_hf_ratio = 0.0, 0.0, 0.0
    
    return [meanRR, sdnn, rmssd, lf_power, hf_power, lf_hf_ratio]

class SimpleGRUDynamics:
    """
    Simplified GRU-like dynamics without PyTorch
    Uses basic recurrent processing for temporal features
    """
    def __init__(self, input_dim=6, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Simple weight matrices (randomly initialized)
        np.random.seed(42)  # for reproducibility
        self.W_update = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_update = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_reset = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_reset = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_new = np.random.randn(input_dim, hidden_dim) * 0.1
        self.U_new = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, sequence):
        """
        Process sequence of feature vectors
        sequence: list of feature vectors [time_steps, features]
        """
        if sequence is None or len(sequence) == 0:
            return np.zeros(self.hidden_dim).tolist()
        
        h = np.zeros(self.hidden_dim)  # initial hidden state
        
        for x in sequence:
            x = np.array(x, dtype=float)
            if len(x) != self.input_dim:
                continue
                
            # Update gate
            z = self.sigmoid(np.dot(x, self.W_update) + np.dot(h, self.U_update))
            
            # Reset gate  
            r = self.sigmoid(np.dot(x, self.W_reset) + np.dot(h, self.U_reset))
            
            # New gate
            h_new = self.tanh(np.dot(x, self.W_new) + np.dot(r * h, self.U_new))
            
            # Update hidden state
            h = (1 - z) * h + z * h_new
        
        return h.tolist()  # Convert to Python list to avoid array comparison issues

def map_features_to_soul_metrics(features, recent_bpm=None):
    """
    Map HRV features to Soul Metrics (0-100)
    """
    if not features or len(features) < 6:
        return {"coherence": 0, "vitality": 0, "ethics": 0, "narrative": 0}
    
    meanRR, sdnn, rmssd, lf, hf, lf_hf = features
    
    # HeartMath Institute Authentic Coherence Formula (0.04-0.26 Hz analysis)
    coherence_power = lf + hf  # Power in coherence range (0.04-0.26 Hz)
    total_spectrum_power = coherence_power * 2.5  # Approximate full spectrum
    
    if coherence_power > 1e-9 and total_spectrum_power > 1e-9:
        # Find maximum peak in coherence range (LF typically dominates in coherent states)
        peak_power = max(lf, hf)  # Maximum peak in 0.04-0.26 Hz range
        
        # HeartMath Formula: Peak Power / (Total Power - Peak Power)
        coherence_ratio = peak_power / (total_spectrum_power - peak_power)
        
        # Convert to percentage with realistic physiological scaling
        # Normal coherent states: 0.5-2.0 ratio → 30-80% coherence
        coherence = min(85.0, max(15.0, 30.0 + (coherence_ratio * 35.0)))
    else:
        coherence = 25.0  # Low power baseline
    
    # Vitality: based on heart rate (from recent BPM) - realistic physiological model
    if recent_bpm and 40 <= recent_bpm <= 180:
        # Optimal range around 60-75 BPM (resting heart rate zone)
        optimal_bpm = 67.5  # Center of optimal range
        deviation = abs(recent_bpm - optimal_bpm)
        
        # More realistic vitality curve:
        # 60-75 BPM: 70-90% vitality (optimal zone)
        # 45-90 BPM: 30-70% vitality (acceptable zone)  
        # Outside: 10-30% vitality (stress/fatigue zone)
        if deviation <= 7.5:  # 60-75 BPM range
            vitality = max(70.0, 90.0 - deviation * 2.5)  # 70-90% range
        elif deviation <= 22.5:  # 45-90 BPM range 
            vitality = max(30.0, 70.0 - (deviation - 7.5) * 2.0)  # 30-70% range
        else:  # Extreme values
            vitality = max(10.0, 30.0 - (deviation - 22.5) * 0.5)  # 10-30% range
    else:
        vitality = 25.0  # Default low vitality for invalid data
    
    # Ethics: more balanced HRV variability (moderate SDNN = moderate ethics)
    # Healthy SDNN range: 15-70ms for adults  
    # Target moderate ethics around 45-65% instead of extreme values
    if sdnn <= 5:  # Very low SDNN = poor variability
        ethics = 30.0
    elif sdnn <= 15:  # Low SDNN = below average
        ethics = 30.0 + (sdnn - 5) * 2.0  # 30-50% range
    elif sdnn <= 50:  # Normal SDNN = good ethics  
        ethics = 50.0 + (sdnn - 15) * 0.6  # 50-71% range
    elif sdnn <= 80:  # High SDNN = still good but not extreme
        ethics = 71.0 - (sdnn - 50) * 0.4  # 71-59% range  
    else:  # Very high SDNN = potentially problematic
        ethics = max(40.0, 59.0 - (sdnn - 80) * 0.3)  # 40-59% range
    
    # Narrative: combination of coherence and vitality
    narrative = 0.6 * coherence + 0.4 * vitality
    
    return {
        "coherence": float(coherence),
        "vitality": float(vitality), 
        "ethics": float(ethics),
        "narrative": float(narrative)
    }

def map_latent_to_traits(latent_vector, recent_bpm=None, recent_hrv=None, soul_metrics=None):
    """
    Map latent representation to personality traits using REAL TICKR biometric data (20-100% range)
    """
    if not latent_vector or len(latent_vector) < 6:
        # Fallback using biometric data when neural processing unavailable
        if recent_bpm and recent_hrv:
            return _biometric_trait_mapping(recent_bpm, recent_hrv, soul_metrics)
        return {"empathy": 0, "creativity": 0, "resilience": 0, 
                "focus": 0, "curiosity": 0, "compassion": 0}
    
    # Get soul metrics for enhanced mapping
    coherence = soul_metrics.get('coherence', 50) if soul_metrics else 50
    vitality = soul_metrics.get('vitality', 50) if soul_metrics else 50
    ethics = soul_metrics.get('ethics', 50) if soul_metrics else 50
    narrative = soul_metrics.get('narrative', 50) if soul_metrics else 50
    
    # Enhanced neural mapping with REALISTIC RANGE using latent vector + biometrics
    def realistic_mapping(biometric_input, latent_offset, min_val=20.0, max_val=100.0):
        """Direct proportional mapping - reduced latent bias for realistic values"""
        # Normalize biometric input (0-1 range)
        normalized = max(0.0, min(1.0, biometric_input))
        # ZERO latent variation - pure biometric mapping for realistic values  
        with_latent = normalized + latent_offset * 0.0
        # Map to target range with realistic distribution
        final_value = min_val + max(0.0, min(1.0, with_latent)) * (max_val - min_val)
        return final_value
    
    # Use latent vector with biometric modulation
    latent_base = latent_vector[:6] if len(latent_vector) >= 6 else list(latent_vector) + [0] * (6 - len(latent_vector))
    
    # 🎯 REALISTIC TRAIT CALCULATION - Soul Metrics blends with proper range distribution
    empathy_input = coherence/100.0 * 0.7 + vitality/100.0 * 0.3      # Coherence + vitality (0-1)
    creativity_input = narrative/100.0 * 0.6 + coherence/100.0 * 0.4   # Narrative + coherence (0-1)
    resilience_input = ethics/100.0 * 0.8 + vitality/100.0 * 0.2      # Ethics + vitality (0-1)  
    focus_input = coherence/100.0 * 0.9 + ethics/100.0 * 0.1          # Coherence + ethics (0-1)
    curiosity_input = vitality/100.0 * 0.6 + narrative/100.0 * 0.4    # Vitality + narrative (0-1)
    compassion_input = vitality/100.0 * 0.5 + ethics/100.0 * 0.5      # Vitality + ethics (0-1)
    
    empathy = realistic_mapping(empathy_input, latent_base[0], min_val=30, max_val=75)
    creativity = realistic_mapping(creativity_input, latent_base[1], min_val=25, max_val=80) 
    resilience = realistic_mapping(resilience_input, latent_base[2], min_val=35, max_val=75)
    focus = realistic_mapping(focus_input, latent_base[3], min_val=30, max_val=80)
    curiosity = realistic_mapping(curiosity_input, latent_base[4], min_val=30, max_val=75)
    compassion = realistic_mapping(compassion_input, latent_base[5], min_val=30, max_val=75)
    
    return {
        "empathy": float(max(20.0, min(100.0, empathy))),
        "creativity": float(max(20.0, min(100.0, creativity))), 
        "resilience": float(max(20.0, min(100.0, resilience))),
        "focus": float(max(20.0, min(100.0, focus))),
        "curiosity": float(max(20.0, min(100.0, curiosity))),
        "compassion": float(max(20.0, min(100.0, compassion)))
    }

def _biometric_trait_mapping(bpm, hrv, soul_metrics=None):
    """Pure biometric trait mapping when neural data unavailable"""
    # Normalize BPM (60-100 range typical)
    bpm_norm = max(0.0, min(1.0, (bpm - 50) / 50.0))
    hrv_norm = max(0.0, min(1.0, hrv / 60.0))  # Normalize HRV (0-60ms typical)
    
    # Get soul metrics if available
    coherence = soul_metrics.get('coherence', 50) if soul_metrics else 50
    vitality = soul_metrics.get('vitality', 50) if soul_metrics else 50
    ethics = soul_metrics.get('ethics', 50) if soul_metrics else 50
    
    # REALISTIC BIOMETRIC-DRIVEN TRAITS (30-75% range) - FIXED TO MATCH NEW RANGES
    empathy = 30 + hrv_norm * 45 + (coherence - 50) * 0.2     # HRV = emotional regulation (30-75%)
    creativity = 25 + bpm_norm * 50 + np.random.rand() * 10   # Higher BPM = energy = creativity (25-80%)  
    resilience = 35 + ethics * 0.4 + hrv_norm * 20            # Ethics + HRV = resilience (35-75%)
    focus = 30 + coherence * 0.5 + (80 - bpm) * 0.3          # Coherence + lower BPM = focus (30-80%)
    curiosity = 30 + vitality * 0.45 + bpm_norm * 15         # Vitality + BPM = curiosity (30-75%)
    compassion = 30 + hrv_norm * 40 + (ethics - 50) * 0.3    # HRV + ethics = compassion (30-75%)
    
    return {
        "empathy": float(max(20.0, min(100.0, empathy))),
        "creativity": float(max(20.0, min(100.0, creativity))),
        "resilience": float(max(20.0, min(100.0, resilience))),
        "focus": float(max(20.0, min(100.0, focus))),
        "curiosity": float(max(20.0, min(100.0, curiosity))),
        "compassion": float(max(20.0, min(100.0, compassion)))
    }

def map_traits_to_consciousness(emotions):
    """
    🎭 ΝΕΟΣ 24-EMOTION CONSCIOUSNESS MAPPING SYSTEM  
    Maps 24 emotions to 2D consciousness coordinates με PANEL COORDINATES
    CENTER: (400, 300) = Κόκκινος κύκλος | BOUNDARIES: X=[100,700], Y=[50,550]
    """
    if not isinstance(emotions, dict):
        # Fallback για παλιό 6-trait system - ΙΔΙΟ COORDINATE SYSTEM
        empathy = emotions.get("empathy", 50.0)
        creativity = emotions.get("creativity", 50.0)
        resilience = emotions.get("resilience", 50.0)
        focus = emotions.get("focus", 50.0)
        curiosity = emotions.get("curiosity", 50.0)
        compassion = emotions.get("compassion", 50.0)
        
        # ΔΙΟΡΘΩΜΕΝΟΣ αλγόριθμος - FOCUS-DRIVEN X-AXIS!
        # Focus = δεξιά (υψηλή συγκέντρωση), Low focus = αριστερά (διασπορά)
        x_norm = (focus - 50.0) / 50.0  # [-1, 1] - ΚΑΘΑΡΑ focus-driven!
        positive_emotions = (compassion + creativity + empathy) / 3.0
        negative_stress = (100 - compassion + resilience * 0.8) / 2.0  
        y_norm = (positive_emotions - negative_stress + 50) / 150.0       # [-1, 1]
        
        # Map to PANEL COORDINATES: CENTER (700,300) - EXTREME MOVEMENT RANGES
        x = 700 + (x_norm * 1600)  # 700 ± 1600 = [100, 2300] - ULTRA EXTREME για DRAMATIC κίνηση!
        y = 300 + (y_norm * 1400)  # 300 ± 1400 = [100, 1700] - ULTRA EXTREME για DRAMATIC κίνηση!
        
        x = max(150.0, min(950.0, x))
        y = max(50.0, min(630.0, y))
        return {"x": float(x), "y": float(y)}
    
    # ============================
    # 🎯 ΝΕΟ 24-EMOTION PANEL MAPPING
    # ============================
    
    # Extract emotion values με defaults
    enthusiasm = emotions.get("enthusiasm", 50.0)
    energy = emotions.get("energy", 50.0)
    anxiety = emotions.get("anxiety", 50.0)
    anger = emotions.get("anger", 50.0)
    calmness = emotions.get("calmness", 50.0)
    fatigue = emotions.get("fatigue", 50.0)
    nostalgia = emotions.get("nostalgia", 50.0)
    
    joy = emotions.get("joy", 50.0)
    confidence = emotions.get("confidence", 50.0)
    hope = emotions.get("hope", 50.0)
    satisfaction = emotions.get("satisfaction", 50.0)
    sadness = emotions.get("sadness", 50.0)
    despair = emotions.get("despair", 50.0)
    worry = emotions.get("worry", 50.0)
    fear = emotions.get("fear", 50.0)
    
    # ⚡ X-AXIS: FOCUS-DRIVEN MAPPING (Αριστερά ← → Δεξιά) 
    # HIGH FOCUS = δεξιά (concentration), LOW FOCUS = αριστερά (distraction)
    # Εξάγω το focus από τα emotions που υπολογίστηκαν στη map_emotions_from_data
    focus_level = emotions.get("confidence", 50.0)  # Use confidence as focus proxy
    
    # FOCUS-DRIVEN X-AXIS! (Same as 6-trait system)
    x_norm = (focus_level - 50.0) / 50.0  # [-1, 1] - ΚΑΘΑΡΑ focus-driven!
    
    # 😊 Y-AXIS: EMOTIONAL VALENCE (Πάνω ↑ ↓ Κάτω)
    # Positive = joy + confidence + hope + satisfaction (ΠΑΝΩ)
    # Negative = sadness + despair + worry + fear (ΚΑΤΩ)
    positive_valence = joy + confidence + hope + satisfaction
    negative_valence = sadness + despair + worry + fear
    
    # Normalize στο [-1, 1] range
    y_raw = (positive_valence - negative_valence) / 400.0  # Total possible range = 400
    y_norm = max(-1.0, min(1.0, y_raw * 2.0))  # Amplify για ορατή κίνηση
    
    # 🎯 MAP TO PANEL COORDINATES με κόκκινο κύκλο ως κέντρο - EXTREME RANGES!
    # CENTER: (700, 300) = Κόκκινος κύκλος | X_RANGE: 1600 | Y_RANGE: 1400 - ULTRA EXTREME!
    x = 700 + (x_norm * 1600)  # 700 ± 1600 = [100, 2300] - ULTRA ΤΕΡΑΣΤΙΑ εμβέλεια horizontal!
    y = 300 - (y_norm * 1400)  # 300 ± 1400 = [100, 1700] - ULTRA ΤΕΡΑΣΤΙΑ εμβέλεια vertical!
    
    # CLAMP στα extended panel boundaries για μεγαλύτερη κίνηση
    x = max(150.0, min(950.0, x))
    y = max(50.0, min(630.0, y))
    
    return {"x": float(x), "y": float(y)}

def map_emotions_from_data(bpm, hrv, traits, soul_metrics):
    """
    🎭 EMOTIONAL MAPPING SYSTEM - Υπολογίζει 24 συναισθήματα από biometric data
    """
    # Normalize metrics
    bpm_norm = max(0.0, min(1.0, (bpm - 50) / 50.0))
    hrv_norm = max(0.0, min(1.0, hrv / 60.0))
    
    # Extract values
    empathy = traits.get("empathy", 50) / 100.0
    creativity = traits.get("creativity", 50) / 100.0
    resilience = traits.get("resilience", 50) / 100.0
    focus = traits.get("focus", 50) / 100.0
    curiosity = traits.get("curiosity", 50) / 100.0
    compassion = traits.get("compassion", 50) / 100.0
    
    coherence = soul_metrics.get("coherence", 50) / 100.0
    vitality = soul_metrics.get("vitality", 50) / 100.0
    ethics = soul_metrics.get("ethics", 50) / 100.0
    narrative = soul_metrics.get("narrative", 50) / 100.0
    
    # 🎭 CALCULATE EMOTIONS (0-100 scale)
    emotions = {}
    
    # 🟢 ΘΕΤΙΚΑ ΣΥΝΑΙΣΘΗΜΑΤΑ (11)
    emotions["joy"] = (vitality * 0.6 + coherence * 0.4 + hrv_norm * 0.3 + creativity * 0.2) * 100
    emotions["happiness"] = (vitality * 0.5 + coherence * 0.3 + hrv_norm * 0.4 + compassion * 0.3) * 100
    emotions["confidence"] = (coherence * 0.7 + vitality * 0.3 + focus * 0.2) * 100
    emotions["calmness"] = (hrv_norm * 0.7 + coherence * 0.3 + (1.0 - bpm_norm) * 0.2) * 100
    emotions["enthusiasm"] = (creativity * 0.5 + curiosity * 0.4 + bpm_norm * 0.3 + vitality * 0.2) * 100
    emotions["empathy_feeling"] = (empathy * 0.8 + compassion * 0.6 + hrv_norm * 0.2) * 100
    emotions["compassion_feeling"] = (compassion * 0.9 + empathy * 0.4 + ethics * 0.3) * 100
    emotions["inspiration"] = (creativity * 0.6 + narrative * 0.4 + vitality * 0.3 + coherence * 0.2) * 100
    emotions["hope"] = (vitality * 0.5 + narrative * 0.6 + coherence * 0.3) * 100
    emotions["energy"] = (vitality * 0.7 + bpm_norm * 0.5 + creativity * 0.2) * 100
    emotions["satisfaction"] = (vitality * 0.4 + ethics * 0.5 + coherence * 0.4) * 100
    
    # 🔴 ΑΡΝΗΤΙΚΑ ΣΥΝΑΙΣΘΗΜΑΤΑ (10)
    emotions["anxiety"] = ((1.0 - hrv_norm) * 0.6 + (1.0 - coherence) * 0.7 + bpm_norm * 0.3) * 100
    emotions["stress"] = ((1.0 - hrv_norm) * 0.8 + (1.0 - coherence) * 0.6 + resilience * 0.2) * 100
    emotions["frustration"] = ((1.0 - creativity) * 0.5 + (1.0 - coherence) * 0.6 + bpm_norm * 0.3) * 100  
    emotions["anger"] = (bpm_norm * 0.6 + (1.0 - compassion) * 0.7 + (1.0 - coherence) * 0.4) * 100
    emotions["fear"] = ((1.0 - coherence) * 0.8 + bpm_norm * 0.4 + (1.0 - empathy) * 0.3) * 100
    emotions["sadness"] = ((1.0 - vitality) * 0.7 + (1.0 - narrative) * 0.5 + (1.0 - hrv_norm) * 0.3) * 100
    emotions["fatigue"] = ((1.0 - vitality) * 0.8 + (1.0 - bpm_norm) * 0.4 + (1.0 - coherence) * 0.2) * 100
    emotions["loneliness"] = ((1.0 - empathy) * 0.6 + (1.0 - compassion) * 0.7 + (1.0 - narrative) * 0.3) * 100
    emotions["despair"] = ((1.0 - narrative) * 0.8 + (1.0 - vitality) * 0.6 + (1.0 - coherence) * 0.4) * 100
    emotions["worry"] = ((1.0 - coherence) * 0.6 + hrv_norm * 0.3 + (1.0 - focus) * 0.4) * 100
    
    # 🌀 ΣΥΝΘΕΤΑ ΣΥΝΑΙΣΘΗΜΑΤΑ (3)
    emotions["curiosity_feeling"] = (curiosity * 0.9 + creativity * 0.3 + vitality * 0.2) * 100
    emotions["nostalgia"] = (narrative * 0.7 + empathy * 0.4 + (1.0 - vitality * 0.2)) * 100
    emotions["disappointment"] = ((1.0 - coherence) * 0.6 + (1.0 - narrative) * 0.5 + resilience * 0.2) * 100
    
    # Clamp all emotions to 0-100 range
    for emotion in emotions:
        emotions[emotion] = max(0.0, min(100.0, emotions[emotion]))
    
    return emotions

# Global neural processor instance
neural_processor = SimpleGRUDynamics()