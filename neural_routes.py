"""
Neural HRV Enhanced Routes - Compatible Implementation
Integrates neural processing without external ML dependencies
"""
from flask import render_template, request, jsonify, redirect, url_for, flash
import logging
from app import app, db
from models import InteractionLog, TrainingJob, PipelineStatus
from collections import deque
from datetime import datetime, timezone
import time
import math
import numpy as np

# Import our simplified neural modules
from neural_modules import (
    extract_hrv_features, map_features_to_soul_metrics, 
    map_latent_to_traits, map_traits_to_consciousness,
    neural_processor, map_emotions_from_data
)

# Import advanced neural HRV system
try:
    from advanced_neural_hrv_analyzer import (
        AdvancedNeuralHRVAnalyzer, GRUDynamics, NeuralODEDynamics, 
        MultimodalGenerator, HRVManifoldAnalyzer
    )
    ADVANCED_NEURAL_AVAILABLE = True
    logging.info("🧠 Advanced Neural HRV modules loaded successfully")
except ImportError as e:
    ADVANCED_NEURAL_AVAILABLE = False
    logging.warning(f"🧠 Advanced Neural HRV modules not available: {e}")

# Initialize advanced neural components
if ADVANCED_NEURAL_AVAILABLE:
    try:
        advanced_analyzer = AdvancedNeuralHRVAnalyzer()
        gru_dynamics = GRUDynamics(hrv_dim=6, latent_dim=32)
        neural_ode = NeuralODEDynamics(hrv_dim=6, latent_dim=32)
        multimodal_gen = MultimodalGenerator()
        manifold_analyzer = HRVManifoldAnalyzer()
        logging.info("🚀 Advanced Neural HRV components initialized successfully")
    except Exception as e:
        ADVANCED_NEURAL_AVAILABLE = False
        logging.error(f"🚫 Failed to initialize advanced components: {e}")
else:
    advanced_analyzer = None
    gru_dynamics = None
    neural_ode = None 
    multimodal_gen = None
    manifold_analyzer = None

# =========================
#   Runtime State
# =========================
# Buffer management
BUF_SECONDS = 60
RR_BUFFER = deque(maxlen=1024)  # RR intervals (ms)
BPM_BUFFER = deque(maxlen=2048)  # BPM timeline
FEAT_BUFFER = deque(maxlen=256)  # Feature vectors timeline

last_signal_ts = 0.0   # Watchdog timer
last_payload = None    # Anti-freeze cache

# Store autonomous responses for frontend polling (keep existing functionality)
autonomous_responses = deque(maxlen=50)
last_hrv_frame = None

# =========================
#   Utilities
# =========================
def clip01(x): 
    return max(0.0, min(100.0, float(x)))

def now_utc(): 
    return datetime.now(timezone.utc).isoformat()

def bpm_to_rr(bpm):
    """Convert BPM to RR interval (ms)"""
    if bpm is None or bpm <= 0: 
        return None
    return 60000.0 / float(bpm)

def features_sequence_tensor():
    """
    Get recent feature sequence for neural processing
    Returns last 10 feature vectors from buffer
    """
    T = min(10, len(FEAT_BUFFER))
    if T < 3:
        return None
    
    # Get recent features and normalize
    recent_features = list(FEAT_BUFFER)[-T:]
    sequence = np.array(recent_features, dtype=float)
    
    # Z-score normalization per feature
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True) + 1e-6
    normalized = (sequence - mean) / std
    
    return normalized.tolist()

# =========================
#   Neural HRV API Endpoints
# =========================
def register_neural_routes(app):
    """Register neural processing routes to Flask app"""
    
    @app.route("/neural/ingest-hr", methods=["POST"])
    def neural_ingest_hr():
        """
        Neural HRV data ingestion endpoint
        Accepts: {"hr_bpm": 78, "rr": [800, 790, 810, ...], "ts": 1720000000}
        """
        global last_signal_ts, last_payload
        from flask import request

        data = request.get_json(force=True, silent=True) or {}
        ts = float(data.get("ts", time.time()))
        bpm = data.get("hr_bpm", data.get("bpm", None))  # Accept both hr_bpm and bpm
        rr_list = data.get("rr", None)

        # Update watchdog
        last_signal_ts = ts

        # Process RR intervals
        rr_from_bpm = bpm_to_rr(bpm) if bpm is not None else None

        if rr_list and isinstance(rr_list, (list, tuple)) and len(rr_list) >= 1:
            for v in rr_list:
                try: 
                    RR_BUFFER.append(float(v))
                except: 
                    pass
        elif rr_from_bpm:
            RR_BUFFER.append(rr_from_bpm)

        if bpm is not None:
            try: 
                BPM_BUFFER.append(float(bpm))
            except: 
                pass

        # Extract features and update buffer
        if len(RR_BUFFER) >= 5:
            features = extract_hrv_features(list(RR_BUFFER)[-300:])  # Use last 300 RR intervals
            if features is not None:
                FEAT_BUFFER.append(features)
                logging.debug(f"🧠 Neural features extracted: {[f'{f:.2f}' for f in features]}")

        return jsonify({"accepted": True, "ts": now_utc()})

    @app.route("/neural/analysis", methods=["GET"])
    def neural_analysis():
        """
        Enhanced live data endpoint with Neural HRV processing
        Returns: Soul Metrics + Traits + Consciousness mapping + HRV panel
        """
        global last_payload
        has_live = (time.time() - last_signal_ts) <= 3.0

        if len(FEAT_BUFFER) >= 3:
            try:
                # 🧠 ADVANCED NEURAL PROCESSING ENGINE  
                if ADVANCED_NEURAL_AVAILABLE and len(FEAT_BUFFER) >= 5:
                    # Use advanced neural analyzer with comprehensive analysis
                    rr_data = list(RR_BUFFER)[-50:]  # Last 50 RR intervals for comprehensive analysis
                    
                    # Advanced neural analysis with comprehensive techniques
                    advanced_result = advanced_analyzer.comprehensive_analysis(rr_data, context={"enable_multimodal": True})
                    
                    # Extract advanced neural features from comprehensive analysis
                    neural_analysis = advanced_result.get("neural_analysis", {})
                    latent = neural_analysis.get("gru_embedding", [])
                    manifold_analysis = advanced_result.get("manifold_analysis", {})
                    manifold_coords = manifold_analysis.get("coordinates", {"x": 0, "y": 0})
                    multimodal_feedback = advanced_result.get("multimodal_feedback", {})
                    risk_zones = manifold_analysis.get("risk_zones", {})
                    
                    logging.info(f"🚀 ADVANCED NEURAL: VAE+ODE+UMAP analysis complete - latent_dim={len(latent)}")
                    logging.debug(f"🧬 Manifold coords: {manifold_coords}")
                    
                    # Log advanced features
                    if "vae_latent" in advanced_result:
                        vae_features = advanced_result["vae_latent"]
                        logging.debug(f"🧬 VAE representation learning: {len(vae_features)} features")
                    
                    if "ode_trajectory" in advanced_result:
                        ode_path = advanced_result["ode_trajectory"]
                        logging.debug(f"🌊 Neural ODE dynamics: {len(ode_path)} trajectory points")
                        
                    if "umap_embedding" in advanced_result:
                        umap_coords = advanced_result["umap_embedding"]
                        logging.debug(f"🗺️ UMAP manifold embedding: {umap_coords}")
                        
                else:
                    # Fallback to simplified neural processing
                    sequence = features_sequence_tensor()
                    
                    if sequence is not None:
                        # Neural processing: features → latent representation
                        latent = neural_processor.forward(sequence)
                        logging.debug(f"🧠 Neural latent vector: shape={len(latent)}")
                    else:
                        latent = neural_processor.forward(FEAT_BUFFER[-1])
                        
                    manifold_coords = {"x": 0, "y": 0}
                    multimodal_feedback = {}
                    risk_zones = {}
                    
                # Map features to Soul Metrics
                bpm_recent = BPM_BUFFER[-1] if len(BPM_BUFFER) else None
                soul_metrics = map_features_to_soul_metrics(FEAT_BUFFER[-1], bpm_recent)
                
                # Map latent to personality traits (enhanced with manifold data)
                traits = map_latent_to_traits(latent, recent_bpm=BPM_BUFFER[-1] if len(BPM_BUFFER) > 0 else None, 
                                                  recent_hrv=RR_BUFFER[-1] if len(RR_BUFFER) > 0 else None, 
                                                  soul_metrics=soul_metrics)
                
                logging.debug(f"🧠 Neural features forwarded: Evolution traits={traits}")
                
                # Calculate 24 emotions from biometric data
                current_bpm = BPM_BUFFER[-1] if len(BPM_BUFFER) > 0 else 75.0
                current_hrv = RR_BUFFER[-1] if len(RR_BUFFER) > 0 else 50.0
                emotions = map_emotions_from_data(current_bpm, current_hrv, traits, soul_metrics)
                
                # Map emotions to consciousness coordinates
                consciousness = map_traits_to_consciousness(emotions)
                
                # HRV panel summary with advanced neural feedback
                rr_ms = list(RR_BUFFER)[-30:] if len(RR_BUFFER) else []
                rmssd = 0.0
                if len(rr_ms) >= 5:
                    d = np.diff(np.array(rr_ms))
                    rmssd = float(np.sqrt(np.mean(d**2)))
                
                # Calculate values safely to avoid array comparison errors
                hrv_value = float((soul_metrics["coherence"] + soul_metrics["ethics"]) / 2.0)
                breath_value = float(12.0 + 6.0 * np.random.rand())
                
                # Enhanced HRV panel with neural feedback
                hrv_panel = {
                    "hrv": max(0.0, min(100.0, hrv_value)),
                    "breath": max(0.0, min(30.0, breath_value)),
                    "coherence": float(soul_metrics["coherence"]),
                    "focus": float(traits["focus"]),
                    "rmssd": float(rmssd)
                }
                
                # Advanced neural features in payload
                advanced_features = {}
                if ADVANCED_NEURAL_AVAILABLE and len(FEAT_BUFFER) >= 5:
                    advanced_features = {
                        "neural_processing": "advanced",
                        "manifold_coords": manifold_coords,
                        "multimodal_feedback": multimodal_feedback,
                        "risk_zones": risk_zones,
                        "advanced_features": True  # Flag for frontend
                    }
                
                # Construct enhanced response payload
                payload = {
                    "metricsBinding": True,
                    "hasLiveSignal": has_live,
                    "live": has_live,
                    "soul_metrics": soul_metrics,
                    "evolution": traits,  # ✅ REAL-TIME TRAITS from HRV (for display only)
                    "traits": traits,     # ✅ BACKEND CALCULATION same as Soul Metrics
                    "hrv": hrv_panel,
                    "consciousness": consciousness,
                    # 🎯 FRONTEND ΣΥΓΧΡΟΝΙΣΜΟΣ: Προσθήκη consciousness coordinates
                    "consciousness_x": consciousness.get("x", 0.0),
                    "consciousness_y": consciousness.get("y", 0.0)
                }
                
                # Add advanced neural features if available
                if ADVANCED_NEURAL_AVAILABLE and advanced_features:
                    payload.update(advanced_features)
                
                last_payload = payload
                
                # 🎯 STORE DATA FOR 24-EMOTION MAPPING
                from routes import telemetry_store
                telemetry_store["soul_metrics"] = soul_metrics
                telemetry_store["consciousness_data"] = consciousness
                
                logging.debug(f"🧠 Neural payload generated: Soul={soul_metrics}, Live={has_live}")
                return jsonify(payload)
                    
            except Exception as e:
                logging.error(f"🚫 Neural processing error: {e}")

        # Fallback: return last good payload with updated live status
        if last_payload:
            p = dict(last_payload)
            p["hasLiveSignal"] = has_live
            p["live"] = has_live
            return jsonify(p)

        # Initial fallback state - 0 VALUES when NO LIVE SIGNAL (ΖΩΝΗ OFF)
        return jsonify({
            "metricsBinding": True,
            "hasLiveSignal": has_live,
            "live": has_live,
            "soul_metrics": {"coherence": 0, "vitality": 0, "ethics": 0, "narrative": 0},
            "evolution": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
            "traits": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
            "hrv": {"hrv": 0, "breath": 0, "coherence": 0, "focus": 0, "rmssd": 0},
            "consciousness": {"x": 0.0, "y": 0.0}
        })

    # Neural buffer status endpoint
    @app.route("/neural/buffer-status", methods=["GET"])
    def neural_buffer_status():
        """Neural buffer diagnostics"""
        return jsonify({
            'buffers': {
                'rr_buffer': len(RR_BUFFER),
                'bpm_buffer': len(BPM_BUFFER),
                'feat_buffer': len(FEAT_BUFFER)
            },
            'neural_system': 'active',
            'last_signal': time.time() - last_signal_ts if last_signal_ts > 0 else None,
            'has_live_signal': (time.time() - last_signal_ts) <= 3.0
        })

    # Neural buffer reset endpoint
    @app.route("/neural/reset-buffers", methods=["POST"])
    def neural_reset_buffers():
        """Reset all neural buffers"""
        global last_signal_ts, last_payload
        RR_BUFFER.clear()
        BPM_BUFFER.clear()
        FEAT_BUFFER.clear()
        last_signal_ts = 0.0
        last_payload = None
        return jsonify({"status": "buffers_reset", "timestamp": now_utc()})

    @app.route("/neural/emotions", methods=["GET"])
    def neural_emotions():
        """
        🎭 24-EMOTION MAPPING ENDPOINT - DIRECT FROM YOU_NODE_STATE  
        Returns real-time 24 emotions calculated in routes.py
        """
        # Import YOU_NODE_STATE from routes.py
        from routes import YOU_NODE_STATE
        
        # Check if we have valid emotions data
        if YOU_NODE_STATE and "emotions" in YOU_NODE_STATE and YOU_NODE_STATE["emotions"]:
            emotions = YOU_NODE_STATE["emotions"]
            
            # Find dominant emotion (highest value)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotional_intensity = dominant_emotion[1] / 100.0
            
            return jsonify({
                "live": True,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion[0],
                "emotional_intensity": emotional_intensity,
                "coordinates": {
                    "x": YOU_NODE_STATE.get("x", 400),  # 🎯 ΚΡΙΣΙΜΟ: Backend coordinates!
                    "y": YOU_NODE_STATE.get("y", 300)   # 🎯 ΚΡΙΣΙΜΟ: Backend coordinates!
                },
                "biometrics": {
                    "vitality": YOU_NODE_STATE.get("vitality", 0.5) * 100,
                    "coherence": YOU_NODE_STATE.get("coherence", 0.5) * 100
                },
                "disclaimer": "🧠 LIVE: Real-time 24-emotion analysis from biometric data"
            })
        else:
            # Return neutral emotional state if no live data
            neutral_emotions = {
                "joy": 50.0, "confidence": 50.0, "hope": 50.0, "satisfaction": 50.0,
                "contentment": 50.0, "gratitude": 50.0, "sadness": 50.0, "despair": 50.0,
                "worry": 50.0, "fear": 50.0, "frustration": 50.0, "anger": 50.0,
                "concentration": 50.0, "determination": 50.0, "alertness": 50.0, "clarity": 50.0,
                "mindfulness": 50.0, "serenity": 50.0, "inspiration": 50.0, "wonder": 50.0,
                "compassion": 50.0, "love": 50.0, "empathy": 50.0, "connection": 50.0,
                "peace": 50.0, "balance": 50.0
            }
            return jsonify({
                "live": False,
                "emotions": neutral_emotions,
                "dominant_emotion": "neutral",
                "emotional_intensity": 0.5,
                "disclaimer": "🚫 NO LIVE DATA: Neutral emotional state"
            })

    logging.info("🧠 Neural routes registered successfully")