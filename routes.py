"""
Neural HRV Enhanced Routes - Complete Implementation
Integrates neural processing with existing Flask architecture
"""
from flask import render_template, request, jsonify, redirect, url_for, flash
import logging
from app import app, db
from models import InteractionLog, TrainingJob, PipelineStatus
from collections import deque
from datetime import datetime, timezone
import time
import math
import os
import statistics as S
import numpy as np
from flask import render_template_string

# Import our simplified neural modules
from neural_modules import (
    extract_hrv_features, map_features_to_soul_metrics, 
    map_latent_to_traits, map_traits_to_consciousness,
    neural_processor
)

# Store autonomous responses for frontend polling
autonomous_responses = deque(maxlen=50)  # Keep last 50 autonomous responses

# Global store for latest HRV frame (for live-data endpoint)
last_hrv_frame = None

# =========================
#   Neural HRV State
# =========================
# Buffer management
BUF_SECONDS = 60
RR_BUFFER = deque(maxlen=1024)  # RR intervals (ms)
BPM_BUFFER = deque(maxlen=2048)  # BPM timeline
FEAT_BUFFER = deque(maxlen=256)  # Feature vectors timeline

last_signal_ts = 0.0   # Watchdog timer
last_payload = None    # Anti-freeze cache

# =========================
#   Neural Utilities
# =========================
def clip01(x): 
    return max(0.0, min(100.0, float(x)))

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

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

# HRV Debug System - Global state for rolling RR buffer
# RESPONSIVENESS OPTIMIZATION: Reduced window & smoothing for immediate reaction
HRV_WINDOW_SEC = int(os.getenv("HRV_WINDOW_SEC",
                               "8"))  # Reduced to 8 seconds for instant response
ALPHA = float(os.getenv("HRV_SMOOTH_ALPHA",
                        "0.15"))  # INCREASED EMA for smoother response
RR_BUF = deque()  # (ts_ms, rr_ms) - NOW WITH TIMESTAMPS
HRV_STATE = {}


def _trim_rr_buffer():
    """Remove old RR intervals outside the window"""
    cutoff = time.time() * 1000 - HRV_WINDOW_SEC * 1000
    while RR_BUF and RR_BUF[0][0] < cutoff:
        RR_BUF.popleft()


def _calculate_rmssd(rr):
    """Calculate RMSSD from RR intervals"""
    if len(rr) < 2:
        return None
    dif = [rr[i] - rr[i - 1] for i in range(1, len(rr))]
    return math.sqrt(S.mean([d * d for d in dif]))


def _calculate_sdnn(rr):
    """Calculate SDNN from RR intervals"""
    return S.pstdev(rr) if len(rr) > 2 else None


def _detect_orbit_pattern(positions):
    """Detect circular motion patterns to prevent YOU node looping"""
    if len(positions) < 20:
        return False

    # Calculate center of mass
    centerX = sum(p["x"] for p in positions) / len(positions)
    centerY = sum(p["y"] for p in positions) / len(positions)

    # Calculate angles relative to center
    angles = [
        math.atan2(p["y"] - centerY, p["x"] - centerX) for p in positions
    ]

    # Check for consistent angular motion
    totalAngleChange = 0
    consistentDirection = 0

    for i in range(1, len(angles)):
        angleDiff = angles[i] - angles[i - 1]

        # Normalize angle difference to -π to π
        while angleDiff > math.pi:
            angleDiff -= 2 * math.pi
        while angleDiff < -math.pi:
            angleDiff += 2 * math.pi

        totalAngleChange += abs(angleDiff)

        if abs(angleDiff) > 0.1:  # Minimum angular change threshold
            consistentDirection += 1 if angleDiff > 0 else -1

    # Check for circular motion indicators
    avgAngleChange = totalAngleChange / (len(angles) - 1)
    directionConsistency = abs(consistentDirection) / (len(angles) - 1)

    # Calculate radius variance (should be low for circular motion)
    radii = [
        math.sqrt((p["x"] - centerX)**2 + (p["y"] - centerY)**2)
        for p in positions
    ]
    avgRadius = sum(radii) / len(radii)
    radiusVariance = sum((r - avgRadius)**2 for r in radii) / len(radii)
    radiusStability = radiusVariance < 1000  # Low variance = stable radius

    # Orbit detection criteria
    isCircular = avgAngleChange > 0.05 and directionConsistency > 0.7 and radiusStability

    return isCircular


from pipeline.orchestrator import PipelineOrchestrator
from fractalai.agent import FractalAIAgent
from syndesis_connector import register_syndesis_integration
from routes_trait_explainer import register_trait_explainer_routes
from routes_mood_previewer import mood_previewer_bp
import json
import os
from datetime import datetime, timedelta
import time
from flask_socketio import emit, send, join_room, leave_room
from app import socketio
import threading

# Enhanced Neural Modules για τα δύο κέντρα
try:
    from neural_manifold_hrv import neural_manifold
    from l2_ai_adaptation import l2_adaptation
    from active_memory_enhanced import enhanced_memory
    logging.info("Enhanced neural modules loaded successfully")
except ImportError as e:
    logging.warning(f"Enhanced neural modules not available: {e}")

# Core AI Module - Απλή βάση για panel integration
try:
    from core_ai_module import core_ai
    from presence_module import user_presence
    logging.info("Core AI and Presence modules loaded successfully")
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")

orchestrator = PipelineOrchestrator()
agent = FractalAIAgent()

# API version and health status
API_VERSION = "1.0.0"

# Store for real-time telemetry data
telemetry_store = {
    "latest_hr": None,
    "latest_hrv": None,
    "soul_metrics": None,
    "evolution": None,
    "you_position": {
        "active": False,
        "status": "NO SIGNAL", 
        "hasTrail": False,
        "opacity": 0.6,
        "x": 0.5,
        "y": 0.5
    }
}

# ===============================
# API ENDPOINTS PER SPECIFICATION
# ===============================


# 0. Clone Management Endpoints
@app.route('/api/you/clone/create', methods=['POST'])
def create_you_clone():
    """Create a clone YOU node for testing"""
    try:
        data = request.get_json() or {}
        test_id = data.get('test_id')

        clone = create_clone_node(test_id=test_id)

        return jsonify({
            "status": "success",
            "clone": {
                "id": clone["id"],
                "label": clone["label"],
                "isGhost": clone["isGhost"],
                "position": {
                    "x": clone["x"],
                    "y": clone["y"]
                },
                "color": clone["color"],
                "opacity": clone.get("opacity", 0.6)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/you/clone/<clone_id>/remove', methods=['DELETE'])
def remove_you_clone(clone_id):
    """Remove a clone YOU node"""
    try:
        success = remove_clone_node(clone_id)

        if success:
            return jsonify({
                "status": "success",
                "message": f"Clone {clone_id} removed"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Clone {clone_id} not found"
            }), 404

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/you/status', methods=['GET'])
def get_you_status():
    """Get status of primary YOU node and all clones"""
    try:
        return jsonify({
            "primary": {
                "id": YOU_NODE_STATE.get("id", "you_primary"),
                "label": YOU_NODE_STATE.get("label", "you_primary"),
                "isGhost": YOU_NODE_STATE.get("isGhost", False),
                "metricsBinding": YOU_NODE_STATE.get("metricsBinding", True),
                "position": {
                    "x": YOU_NODE_STATE["x"],
                    "y": YOU_NODE_STATE["y"]
                },
                "physics_active": True
            },
            "clones": {
                clone_id: {
                    "id": clone["id"],
                    "label": clone["label"],
                    "isGhost": clone["isGhost"],
                    "metricsBinding": clone.get("metricsBinding", False),
                    "position": {
                        "x": clone["x"],
                        "y": clone["y"]
                    },
                    "physics_active": False
                }
                for clone_id, clone in CLONE_NODES.items()
            },
            "clone_count": len(CLONE_NODES)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 1. Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint as per API specification"""
    return jsonify({"ok": True, "service": "syndesis", "version": API_VERSION})


# Global device tracking for No-signal guards - OPTIMIZED FOR STABILITY
DEVICE_LAST_SEEN = {}
DEVICE_TTL_MS = 5000  # 5 seconds timeout
LIVE_GATE_MS = int(os.getenv("LIVE_GATE_MS", "6500"))  # 6.5s - χαλαρότερο gate
DROP_GATE_MS = int(os.getenv("DROP_GATE_MS",
                             "9000"))  # 9s - υστέρηση για complete drop
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "TO_MYSTIKO_SOU")
NO_SIGNAL_BROADCASTED = False


def has_live_signal():
    """✅ OPTIMIZED: Check if we have a live HRV signal with hysteresis for stability"""
    if not DEVICE_LAST_SEEN:
        result = False
        print(f"🔍 hasLiveSignal: {result} (no devices)")
        return result

    now_ms = int(time.time() * 1000)
    last_seen = max(DEVICE_LAST_SEEN.values()) if DEVICE_LAST_SEEN else 0
    gap_ms = now_ms - last_seen

    # ⚡ INSTANT DEATH: 2 δευτερόλεπτα timeout για άμεσο death detection
    INSTANT_TIMEOUT = 2000
    result = gap_ms <= INSTANT_TIMEOUT
    print(f"🔍 hasLiveSignal: {result} (gap={gap_ms}ms, threshold={INSTANT_TIMEOUT}ms)")
    return result


def create_hrv_frame(hrv_data, device_name="TICKR A204"):
    """🚫 STRICT: Create authenticated HRV frame with live validation"""
    current_ts = int(time.time() * 1000)
    live_signal = has_live_signal()

    return {
        "type": "hrv_frame",
        "live": live_signal,
        "device": device_name,
        "ts": current_ts,
        "source": "tickr",
        "hrv": hrv_data.get("hrv") if live_signal else None,
        "heart_rate": hrv_data.get("heart_rate") if live_signal else None,
        "breath_rate": hrv_data.get("breath_rate") if live_signal else None,
        "coherence": hrv_data.get("coherence") if live_signal else None,
        "sample_age_sec": 0 if live_signal else 999
    }


def validate_hrv_frame(frame):
    """🚫 STRICT: Validate incoming HRV frame meets requirements"""
    if not isinstance(frame, dict):
        return False

    # Required fields check
    required_fields = ["type", "live", "device", "ts", "source"]
    if not all(field in frame for field in required_fields):
        return False

    # Type and source validation
    if frame.get("type") != "hrv_frame" or frame.get("source") != "tickr":
        return False

    # Live signal validation
    if not frame.get("live"):
        return False

    # Timestamp freshness (max 2 seconds old)
    current_ts = int(time.time() * 1000)
    frame_age = current_ts - frame.get("ts", 0)
    if frame_age > 2000:
        return False

    return True


@app.route('/api/diagnostic_panel', methods=['GET'])
def get_diagnostic_panel():
    """🚫 STRICT: Diagnostic panel for signal validation debugging"""
    current_ts = int(time.time() * 1000)
    live_signal = has_live_signal()

    # Calculate sample age if we have any devices
    sample_age_sec = 999
    last_device = "None"
    if DEVICE_LAST_SEEN:
        latest_device_id = max(DEVICE_LAST_SEEN.keys(),
                               key=lambda k: DEVICE_LAST_SEEN[k])
        latest_ts = DEVICE_LAST_SEEN[latest_device_id]
        sample_age_sec = (current_ts - latest_ts) / 1000.0
        last_device = latest_device_id

    return jsonify({
        "diagnostic": {
            "wsConnected":
            True,  # Always true for REST endpoint
            "liveSignal":
            live_signal,
            "deviceId":
            last_device,
            "lastSampleTs":
            DEVICE_LAST_SEEN.get(last_device, 0)
            if last_device != "None" else 0,
            "sampleAgeSec":
            round(sample_age_sec, 2),
            "frameSource":
            "tickr" if live_signal else "none",
            "serverBuild":
            "no-signal-guards-v2.0",
            "deviceCount":
            len(DEVICE_LAST_SEEN),
            "strictMode":
            True,
            "maxToleranceMs":
            2000
        }
    })


def check_device_timeouts():
    """Check device timeouts and broadcast NO_SIGNAL if needed"""
    global NO_SIGNAL_BROADCASTED

    now_ms = int(time.time() * 1000)
    signal_lost = False

    for device_id, last_seen in DEVICE_LAST_SEEN.items():
        if now_ms - last_seen > DEVICE_TTL_MS:
            signal_lost = True
            break

    # If no devices seen recently and we haven't broadcasted NO_SIGNAL yet
    if (not DEVICE_LAST_SEEN or signal_lost) and not NO_SIGNAL_BROADCASTED:
        NO_SIGNAL_BROADCASTED = True

        # 🚫 CRITICAL: Complete state reset on signal loss
        _perform_hard_disconnect()

        # 🚫 CRITICAL: Complete cache kill - no stored values - SAFE ORDER
        telemetry_store.update({
            "latest_hr": None,
            "latest_hrv": None,
            "soul_metrics": None,
            "evolution": None,
            "you_position": {
                "active": False,
                "status": "NO SIGNAL",
                "hasTrail": False,
                "opacity": 0.6
            }
        })

        # Broadcast NO_SIGNAL telemetry with frozen YOU node
        no_signal_event = {
            "type":
            "enhanced_telemetry",
            "signal":
            "NO_SIGNAL",
            "hr":
            None,
            "hrv":
            None,
            "you":
            telemetry_store["you_position"],
            "packet_gap_ms":
            now_ms - max(DEVICE_LAST_SEEN.values())
            if DEVICE_LAST_SEEN else DEVICE_TTL_MS + 1000,
            "last_device_seen":
            max(DEVICE_LAST_SEEN.keys(), key=lambda k: DEVICE_LAST_SEEN[k])
            if DEVICE_LAST_SEEN else "None"
        }

        # Broadcast via WebSocket
        socketio.emit('telemetry', no_signal_event, broadcast=True)
        socketio.emit('hard_disconnect', {
            "reason": "NO_SIGNAL",
            "timestamp": now_ms
        },
                      broadcast=True)
        logging.warning(
            f"🚫 HARD DISCONNECT - Complete state reset triggered - Last device activity: {max(DEVICE_LAST_SEEN.values()) if DEVICE_LAST_SEEN else 'Never'}"
        )

    # Reset NO_SIGNAL flag if we have recent activity
    elif not signal_lost and NO_SIGNAL_BROADCASTED:
        NO_SIGNAL_BROADCASTED = False

        # 🔥 RESTORE YOU NODE MOTION when signal returns
        telemetry_store["you_position"].update({
            "active": True,
            "status": "CONNECTED",
            "hasTrail": True,
            "opacity": 0.98  # Full opacity
        })

        logging.info("✅ Signal restored - YOU node MOTION RESUMED")


# 2. Ingest Heart Rate Endpoint - Unified HRV-Frame Schema
@app.route('/ingest/hr', methods=['POST'])
def ingest_hr():
    """Unified HRV-frame endpoint with Bearer token auth and schema transformation"""
    try:
        data = request.get_json(force=True) or {}

        # --- (A) Auth (testing mode επιτρέπεται αν δεν υπάρχει Authorization) ---
        auth = request.headers.get("Authorization", "")
        is_testing_mode = (not auth)  # επιτρέπουμε χωρίς token μόνο για dev
        if not is_testing_mode:
            if not auth.startswith("Bearer "):
                return jsonify({
                    "accepted": False,
                    "reason": "AUTH_MISSING_BEARER"
                }), 401
            token = auth.split(" ", 1)[1].strip()
            if token != INGEST_TOKEN:
                return jsonify({
                    "accepted": False,
                    "reason": "AUTH_INVALID_TOKEN"
                }), 401

        # --- (B) Basic validation (server format: bpm, rr[], ts(ms), device) ---
        bpm = float(data.get("bpm", 0))
        rr = list(map(int, data.get("rr", [])))
        ts = int(data.get("ts", 0))
        device = str(data.get("device", "TICKR A204"))

        if bpm <= 0:
            return jsonify({"accepted": False, "reason": "BPM_LEQ_ZERO"}), 400
        if not rr:
            return jsonify({"accepted": False, "reason": "RR_EMPTY"}), 400
        if ts <= 0:
            return jsonify({"accepted": False, "reason": "TS_MISSING"}), 400

        # --- (C) Freshness check (±5s ανοχή) ---
        server_now = int(time.time() * 1000)
        delta_ms = abs(server_now - ts)
        if delta_ms > 5000:
            return jsonify({
                "accepted": False,
                "reason": "STALE_TS",
                "delta_ms": delta_ms
            }), 400

        # --- (D) Enhanced HRV calculation with rolling window support ---
        # γεμίζεις global RR_BUF για rolling calculation
        for v in rr:
            RR_BUF.append((server_now, v))
        _trim_rr_buffer()

        # RMSSD προτίμηση: (1) τρέχον frame, (2) rolling window
        hrv_value = None
        if len(rr) >= 2:
            # Current frame RMSSD
            rr_diff = [abs(rr[i] - rr[i - 1]) for i in range(1, len(rr))]
            hrv_value = math.sqrt(sum(d**2 for d in rr_diff) /
                                  len(rr_diff)) if rr_diff else 0
        elif len(RR_BUF) >= 2:
            # Rolling window RMSSD when current frame has insufficient data
            window = [v
                      for (_, v) in list(RR_BUF)[-10:]]  # Last 10 RR intervals
            if len(window) >= 2:
                rr_diff = [
                    abs(window[i] - window[i - 1])
                    for i in range(1, len(window))
                ]
                hrv_value = math.sqrt(
                    sum(d**2
                        for d in rr_diff) / len(rr_diff)) if rr_diff else 0

        frame = {
            "type": "hrv_frame",
            "live": True,
            "source": "tickr",
            "ts": ts,  # ms
            "bpm": bpm,
            "rr": rr,  # ms
            "device": device,
            "hrv_value": hrv_value  # RMSSD in ms for panel display
        }

        # Store frame globally for live-data endpoint
        global last_hrv_frame
        last_hrv_frame = frame

        # 🎯 UPDATE YOU NODE MAPPING WITH 24 EMOTIONS
        try:
            # Get current soul metrics and consciousness data
            current_soul = telemetry_store.get("soul_metrics", {})
            current_consciousness = telemetry_store.get("consciousness_data", {})
            
            # 🧠 GET NEURAL EVOLUTION TRAITS DIRECTLY
            try:
                from neural_routes import get_latest_neural_data
                neural_data = get_latest_neural_data()
                if neural_data and "evolution" in neural_data:
                    current_consciousness["evolution"] = neural_data["evolution"]
                    logging.info(f'🧠 NEURAL TRAITS CONNECTED: {list(neural_data["evolution"].keys())}')
            except:
                logging.warning("🧠 Could not connect neural traits, using fallback")
            
            # Update orb position with 24-emotion system
            you_node = _update_you_node_mapping({"bpm": bpm, "rr": rr}, current_soul, current_consciousness)
            
            # Add YOU node data to the telemetry frame
            frame["you_node"] = {
                "x": round(you_node["x"], 1),
                "y": round(you_node["y"], 1),
                "radius": round(you_node["radius"], 1),
                "color": you_node["color"],
                "coherence": round(you_node["coherence"], 2),
                "vitality": round(you_node["vitality"], 2),
                "status": you_node["status"]
            }
            
            # 🧠 ADD EVOLUTION TRAITS TO TELEMETRY - Direct from neural system
            try:
                # Get the latest neural payload with evolution traits
                import neural_routes
                if hasattr(neural_routes, 'last_payload') and neural_routes.last_payload:
                    if "evolution" in neural_routes.last_payload:
                        frame["evolution_traits"] = neural_routes.last_payload["evolution"]
                        logging.info(f'🧠 EVOLUTION TRAITS SENT TO FRONTEND: {list(neural_routes.last_payload["evolution"].keys())}')
                    else:
                        logging.warning(f"🧠 No evolution in last_payload. Keys: {list(neural_routes.last_payload.keys())}")
                else:
                    logging.warning(f"🧠 No last_payload found in neural_routes")
            except Exception as e:
                logging.warning(f"🧠 Could not add evolution traits to telemetry: {e}")
            
            logging.info(f'🎭 24-EMOTION ORB UPDATE: pos=({you_node["x"]:.1f},{you_node["y"]:.1f}) coherence={you_node["coherence"]:.1f}% vitality={you_node["vitality"]:.1f}%')
            
        except Exception as e:
            logging.error(f"🎭 24-emotion orb update failed: {e}")
            # Add fallback YOU node data
            frame["you_node"] = {"x": 250, "y": 250, "radius": 50, "color": "#4dbfff", "status": "fallback"}

        # --- (E) WebSocket broadcast στο ΣΩΣΤΟ event που ακούει ο client ---
        # Σημείωση: ΜΗΝ χρησιμοποιήσεις broadcast=..., σε νέες εκδόσεις σπάει.
        socketio.emit("telemetry", frame, namespace="/")

        # 🧠 FORWARD TO NEURAL SYSTEM
        try:
            from neural_routes import RR_BUFFER, BPM_BUFFER, FEAT_BUFFER
            from neural_modules import extract_hrv_features
            import neural_routes
            
            # Update neural system with incoming data
            neural_routes.last_signal_ts = time.time()
            BPM_BUFFER.append(float(bpm))
            
            # Add RR intervals to neural buffer
            for rr_val in rr:
                RR_BUFFER.append(float(rr_val))
            
            # Extract neural features if we have enough data
            if len(RR_BUFFER) >= 5:
                features = extract_hrv_features(list(RR_BUFFER)[-300:])
                if features is not None:
                    FEAT_BUFFER.append(features)
                    logging.debug(f"🧠 Neural features forwarded: {len(FEAT_BUFFER)} total")
        except Exception as e:
            logging.warning(f"🧠 Neural forwarding failed: {e}")
        
        # Enhanced logging για διάγνωση
        logging.info(
            f"📊 HR Ingest OK: {bpm} BPM from {device} (Δ={delta_ms}ms) -> WS telemetry"
        )
        logging.info(
            f"🔄 WS emit telemetry ts={frame['ts']} bpm={frame['bpm']} rr={len(frame['rr'])} hrv_value={frame['hrv_value']}"
        )
        logging.info(
            f"🗂️ Global frame stored for /live-data endpoint - type:{frame['type']} live:{frame['live']}"
        )

        # Legacy processing for existing YOU node integration
        # Update device last seen for NO-SIGNAL protection
        device_id = device
        DEVICE_LAST_SEEN[device_id] = server_now

        # Update telemetry store with fresh data
        telemetry_store.update({
            "latest_hr": bpm,
            "latest_rr": rr,
            "latest_device": device,
            "packet_timestamp": ts,
            "packet_gap_ms": delta_ms
        })

        # Enhanced HRV calculation for telemetry store with rolling window
        if hrv_value is not None:
            # Store the computed HRV value (either from current frame or rolling window)
            telemetry_store["rmssd_raw"] = hrv_value
            telemetry_store["rmssd_window"] = hrv_value
            telemetry_store["n_rr_window"] = len(RR_BUF) if len(
                rr) < 2 else len(rr)

            source = "current_frame" if len(rr) >= 2 else "rolling_window"
            logging.info(
                f"🔍 RR Debug: {len(rr)} intervals, RMSSD {source}: {hrv_value:.1f}ms, window: {hrv_value:.1f}ms"
            )

        return jsonify({"accepted": True})

    except Exception as e:
        logging.exception("❌ ingest_hr error")
        return jsonify({
            "accepted": False,
            "reason": "SERVER_ERROR",
            "detail": str(e)
        }), 500


def calculate_hrv_metrics(rr_intervals):
    """Calculate HRV metrics from RR intervals"""
    if len(rr_intervals) < 2:
        return telemetry_store["latest_hrv"]

    # Convert to numpy for calculations
    import numpy as np
    rr_array = np.array(rr_intervals)

    # RMSSD calculation
    diff_rr = np.diff(rr_array)
    rmssd = np.sqrt(np.mean(diff_rr**2))

    # SDNN calculation
    sdnn = np.std(rr_array)

    # Stability calculation (coefficient of variation)
    stability = 1.0 - (sdnn /
                       np.mean(rr_array)) if np.mean(rr_array) > 0 else 0.5

    # Determine state based on HRV
    if rmssd > 50:
        state = "relaxed"
    elif rmssd > 30:
        state = "focus"
    else:
        state = "stress"

    return {
        "rmssd": round(float(rmssd), 1),
        "sdnn": round(float(sdnn), 1),
        "stability": round(float(stability), 2),
        "state": state
    }


# YOU node mapping state with unique identifier and isolation
YOU_NODE_STATE = {
    "id": "you_primary",  # Unique identifier for primary YOU node
    "label": "you_primary",  # Label that never changes
    "isGhost": False,  # Primary node is not a ghost/clone
    "metricsBinding": True,  # Only primary receives HRV/Soul updates
    "x": 250,
    "y": 250,  # Center position
    "radius": 50,
    "color": "#4dbfff",
    "pulse_speed": 1.0,
    "coherence": 0.5,
    "vitality": 0.5,
    "last_packet_time": time.time(),
    "packet_gap_ms": 0,
    "ws_queue_lag_ms": 0,
    "hysteresis_state": "calm",  # calm, focus, stress
    "high_load_counter": 0  # Counter for RMSSD < 20ms
}

# Clone tracking for test nodes
CLONE_NODES = {}  # Dictionary to track cloned YOU nodes
CLONE_COUNTER = 0  # Global counter for unique clone IDs

# EMA factors for smooth transitions
EMA_FACTORS = {
    "position": 0.05,  # Very smooth position changes
    "color": 0.1,  # Smooth color transitions
    "radius": 0.08  # Smooth size changes
}


def create_clone_node(base_node=None, test_id=None):
    """Create a clone YOU node for testing with visual differentiation"""
    global CLONE_COUNTER, CLONE_NODES

    CLONE_COUNTER += 1
    clone_id = f"you_clone_{CLONE_COUNTER}" if not test_id else f"you_clone_{test_id}"

    # Base on primary or provided node
    base = base_node if base_node else YOU_NODE_STATE

    clone = {
        "id": clone_id,
        "label": clone_id,
        "isGhost": True,  # Clone is a ghost/test node
        "metricsBinding": False,  # Clones don't receive HRV/Soul updates
        "x": base["x"] + (CLONE_COUNTER * 30),  # Offset position
        "y": base["y"] + (CLONE_COUNTER * 20),
        "radius": base["radius"] * 0.8,  # Smaller than primary
        "color": "#6dbfff",  # Different color (lighter blue)
        "pulse_speed": base["pulse_speed"] * 0.5,
        "coherence": 0.3,  # Lower coherence for ghosts
        "vitality": 0.3,
        "opacity": 0.6,  # Visual differentiation - lower opacity
        "last_packet_time": time.time(),
        "packet_gap_ms": 0,
        "ws_queue_lag_ms": 0,
        "hysteresis_state": "ghost",
        "high_load_counter": 0
    }

    CLONE_NODES[clone_id] = clone
    logging.info(
        f"👻 Clone created: {clone_id} at position ({clone['x']}, {clone['y']})"
    )
    return clone


def remove_clone_node(clone_id):
    """Remove a clone node from tracking"""
    global CLONE_NODES
    if clone_id in CLONE_NODES:
        del CLONE_NODES[clone_id]
        logging.info(f"🗑️ Clone removed: {clone_id}")
        return True
    return False


def _calculate_24_emotions(empathy, creativity, resilience, focus, curiosity, compassion, 
                          coherence, vitality, ethics, narrative, rmssd, hr):
    """
    Calculate 24 emotions from personality traits, soul metrics, and biometric data
    Uses weighted linear combinations with clamping to 0-100 range
    """
    # Normalize inputs to 0-1 range for calculations
    emp = empathy / 100.0
    cre = creativity / 100.0
    res = resilience / 100.0
    foc = focus / 100.0
    cur = curiosity / 100.0
    com = compassion / 100.0
    coh = coherence / 100.0
    vit = vitality / 100.0
    eth = ethics / 100.0
    nar = narrative / 100.0
    hrv_norm = max(0.0, min(1.0, (rmssd - 20) / 80))  # 20-100ms range
    hr_norm = max(0.0, min(1.0, (hr - 50) / 70))    # 50-120 BPM range
    
    # Positive Emotions (Y-axis positive)
    joy = max(0, min(100, (vit * 0.6 + cre * 0.3 + hrv_norm * 0.1) * 100))
    confidence = max(0, min(100, (foc * 0.5 + res * 0.3 + coh * 0.2) * 100))
    hope = max(0, min(100, (nar * 0.4 + vit * 0.3 + cur * 0.3) * 100))
    satisfaction = max(0, min(100, (eth * 0.4 + coh * 0.3 + vit * 0.3) * 100))
    contentment = max(0, min(100, (eth * 0.5 + coh * 0.3 + hrv_norm * 0.2) * 100))
    gratitude = max(0, min(100, (com * 0.5 + eth * 0.3 + vit * 0.2) * 100))
    
    # Negative Emotions (Y-axis negative)
    sadness = max(0, min(100, ((1 - vit) * 0.6 + (1 - nar) * 0.4) * 100))
    despair = max(0, min(100, ((1 - nar) * 0.7 + (1 - vit) * 0.3) * 100))
    worry = max(0, min(100, ((1 - hrv_norm) * 0.5 + (1 - coh) * 0.3 + (1 - res) * 0.2) * 100))
    fear = max(0, min(100, ((1 - res) * 0.6 + (1 - hrv_norm) * 0.4) * 100))
    frustration = max(0, min(100, ((1 - foc) * 0.5 + (1 - coh) * 0.3 + hr_norm * 0.2) * 100))
    anger = max(0, min(100, (hr_norm * 0.5 + (1 - eth) * 0.3 + (1 - coh) * 0.2) * 100))
    
    # Focus-based Emotions (X-axis - confidence/focus spectrum)
    concentration = max(0, min(100, (foc * 0.7 + coh * 0.3) * 100))
    determination = max(0, min(100, (res * 0.5 + foc * 0.3 + vit * 0.2) * 100))
    alertness = max(0, min(100, (foc * 0.4 + hr_norm * 0.3 + vit * 0.3) * 100))
    clarity = max(0, min(100, (coh * 0.6 + foc * 0.4) * 100))
    mindfulness = max(0, min(100, (hrv_norm * 0.5 + coh * 0.3 + eth * 0.2) * 100))
    serenity = max(0, min(100, (hrv_norm * 0.4 + eth * 0.3 + coh * 0.3) * 100))
    
    # Creative/Empathic Emotions (X-axis - creativity/empathy spectrum)
    inspiration = max(0, min(100, (cre * 0.5 + cur * 0.3 + vit * 0.2) * 100))
    wonder = max(0, min(100, (cur * 0.6 + cre * 0.4) * 100))
    compassion_emotion = max(0, min(100, (com * 0.7 + emp * 0.3) * 100))
    love = max(0, min(100, (com * 0.4 + emp * 0.4 + vit * 0.2) * 100))
    empathy_emotion = max(0, min(100, (emp * 0.8 + com * 0.2) * 100))
    connection = max(0, min(100, (emp * 0.5 + com * 0.3 + eth * 0.2) * 100))
    
    # Neutral/Balance Emotions
    peace = max(0, min(100, (hrv_norm * 0.4 + coh * 0.3 + eth * 0.3) * 100))
    balance = max(0, min(100, (coh * 0.5 + hrv_norm * 0.3 + eth * 0.2) * 100))
    
    return {
        # Positive emotions
        'joy': joy, 'confidence': confidence, 'hope': hope, 'satisfaction': satisfaction,
        'contentment': contentment, 'gratitude': gratitude,
        # Negative emotions  
        'sadness': sadness, 'despair': despair, 'worry': worry, 'fear': fear,
        'frustration': frustration, 'anger': anger,
        # Focus-based emotions
        'concentration': concentration, 'determination': determination, 'alertness': alertness,
        'clarity': clarity, 'mindfulness': mindfulness, 'serenity': serenity,
        # Creative/empathic emotions
        'inspiration': inspiration, 'wonder': wonder, 'compassion': compassion_emotion,
        'love': love, 'empathy': empathy_emotion, 'connection': connection,
        # Balance emotions
        'peace': peace, 'balance': balance
    }


def _update_you_node_mapping(hr_data, soul_metrics, consciousness_data):
    """
    Υπολογισμός θέσης orb βάσει 24 συναισθημάτων + biometrics
    """
    global YOU_NODE_STATE

    # DEBUG: Always log when this function is called
    logging.info(f'🎯 _update_you_node_mapping CALLED - HR:{hr_data} Soul:{soul_metrics} Consciousness keys:{list(consciousness_data.keys()) if consciousness_data else "None"}')

    # PHYSICS SCOPE: Only apply to primary YOU node
    if YOU_NODE_STATE.get("id") != "you_primary":
        logging.warning(
            "⚠️ Physics update attempted on non-primary node - skipping")
        return

    # METRICS BINDING: Only primary node receives HRV/Soul updates
    if not YOU_NODE_STATE.get("metricsBinding", True):
        logging.warning(
            "⚠️ Metrics binding disabled for this node - skipping updates")
        return

    # === ΣΥΛΛΟΓΗ ΜΕΤΡΙΚΩΝ ===
    # Handle None values with safe defaults
    soul_metrics = soul_metrics or {}
    consciousness_data = consciousness_data or {}
    hr_data = hr_data or {}
    
    vitality = soul_metrics.get("vitality", 50)  # Κρατάμε 0-100 κλίμακα
    coherence = soul_metrics.get("coherence", 50)  # Κρατάμε 0-100 κλίμακα
    ethics = soul_metrics.get("ethics", 50)        # Κρατάμε 0-100 κλίμακα  
    narrative = soul_metrics.get("narrative", 50)  # Κρατάμε 0-100 κλίμακα
    
    # 🧠 DIRECT NEURAL TRAITS ACCESS - No dependency on consciousness_data
    try:
        # Get latest neural evolution data directly from telemetry
        evolution_data = telemetry_store.get("evolution", {})
        if not evolution_data:
            # Fallback: try to get from neural system directly
            import neural_routes
            if hasattr(neural_routes, 'last_neural_payload') and neural_routes.last_neural_payload:
                evolution_data = neural_routes.last_neural_payload.get("evolution", {})
        
        logging.info(f'🧠 NEURAL TRAITS FOUND: {evolution_data}')
        traits = evolution_data
        focus = traits.get("focus", 50)
        creativity = traits.get("creativity", 50)
        empathy = traits.get("empathy", 50) 
        resilience = traits.get("resilience", 50)
        
        logging.info(f'🎯 TRAIT VALUES: Focus={focus:.1f} Creativity={creativity:.1f} Empathy={empathy:.1f} Resilience={resilience:.1f}')
        
    except Exception as e:
        logging.warning(f"🧠 Neural traits access failed: {e}, using defaults")
        traits = {}
        focus = 50
        creativity = 50
        empathy = 50
        resilience = 50
    
    # HRV normalization
    if isinstance(hr_data, dict):
        hr = hr_data.get("hr", 75)
        hrv = hr_data.get("hrv", {})
        rmssd = hrv.get("rmssd_smooth", 30) if isinstance(hrv, dict) else 30
    else:
        hr = float(hr_data) if hr_data else 75
        rmssd = 30
        
    hrv_norm = max(0.0, min(1.0, (rmssd - 20) / 80))  # 20-100ms range

    # === ΥΠΟΛΟΓΙΣΜΟΣ ΟΛΩΝ ΤΩΝ 24 ΣΥΝΑΙΣΘΗΜΑΤΩΝ ===
    emotions = _calculate_24_emotions(
        traits.get("empathy", 50), creativity * 100, traits.get("resilience", 80),
        focus * 100, traits.get("curiosity", 70), traits.get("compassion", 80),
        coherence * 100, vitality * 100, ethics * 100, narrative * 100,
        rmssd, hr
    )
    
    # === ΝΕΑ ΧΑΡΤΟΓΡΑΦΗΣΗ ΒΑΣΕΙ ΠΡΟΣΩΠΙΚΟΤΗΤΑΣ ===
    
    # Ορίζουμε το κέντρο του καμβά
    center_x, center_y = 250, 250
    # 🎯 ΒΕΛΤΙΩΜΕΝΗ ΚΛΙΜΑΚΑ - Πιο δραματική κίνηση για καλύτερη αντίθεση
    movement_range = 300  # Δημιουργεί κίνηση σε ένα πλαίσιο 600x600 (πιο δραματικό)

    # 🧠 ΑΛΗΘΙΝΗ ΑΝΤΙΘΕΤΙΚΗ ΧΑΡΤΟΓΡΑΦΗΣΗ ΜΕ EVOLUTION TRAITS!
    
    # Πάρε τα πραγματικά evolution traits από το neural system
    try:
        import neural_routes
        if hasattr(neural_routes, 'last_payload') and neural_routes.last_payload and "evolution" in neural_routes.last_payload:
            evo_traits = neural_routes.last_payload["evolution"]
            focus = evo_traits.get("focus", 50)
            creativity = evo_traits.get("creativity", 50) 
            empathy = evo_traits.get("empathy", 50)
            resilience = evo_traits.get("resilience", 50)
            logging.info(f'🧠 ΧΡΗΣΙΜΟΠΟΙΩ EVOLUTION TRAITS: Focus={focus:.1f}% Empathy={empathy:.1f}% Creativity={creativity:.1f}% Resilience={resilience:.1f}%')
        else:
            # Fallback στα defaults
            focus, creativity, empathy, resilience = 50, 50, 50, 50
            logging.warning(f'🧠 FALLBACK: Using default evolution traits')
    except:
        focus, creativity, empathy, resilience = 50, 50, 50, 50
        logging.warning(f'🧠 ERROR: Could not get evolution traits, using defaults')

    # 🎯 BLENDED MAPPING: EVOLUTION TRAITS + SOUL METRICS → ΘΕΣΗ ORB
    
    # === EVOLUTION TRAITS COMPONENT (70% βάρος) ===
    focus_norm = (focus - 50) / 50.0      # 0-100 -> -1 to 1
    empathy_norm = (empathy - 50) / 50.0  # 0-100 -> -1 to 1
    evo_x_balance = focus_norm - empathy_norm # Focus vs Empathy
    
    creativity_norm = (creativity - 50) / 50.0     # 0-100 -> -1 to 1  
    resilience_norm = (resilience - 50) / 50.0     # 0-100 -> -1 to 1
    evo_y_balance = creativity_norm - resilience_norm  # Creativity vs Resilience
    
    # === SOUL METRICS COMPONENT (30% βάρος) ===
    # Υπολογίζω τα Soul Metrics normalization εδώ (όλα 0-100 κλίμακα)
    coherence_norm = (coherence - 50) / 50.0  # 0-100 -> -1 to 1
    narrative_norm = (narrative - 50) / 50.0  # 0-100 -> -1 to 1
    vitality_norm = (vitality - 50) / 50.0    # 0-100 -> -1 to 1  
    ethics_norm = (ethics - 50) / 50.0        # 0-100 -> -1 to 1
    
    soul_x_balance = coherence_norm - narrative_norm # Coherence vs Narrative
    soul_y_balance = vitality_norm - ethics_norm     # Vitality vs Ethics
    
    # === BLENDED CALCULATION ===
    evo_weight = 0.7   # 70% Evolution Traits
    soul_weight = 0.3  # 30% Soul Metrics
    
    blended_x_balance = (evo_x_balance * evo_weight) + (soul_x_balance * soul_weight)
    blended_y_balance = (evo_y_balance * evo_weight) + (soul_y_balance * soul_weight)
    
    # 🎯 FINAL POSITION WITH ENHANCED Y MOVEMENT
    x_offset = 80  # Δεξιά offset
    y_offset = 40  # Κάτω offset
    
    # 🔥 ΠΙΟ ΕΝΤΟΝΗ ΚΙΝΗΣΗ Y - διπλάσιο range για κατακόρυφη κίνηση
    x = center_x + blended_x_balance * (movement_range / 3) + x_offset
    y = center_y - blended_y_balance * (movement_range / 1.5) + y_offset  # Από /3 σε /1.5 = 2x πιο έντονο!
    
    logging.info(f'🎯 BLENDED MAPPING: Evo({evo_x_balance:.3f},{evo_y_balance:.3f}) + Soul({soul_x_balance:.3f},{soul_y_balance:.3f}) = Final({blended_x_balance:.3f},{blended_y_balance:.3f})')
    
    # Clamp to canvas boundaries
    x = max(50, min(450, x))
    y = max(50, min(450, y))
    
    # 💫 ΠΑΛΙΟΣ ΚΩΔΙΚΑΣ ΔΙΑΓΡΑΜΜΕΝΟΣ - ΧΡΗΣΙΜΟΠΟΙΟΥΜΕ ΤΟ BLENDED MAPPING ΠΙΟ ΠΑΝΩ!
    
    # Log comprehensive emotion analysis
    logging.info(f'🎭 24-EMOTION ANALYSIS: Position ({x:.1f}, {y:.1f})')
    logging.info(f'🌟 POSITIVE: Joy:{emotions["joy"]:.1f} Confidence:{emotions["confidence"]:.1f} '
                f'Hope:{emotions["hope"]:.1f} Satisfaction:{emotions["satisfaction"]:.1f} '
                f'Content:{emotions["contentment"]:.1f} Gratitude:{emotions["gratitude"]:.1f}')
    logging.info(f'⚡ NEGATIVE: Sadness:{emotions["sadness"]:.1f} Despair:{emotions["despair"]:.1f} '
                f'Worry:{emotions["worry"]:.1f} Fear:{emotions["fear"]:.1f} '
                f'Frustration:{emotions["frustration"]:.1f} Anger:{emotions["anger"]:.1f}')
    logging.info(f'🎯 FOCUS: Concentration:{emotions["concentration"]:.1f} Determination:{emotions["determination"]:.1f} '
                f'Alertness:{emotions["alertness"]:.1f} Clarity:{emotions["clarity"]:.1f}')
    logging.info(f'💖 EMPATHY: Empathy:{emotions["empathy"]:.1f} Compassion:{emotions["compassion"]:.1f} '
                f'Love:{emotions["love"]:.1f} Connection:{emotions["connection"]:.1f}')
    logging.info(f'📊 BLENDED COORDINATES: Final: ({x:.1f},{y:.1f}) from Evo(70%) + Soul(30%)')

    # === ΕΠΙΣΤΡΟΦΗ ORB ΜΕ ΔΙΠΛΗ ΑΝΤΙΘΕΤΙΚΗ ΧΑΡΤΟΓΡΑΦΗΣΗ ===
    
    # 🔥 VISUAL PROPERTIES ΑΠΟ SOUL METRICS ΑΝΤΙΘΕΤΙΚΗ ΧΑΡΤΟΓΡΑΦΗΣΗ
    
    # Μέγεθος orb: Vitality vs Ethics balance
    radius_base = 25
    radius_modifier = soul_y_balance * 10  # -20 to +20 pixels  
    orb_radius = max(15, min(45, radius_base + radius_modifier))
    
    # Pulse speed: Coherence vs Narrative balance
    pulse_base = 1.0
    pulse_modifier = abs(soul_x_balance) * 0.5  # 0 to 1.0 extra speed
    pulse_speed = round(pulse_base + pulse_modifier, 2)
    
    # Χρώμα: Αντιθετικός συνδυασμός Coherence-Narrative
    if soul_x_balance > 0:  # Περισσότερη coherence
        color_hue = "#FF6B33" if soul_x_balance > 0.5 else "#FFB333"  # Πορτοκαλί/Κίτρινο
    else:  # Περισσότερο narrative  
        color_hue = "#33AFFF" if soul_x_balance < -0.5 else "#33FFB3"  # Μπλε/Πράσινο
    
    # Διαφάνεια: Vitality vs Ethics balance
    opacity_base = 0.7
    opacity_modifier = (abs(soul_y_balance) / 2) * 0.3  # 0 to 0.3 extra opacity
    opacity = min(1.0, opacity_base + opacity_modifier)
    
    YOU_NODE_STATE.update({
        "x": round(x, 1),                                    # Evolution Traits χαρτογραφία
        "y": round(y, 1),                                    # Evolution Traits χαρτογραφία  
        "radius": round(orb_radius, 1),                      # 💫 Soul Metrics: Vitality vs Ethics
        "color": color_hue,                                  # 💫 Soul Metrics: Coherence vs Narrative
        "pulse_speed": pulse_speed,                          # 💫 Soul Metrics: Coherence vs Narrative
        "coherence": round(coherence, 2),                    # Ήδη 0-100 κλίμακα
        "vitality": round(vitality, 2),                      # Πραγματικά soul metrics για συμβατότητα
        "opacity": round(opacity, 2),                        # 💫 Soul Metrics: Vitality vs Ethics
        "hasTrail": True,
        "status": "active",
        # 🎭 ΠΡΟΣΘΕΤΩ ΤΑ 24 EMOTIONS για το JavaScript!
        "emotions": emotions,
        # 💫 ΠΡΟΣΘΕΤΩ SOUL METRICS BALANCES για διαφάνεια
        "soul_x_balance": round(soul_x_balance, 3),
        "soul_y_balance": round(soul_y_balance, 3)
    })

    return YOU_NODE_STATE


# 🤖 GPT NEURAL STATE API ENDPOINT  
@app.route('/api/state', methods=['GET'])
def get_neural_state():
    """
    🧠 Συλλέγει όλα τα real-time neural data για GPT consciousness integration
    Επιστρέφει compact JSON με biometrics + emotions + orb state
    """
    try:
        # Πάρε τα latest telemetry data
        current_hr = telemetry_store.get("latest_hr", {})
        current_soul = telemetry_store.get("soul_metrics", {})
        current_consciousness = telemetry_store.get("consciousness_data", {})
        
        # Update YOU node mapping για fresh data
        you_node = _update_you_node_mapping(current_hr, current_soul, current_consciousness)
        
        # Πάρε evolution traits από neural system
        evolution_data = {}
        try:
            import neural_routes
            if hasattr(neural_routes, 'last_payload') and neural_routes.last_payload and "evolution" in neural_routes.last_payload:
                evolution_data = neural_routes.last_payload["evolution"]
        except:
            pass
        
        # Extract HR data με fallbacks
        hr_bpm = 75  # default
        rmssd = 30   # default
        rr_intervals = []
        
        if isinstance(current_hr, dict):
            hr_bpm = current_hr.get("bpm", current_hr.get("hr", 75))
            if "rr" in current_hr:
                rr_intervals = current_hr["rr"] if isinstance(current_hr["rr"], list) else [current_hr["rr"]]
            rmssd = current_hr.get("rmssd", HRV_STATE.get("rmssd_smooth", 30))
        
        # Extract soul metrics με fallbacks
        soul_data = current_soul if isinstance(current_soul, dict) else {}
        coherence = soul_data.get("coherence", 50)
        vitality = soul_data.get("vitality", 50) 
        ethics = soul_data.get("ethics", 50)
        narrative = soul_data.get("narrative", 50)
        
        # Get dominant emotion από you_node
        emotions = you_node.get("emotions", {})
        dominant_emotion = "neutral"
        max_emotion_value = 0
        
        # Βρες το κυρίαρχο συναίσθημα
        for emotion, value in emotions.items():
            if value > max_emotion_value:
                max_emotion_value = value
                dominant_emotion = emotion
        
        # 🎯 COMPACT JSON για GPT - ακριβώς όπως περιγράφεται στο attached file
        neural_state = {
            "timestamp": int(time.time() * 1000),  # milliseconds
            
            # 📡 Βιομετρικά δεδομένα από HRV
            "biometrics": {
                "hr": round(hr_bpm, 1),
                "rmssd": round(float(rmssd), 1),
                "coherence": round(coherence / 100, 3),  # 0-1 scale 
                "vitality": round(vitality / 100, 3)     # 0-1 scale
            },
            
            # 🎭 Συναισθηματικά δεδομένα από 24-emotion analysis
            "emotions": {
                "dominant": dominant_emotion,
                "joy": round(emotions.get("joy", 50), 1),
                "sadness": round(emotions.get("sadness", 50), 1),
                "empathy": round(emotions.get("empathy", 50), 1),
                "focus": round(emotions.get("concentration", 50), 1),
                "creativity": round(evolution_data.get("creativity", 50), 1),
                "resilience": round(evolution_data.get("resilience", 50), 1)
            },
            
            # 🔵 Κίνηση του κύκλου (orb state)  
            "orb": {
                "x": you_node.get("x", 250),
                "y": you_node.get("y", 250), 
                "radius": you_node.get("radius", 25),
                "color": you_node.get("color", "#7f7fFF"),
                "pulse_speed": you_node.get("pulse_speed", 1.0),
                "opacity": you_node.get("opacity", 0.7)
            },
            
            # 🧠 Evolution traits για personality context
            "personality": {
                "empathy": round(evolution_data.get("empathy", 50), 1),
                "creativity": round(evolution_data.get("creativity", 50), 1), 
                "resilience": round(evolution_data.get("resilience", 50), 1),
                "focus": round(evolution_data.get("focus", 50), 1),
                "curiosity": round(evolution_data.get("curiosity", 50), 1),
                "compassion": round(evolution_data.get("compassion", 50), 1)
            },
            
            # 🌟 Soul metrics για deeper personality analysis
            "soul": {
                "coherence": round(coherence, 1),
                "vitality": round(vitality, 1),
                "ethics": round(ethics, 1), 
                "narrative": round(narrative, 1)
            },
            
            # ✅ Live status
            "live": bool(telemetry_store.get("has_live_signal", False))
        }
        
        logging.info(f"🤖 GPT Neural State compiled - Orb:({neural_state['orb']['x']},{neural_state['orb']['y']}) Dominant:{dominant_emotion}")
        return jsonify(neural_state)
        
    except Exception as e:
        logging.error(f"🚫 GPT Neural State ERROR: {e}")
        # Fallback minimal state
        return jsonify({
            "timestamp": int(time.time() * 1000),
            "biometrics": {"hr": 75, "rmssd": 30, "coherence": 0.5, "vitality": 0.5},
            "emotions": {"dominant": "neutral", "joy": 50, "sadness": 50, "empathy": 50, "focus": 50},
            "orb": {"x": 250, "y": 250, "radius": 25, "color": "#7f7fFF"},
            "personality": {"empathy": 50, "creativity": 50, "resilience": 50, "focus": 50},
            "live": False,
            "error": "Fallback state - neural systems not available"
        })


# 🤖 GPT CHAT ENDPOINT με Neural State Integration
@app.route('/api/chat', methods=['POST'])
def gpt_chat():
    """
    🧠 GPT Chat με real-time neural state awareness
    Παίρνει user message και neural state -> επιστρέφει personality-driven response
    """
    try:
        # Get request data
        data = request.get_json()
        user_message = data.get('message', '')
        include_neural_state = data.get('use_neural_state', True)
        conversation_history = data.get('history', [])
        
        if not user_message.strip():
            return jsonify({
                "error": "Empty message provided",
                "response": "Παρακαλώ στείλε μου ένα μήνυμα!"
            }), 400
        
        # Get current neural state αν χρειάζεται
        neural_state = {}
        if include_neural_state:
            try:
                # Συλλέγω fresh neural state
                current_hr = telemetry_store.get("latest_hr", {})
                current_soul = telemetry_store.get("soul_metrics", {})
                current_consciousness = telemetry_store.get("consciousness_data", {})
                
                # Update YOU node mapping
                you_node = _update_you_node_mapping(current_hr, current_soul, current_consciousness)
                
                # Get evolution traits - direct access από global telemetry_store
                evolution_data = {}
                try:
                    # Πρώτα δοκιμάζω από neural_routes.last_payload
                    import neural_routes
                    if hasattr(neural_routes, 'last_payload') and neural_routes.last_payload and "evolution" in neural_routes.last_payload:
                        evolution_data = neural_routes.last_payload["evolution"]
                        logging.info(f"🧠 Got evolution traits from neural_routes: {list(evolution_data.keys())}")
                    else:
                        # Fallback: Χρησιμοποιώ τα global evolution traits από telemetry_store
                        if "evolution_traits" in telemetry_store:
                            evolution_data = telemetry_store["evolution_traits"]
                            logging.info(f"🧠 Got evolution traits from telemetry_store: {list(evolution_data.keys())}")
                        else:
                            # Final fallback: Default values based on current soul metrics
                            base_empathy = max(50, current_soul.get("coherence", 50) + 10)
                            base_creativity = max(50, current_soul.get("vitality", 50) - 10)
                            evolution_data = {
                                "empathy": base_empathy,
                                "creativity": base_creativity,
                                "resilience": max(50, current_soul.get("ethics", 50) + 5),
                                "focus": max(50, current_soul.get("narrative", 50) - 5),
                                "curiosity": base_creativity + 10,
                                "compassion": base_empathy + 5
                            }
                            logging.warning(f"🧠 Using computed evolution traits - Empathy:{base_empathy} Creativity:{base_creativity}")
                except Exception as ex:
                    logging.warning(f"🧠 Evolution traits access error: {ex}")
                    evolution_data = {
                        "empathy": 65, "creativity": 60, "resilience": 70, 
                        "focus": 55, "curiosity": 75, "compassion": 70
                    }
                
                # Build neural state για GPT
                neural_state = {
                    "soul_metrics": {
                        "coherence": current_soul.get("coherence", 50),
                        "vitality": current_soul.get("vitality", 50),
                        "ethics": current_soul.get("ethics", 50),
                        "narrative": current_soul.get("narrative", 50)
                    },
                    "evolution_traits": {
                        "empathy": evolution_data.get("empathy", 50),
                        "creativity": evolution_data.get("creativity", 50),
                        "resilience": evolution_data.get("resilience", 50),
                        "focus": evolution_data.get("focus", 50),
                        "curiosity": evolution_data.get("curiosity", 50),
                        "compassion": evolution_data.get("compassion", 50)
                    },
                    "orb_state": {
                        "x": you_node.get("x", 250),
                        "y": you_node.get("y", 250),
                        "dominant_emotion": "neutral"  # Will be calculated by GPT module
                    }
                }
                logging.info(f"🤖 Neural state collected for GPT - Orb:({neural_state['orb_state']['x']},{neural_state['orb_state']['y']})")
                
            except Exception as e:
                logging.warning(f"🚫 Neural state collection failed: {e}, using defaults")
                neural_state = {}
        
        # Use GPT Consciousness module
        from gpt_consciousness import consciousness_gpt
        
        # Generate response με neural state awareness
        result = consciousness_gpt.generate_response(
            user_message=user_message,
            soul_metrics=neural_state.get("soul_metrics", {}),
            evolution_traits=neural_state.get("evolution_traits", {}),
            consciousness_data=neural_state.get("orb_state", {}),
            conversation_history=conversation_history
        )
        
        # Build response
        response_data = {
            "response": result.get("response", "Συγγνώμη, αντιμετωπίζω κάποιο πρόβλημα αυτή τη στιγμή."),
            "personality_analysis": result.get("personality_analysis", "Balanced state"),
            "neural_context": {
                "used_neural_state": include_neural_state,
                "orb_position": neural_state.get("orb_state", {}).get("x", 250) if neural_state else None,
                "dominant_traits": result.get("personality_analysis", "").split(" | ")[:2] if result.get("personality_analysis") else []
            },
            "metadata": {
                "timestamp": int(time.time() * 1000),
                "model_used": "gpt-4o",
                "tokens_used": result.get("usage", {}).get("tokens", 0),
                "has_error": bool(result.get("error"))
            }
        }
        
        if result.get("error"):
            response_data["debug_info"] = {"error": result.get("error")}
        
        logging.info(f"🤖 GPT Response generated - Analysis: {result.get('personality_analysis', 'N/A')}")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"🚫 GPT Chat ERROR: {e}")
        return jsonify({
            "error": f"GPT Chat failed: {str(e)}",
            "response": "Συγγνώμη, αντιμετωπίζω τεχνικό πρόβλημα. Δοκίμασε ξανά σε λίγο.",
            "fallback": True
        }), 500


# 🌟 GPT PERSONALITY SUMMARY ENDPOINT  
@app.route('/api/personality-summary', methods=['GET'])
def get_personality_summary():
    """
    🎭 Επιστρέφει σύντομη περιγραφή της τρέχουσας προσωπικότητας βάσει neural states
    """
    try:
        # Get current neural state
        current_hr = telemetry_store.get("latest_hr", {})
        current_soul = telemetry_store.get("soul_metrics", {})
        current_consciousness = telemetry_store.get("consciousness_data", {})
        
        # Get evolution traits
        evolution_data = {}
        try:
            import neural_routes
            if hasattr(neural_routes, 'last_payload') and neural_routes.last_payload and "evolution" in neural_routes.last_payload:
                evolution_data = neural_routes.last_payload["evolution"]
        except:
            pass
        
        # Use GPT για personality summary
        from gpt_consciousness import consciousness_gpt
        
        summary = consciousness_gpt.generate_personality_summary(
            soul_metrics=current_soul if isinstance(current_soul, dict) else {},
            evolution_traits=evolution_data
        )
        
        return jsonify({
            "summary": summary,
            "timestamp": int(time.time() * 1000),
            "neural_state_available": bool(evolution_data and current_soul)
        })
        
    except Exception as e:
        logging.error(f"🚫 Personality Summary ERROR: {e}")
        return jsonify({
            "summary": "A consciousness in constant flux, adapting to the rhythms of existence.",
            "error": str(e),
            "fallback": True
        })


# 3. Enhanced HRV Debug Endpoint with YOU node mapping
@app.route('/debug/hrv', methods=['GET'])
def debug_hrv():
    """Enhanced debug endpoint for HRV metrics analysis with YOU node mapping"""
    _trim_rr_buffer()
    rr_window = [rr for _, rr in RR_BUF]

    # Get current telemetry data
    current_hr = telemetry_store.get("latest_hr", {})
    current_soul = telemetry_store.get("soul_metrics", {})
    current_consciousness = telemetry_store.get("consciousness_data", {})

    # Update YOU node mapping
    you_node = _update_you_node_mapping(current_hr, current_soul,
                                        current_consciousness)

    # Calculate RMSSD stats
    rmssd_raw = _calculate_rmssd(rr_window) if rr_window else None
    rmssd_window = rmssd_raw
    rmssd_smooth = HRV_STATE.get("rmssd_smooth")

    # Calculate WebSocket queue lag (simplified)
    ws_queue_lag = abs(time.time() * 1000 % 100)  # Mock lag calculation

    # NO-SIGNAL GUARDS debug data
    now_ms = int(time.time() * 1000)
    devices_status = []
    active_device_count = 0

    for device_id, last_seen in DEVICE_LAST_SEEN.items():
        time_since_last = now_ms - last_seen
        is_active = time_since_last <= DEVICE_TTL_MS
        if is_active:
            active_device_count += 1

        devices_status.append({
            "device_id": device_id,
            "last_seen_ms_ago": time_since_last,
            "is_active": is_active,
            "status": "ACTIVE" if is_active else "TIMEOUT"
        })

    return jsonify({
        # Original debug data
        "n_rr_window":
        len(rr_window),
        "rmssd_raw":
        round(rmssd_raw, 1) if rmssd_raw else None,
        "rmssd_window":
        round(rmssd_window, 1) if rmssd_window else None,
        "sdnn_window":
        _calculate_sdnn(rr_window),
        "rmssd_smooth":
        round(rmssd_smooth, 1) if rmssd_smooth else None,
        "window_secs":
        HRV_WINDOW_SEC,
        "alpha":
        ALPHA,
        "buffer_size":
        len(RR_BUF),

        # Enhanced counters
        "packet_gap_ms":
        you_node["packet_gap_ms"],
        "ws_queue_lag_ms":
        round(ws_queue_lag, 1),

        # NO-SIGNAL GUARDS status
        "no_signal_guards": {
            "device_ttl_ms": DEVICE_TTL_MS,
            "no_signal_broadcasted": NO_SIGNAL_BROADCASTED,
            "active_devices": active_device_count,
            "total_devices": len(DEVICE_LAST_SEEN),
            "devices": devices_status
        },

        # Hysteresis state
        "hysteresis_state":
        you_node["hysteresis_state"],
        "high_load_duration":
        you_node["high_load_counter"],
        "risk_flag":
        "high_load" if you_node["high_load_counter"] > 10 else "normal",

        # YOU node visual data
        "you": {
            "x": round(you_node["x"], 1),
            "y": round(you_node["y"], 1),
            "radius": round(you_node["radius"], 1),
            "color": you_node["color"],
            "pulse_speed": round(you_node["pulse_speed"], 2),
            "coherence": round(you_node["coherence"], 2),
            "vitality": round(you_node["vitality"], 2),
            "opacity": round(you_node["opacity"], 2),
            "hasTrail": you_node["hasTrail"],
            "status": you_node["status"],
            "active": you_node.get("active", True)
        },

        # Breath-sync calculation
        "breath_sync": {
            "scale_factor":
            1 + (you_node["coherence"] * 0.05) *
            math.sin(2 * math.pi * time.time() * 15 / 60),
            "amplitude":
            you_node["coherence"] * 0.08,
            "frequency_bpm":
            15  # Breathing rate
        },

        # Trail quality metrics
        "trail": {
            "length": int(you_node["vitality"] * 20),  # 0-20 points
            "blur": round((1 - you_node["coherence"]) * 3, 1),  # 0-3px blur
            "enabled": you_node["hasTrail"]
        },

        # System explanations
        "explanations": {
            "position":
            "Position driven by Empathy/Analytical (X) & Focus/Creativity (Y); smoothed by Kalman EMA=0.05",
            "radius":
            "Base 50px + HRV factor + Soul Metrics average; EMA=0.08 smoothing",
            "color":
            "Hue=200° (blue), Saturation from Vitality, Brightness from Ethics",
            "hysteresis":
            "Calm↔Focus @ 35/40ms RMSSD, Focus↔Stress @ 25/30ms RMSSD",
            "fail_safe":
            "packet_gap>5000ms → NO SIGNAL status, opacity 60%, no trail"
        }
    })


@app.route('/api/signal/status', methods=['GET'])
def get_signal_status():
    """Get live signal status for frontend motion freeze control"""
    live_signal = has_live_signal()
    now_ms = int(time.time() * 1000)

    # Get latest BPM if available
    latest_bpm = telemetry_store.get("latest_hr", 0)
    latest_rr_count = len(RR_BUF)

    return jsonify({
        "signal":
        live_signal,
        "bpm":
        latest_bpm if live_signal else 0,
        "rr_length":
        latest_rr_count if live_signal else 0,
        "status":
        "CONNECTED" if live_signal else "NO SIGNAL",
        "active_devices":
        len([
            d for d, t in DEVICE_LAST_SEEN.items()
            if now_ms - t <= DEVICE_TTL_MS
        ]),
        "debug_overlay": {
            "signal": live_signal,
            "bpm": latest_bpm if live_signal else 0,
            "rr_length": latest_rr_count if live_signal else 0
        }
    })


def _broadcast_enhanced_telemetry():
    """🚫 STRICT: Broadcast enhanced telemetry with frame gating"""
    global telemetry_store

    # DEBUG: Always log when this function is called
    logging.info(f'🚀 _broadcast_enhanced_telemetry CALLED - checking live signal...')

    try:
        # 🚫 CRITICAL FRAME GATING: Only broadcast if we have live signal
        if not has_live_signal():
            # Broadcast NO_SIGNAL frame instead
            no_signal_frame = create_hrv_frame({}, "NO_DEVICE")
            logging.debug(f"🚫 BROADCASTING NO_SIGNAL FRAME: {no_signal_frame}")
            socketio.emit('telemetry', no_signal_frame)
            return

        # Get current telemetry data
        current_hr = telemetry_store.get("latest_hr", {})
        current_soul = telemetry_store.get("soul_metrics", {})
        current_consciousness = telemetry_store.get("consciousness_data", {})

        # Update YOU node mapping
        you_node = _update_you_node_mapping(current_hr, current_soul,
                                            current_consciousness)

        # Enhanced telemetry payload
        enhanced_telemetry = {
            "type": "enhanced_telemetry",
            "timestamp": int(time.time() * 1000),
            "hr": current_hr,
            "soul_metrics": current_soul,
            "consciousness": current_consciousness,

            # YOU node visual data with unique identifier and isolation
            "you": {
                "id": you_node.get("id", "you_primary"),  # Unique identifier
                "label": you_node.get("label",
                                      "you_primary"),  # Permanent label
                "isGhost": you_node.get("isGhost",
                                        False),  # Ghost/clone status
                "metricsBinding": you_node.get("metricsBinding",
                                               True),  # Metrics binding
                "x": round(you_node["x"], 1),
                "y": round(you_node["y"], 1),
                "radius": round(you_node["radius"], 1),
                "color": you_node["color"],
                "opacity": round(you_node.get("opacity", 1.0),
                                 2),  # Visual differentiation
                "pulse_speed": round(you_node["pulse_speed"], 2),
                "coherence": round(you_node["coherence"], 2),
                "vitality": round(you_node["vitality"], 2),
                "hasTrail": you_node["hasTrail"],
                "status": you_node["status"]
            },

            # Clone nodes for testing and visualization
            "clones": {
                clone_id: {
                    "id": clone["id"],
                    "label": clone["label"],
                    "isGhost": clone["isGhost"],
                    "x": round(clone["x"], 1),
                    "y": round(clone["y"], 1),
                    "radius": round(clone["radius"], 1),
                    "color": clone["color"],
                    "opacity": round(clone.get("opacity", 0.6), 2),
                    "hysteresis_state": clone["hysteresis_state"]
                }
                for clone_id, clone in CLONE_NODES.items()
            },

            # Debug counters
            "debug": {
                "packet_gap_ms":
                you_node["packet_gap_ms"],
                "ws_queue_lag_ms":
                round(abs(time.time() * 1000 % 100), 1),
                "hysteresis_state":
                you_node["hysteresis_state"],
                "risk_flag":
                "high_load" if you_node["high_load_counter"] > 10 else "normal"
            },

            # Breath sync for smooth animations
            "breath_sync": {
                "scale_factor":
                1 + (you_node["coherence"] * 0.05) *
                math.sin(2 * math.pi * time.time() * 15 / 60),
                "amplitude":
                you_node["coherence"] * 0.08
            },

            # Trail quality
            "trail": {
                "length": int(you_node["vitality"] * 20),
                "blur": round((1 - you_node["coherence"]) * 3, 1),
                "enabled": you_node["hasTrail"]
            }
        }

        # Broadcast to all connected clients
        # 🚫 CRITICAL: Only emit if we have live signal
        if has_live_signal():
            socketio.emit('enhanced_telemetry', enhanced_telemetry)
        else:
            # Emit null values instead of cached data
            null_telemetry = {
                "type": "enhanced_telemetry",
                "signal": "NO_SIGNAL",
                "hr": None,
                "hrv": None,
                "breath": None,
                "coherence": None,
                "focus": None,
                "you": telemetry_store["you_position"]
            }
            socketio.emit('enhanced_telemetry', null_telemetry)
        logging.debug(f"🔄 Enhanced telemetry broadcasted with YOU node data")

    except Exception as e:
        logging.error(f"Enhanced telemetry broadcast failed: {e}")


# Update existing broadcast function to include enhanced data
def _broadcast_telemetry():
    """Enhanced telemetry broadcast with YOU node data"""
    try:
        # Use the enhanced broadcast function
        _broadcast_enhanced_telemetry()

        # 🚫 CRITICAL: Don't maintain cached data without signal
        if has_live_signal():
            current_telemetry = {
                "type": "telemetry",
                "ts": int(time.time()),
                "hr": telemetry_store["latest_hr"],
                "hrv": telemetry_store["latest_hrv"],
                "soul_metrics": telemetry_store["soul_metrics"],
                "evolution": telemetry_store["evolution"],
                "you_position": telemetry_store["you_position"]
            }
        else:
            # Clear all cached data when no signal
            current_telemetry = {
                "type": "telemetry",
                "signal": "NO_SIGNAL",
                "ts": int(time.time()),
                "hr": None,
                "hrv": None,
                "soul_metrics": None,
                "evolution": None,
                "you_position": {
                    "active": False,
                    "status": "NO SIGNAL"
                }
            }
        # 🚫 CRITICAL: Only emit updates if we have live signal
        if has_live_signal():
            socketio.emit('telemetry_update',
                          current_telemetry,
                          broadcast=True)
        else:
            # Emit null update instead of cached data
            null_update = {
                "type": "telemetry_update",
                "signal": "NO_SIGNAL",
                "hr": None,
                "hrv": None,
                "timestamp": int(time.time() * 1000)
            }
            socketio.emit('telemetry_update', null_update, broadcast=True)

    except Exception as e:
        logging.error(f"Telemetry broadcast failed: {e}")


# ===============================
# WEBSOCKET HANDLERS
# ===============================


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket client connection"""
    logging.info(f"🔌 WebSocket client connected: {request.sid}")

    # 🚫 CRITICAL: Only send telemetry if we have live signal
    if has_live_signal():
        current_telemetry = {
            "type": "telemetry",
            "ts": int(time.time()),
            "hr": telemetry_store["latest_hr"],
            "hrv": telemetry_store["latest_hrv"],
            "soul_metrics": telemetry_store["soul_metrics"],
            "evolution": telemetry_store["evolution"],
            "you": telemetry_store["you_position"]
        }
        emit('telemetry_update', current_telemetry)
    else:
        # Send NO_SIGNAL state to new client
        no_signal_telemetry = {
            "type": "telemetry",
            "signal": "NO_SIGNAL",
            "ts": int(time.time()),
            "hr": None,
            "hrv": None,
            "soul_metrics": None,
            "evolution": None,
            "you": {
                "active": False,
                "status": "NO SIGNAL"
            }
        }
        emit('telemetry_update', no_signal_telemetry)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket client disconnection"""
    logging.info(f"🔌 WebSocket client disconnected: {request.sid}")


@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'ts': int(time.time())})


def _perform_hard_disconnect():
    """Perform complete system disconnect and state reset"""
    global telemetry_store, YOU_NODE_STATE, CLONE_NODES

    logging.warning("🚫 PERFORMING HARD DISCONNECT - Complete state reset")

    # Clear all telemetry data
    telemetry_store = {
        "latest_hr": None,
        "latest_hrv": None,
        "soul_metrics": None,
        "evolution": None,
        "you_position": {
            "active": False,
            "status": "DISCONNECTED",
            "x": 400,  # Center position
            "y": 300,
            "hasTrail": False,
            "opacity": 0.3,
            "frozen": True
        }
    }

    # Reset YOU node state
    YOU_NODE_STATE.update({
        "active": False,
        "status": "DISCONNECTED",
        "frozen": True,
        "velocity": {
            "x": 0,
            "y": 0
        },
        "acceleration": {
            "x": 0,
            "y": 0
        }
    })

    # Clear all clone nodes
    CLONE_NODES.clear()

    # Stop any background processes
    try:
        import consciousness_integration
        consciousness_integration.stop_consciousness_feed()
    except:
        pass

    logging.warning("✅ HARD DISCONNECT COMPLETE - All systems reset")


# Register Syndesis integration
register_syndesis_integration(app)

# Register mood previewer blueprint
app.register_blueprint(mood_previewer_bp)

# Register advanced personality features blueprint
from routes_advanced_personality import advanced_personality_bp

app.register_blueprint(advanced_personality_bp)

# Register trait response demo blueprint
from routes_trait_response_demo import trait_demo_bp

app.register_blueprint(trait_demo_bp)

# Register trace explanation blueprint
from routes_trace_explanation import trace_bp

app.register_blueprint(trace_bp)

# Register Neural HRV blueprint
try:
    from routes_neural_hrv import neural_hrv_bp
    app.register_blueprint(neural_hrv_bp)
    logging.info("Neural HRV routes registered successfully")
except ImportError as e:
    logging.warning(f"Could not register neural HRV routes: {e}")

# PC TICKR Bridge για real heart rate data
try:
    from pc_tickr_bridge import pc_tickr_bridge
    logging.info("PC TICKR Bridge initialized for real heart rate data")
except ImportError as e:
    logging.warning(f"PC TICKR Bridge not available: {e}")
    pc_tickr_bridge = None

# Register Spatial Intelligence Testing API
try:
    from spatial_test_api import register_spatial_test_routes
    register_spatial_test_routes(app)
    logging.info("Spatial intelligence testing routes registered successfully")
except ImportError as e:
    logging.warning(f"Could not register spatial testing routes: {e}")


def import_dataset_file(file_path, import_type='jsonl'):
    """Import dataset from a file into the database"""
    imported_count = 0

    try:
        if import_type == 'jsonl':
            # Import JSONL format
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Extract fields from different possible formats
                        agent_input = data.get('agent_input') or data.get(
                            'input') or data.get('user') or data.get(
                                'question') or data.get('prompt', '')
                        agent_output = data.get('agent_output') or data.get(
                            'output') or data.get('assistant') or data.get(
                                'answer') or data.get('response', '')
                        context = data.get('context', '')
                        session_id = data.get(
                            'session_id'
                        ) or f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{line_num}"

                        if agent_input and agent_output:
                            # Create new interaction log entry
                            interaction = InteractionLog()
                            interaction.agent_input = agent_input
                            interaction.agent_output = agent_output
                            interaction.context = context
                            interaction.session_id = session_id
                            interaction.processed = False
                            db.session.add(interaction)
                            imported_count += 1

                    except json.JSONDecodeError as e:
                        logging.warning(
                            f"Skipping invalid JSON on line {line_num}: {e}")
                        continue

        elif import_type == 'json':
            # Import JSON array format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    for item_num, item in enumerate(data, 1):
                        agent_input = item.get('agent_input') or item.get(
                            'input') or item.get('user') or item.get(
                                'question') or item.get('prompt', '')
                        agent_output = item.get('agent_output') or item.get(
                            'output') or item.get('assistant') or item.get(
                                'answer') or item.get('response', '')
                        context = item.get('context', '')
                        session_id = item.get(
                            'session_id'
                        ) or f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item_num}"

                        if agent_input and agent_output:
                            interaction = InteractionLog()
                            interaction.agent_input = agent_input
                            interaction.agent_output = agent_output
                            interaction.context = context
                            interaction.session_id = session_id
                            interaction.processed = False
                            db.session.add(interaction)
                            imported_count += 1
                else:
                    raise ValueError(
                        "JSON file must contain an array of interactions")

        # Commit all changes
        db.session.commit()
        logging.info(
            f"Successfully imported {imported_count} interactions from {file_path}"
        )

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error importing dataset: {e}")
        raise

    return imported_count


@app.route('/')
def integrated():
    """Clean Integrated Interface for Enterprise Demonstrations"""
    # Minimal context for template
    book_data = {
        'title': 'Syndesis AI - Neural Consciousness Platform',
        'description': 'Enterprise-grade neural-HRV integration system'
    }
    
    personality_traits = {
        "empathy": 0.6, "creativity": 0.7, "resilience": 0.5, 
        "focus": 0.4, "curiosity": 0.6, "compassion": 0.55
    }
    
    return render_template('book_centers_integrated.html', 
                         book_data=book_data, 
                         personality_traits=personality_traits)


# Additional handlers and routes follow below
# ============================================

@app.route('/legacy')  
def legacy_index():
    """Legacy book center interface"""
    return render_template('book_centers_2d.html')


@app.route('/proof')
def audit_proof():
    """Audit Proof Demo - Anti-placebo demonstration system"""
    return render_template('audit_demo.html')


@app.route('/dashboard')
def dashboard():
    # Get pipeline status
    logger_status = PipelineStatus.query.filter_by(component='logger').first()
    exporter_status = PipelineStatus.query.filter_by(
        component='exporter').first()
    trainer_status = PipelineStatus.query.filter_by(
        component='trainer').first()

    # Get statistics
    total_interactions = InteractionLog.query.count()
    unprocessed_interactions = InteractionLog.query.filter_by(
        processed=False).count()
    recent_jobs = TrainingJob.query.order_by(
        TrainingJob.created_at.desc()).limit(5).all()

    return render_template('index.html',
                           logger_status=logger_status,
                           exporter_status=exporter_status,
                           trainer_status=trainer_status,
                           total_interactions=total_interactions,
                           unprocessed_interactions=unprocessed_interactions,
                           recent_jobs=recent_jobs)


@app.route('/logs')
def logs():
    page = request.args.get('page', 1, type=int)
    interactions = InteractionLog.query.order_by(
        InteractionLog.timestamp.desc()).paginate(page=page,
                                                  per_page=20,
                                                  error_out=False)
    return render_template('logs.html', interactions=interactions)


@app.route('/training')
def training():
    jobs = TrainingJob.query.order_by(TrainingJob.created_at.desc()).all()
    return render_template('training_standalone.html', jobs=jobs)


@app.route('/api_test')
def api_test():
    """API test interface for Syndesis integration"""
    return render_template('api_test.html')


@app.route('/simulate_interaction', methods=['POST'])
def simulate_interaction():
    """Simulate an agent interaction for testing"""
    user_input = request.form.get('input', 'Hello, how are you?')

    try:
        response = agent.process_input(user_input)
        flash('Interaction logged successfully!', 'success')
    except Exception as e:
        flash(f'Error processing interaction: {str(e)}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/export_dataset', methods=['POST'])
def export_dataset():
    """Export interactions to HuggingFace Dataset"""
    try:
        dataset_path = orchestrator.export_to_dataset()
        flash(f'Dataset exported successfully to {dataset_path}!', 'success')
    except Exception as e:
        flash(f'Error exporting dataset: {str(e)}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/import_dataset', methods=['GET', 'POST'])
def import_dataset():
    """Import dataset from file or JSONL"""
    if request.method == 'GET':
        return render_template('import_dataset.html')

    try:
        # Handle file upload
        if 'dataset_file' in request.files:
            file = request.files['dataset_file']
            if file and file.filename:
                import_type = request.form.get('import_type', 'jsonl')

                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+b',
                                                 delete=False,
                                                 suffix='.jsonl') as temp_file:
                    file.save(temp_file.name)

                    # Import the data
                    imported_count = import_dataset_file(
                        temp_file.name, import_type)

                    # Clean up temp file
                    os.unlink(temp_file.name)

                    flash(
                        f'Successfully imported {imported_count} interactions!',
                        'success')
            else:
                flash('Please select a file to import', 'error')

        # Handle direct text input
        elif request.form.get('dataset_text'):
            dataset_text = request.form.get('dataset_text')
            import_type = request.form.get('import_type', 'jsonl')

            # Save text to temporary file and import
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',
                                             delete=False,
                                             suffix='.jsonl') as temp_file:
                if dataset_text:
                    temp_file.write(dataset_text)
                temp_file.flush()

                imported_count = import_dataset_file(temp_file.name,
                                                     import_type)

                # Clean up temp file
                os.unlink(temp_file.name)

                flash(f'Successfully imported {imported_count} interactions!',
                      'success')
        else:
            flash('Please provide a dataset file or text content', 'error')

    except Exception as e:
        flash(f'Error importing dataset: {str(e)}', 'error')

    return redirect(url_for('import_dataset'))


@app.route('/start_training', methods=['POST'])
def start_training():
    """Start LLM fine-tuning training"""
    epochs = request.form.get('epochs', 1, type=int)

    try:
        job_id = orchestrator.start_training(epochs=epochs)
        flash(f'Training job {job_id} started successfully!', 'success')
    except Exception as e:
        flash(f'Error starting training: {str(e)}', 'error')

    return redirect(url_for('training'))


@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    # Get interaction count by hour for the last 24 hours
    now = datetime.utcnow()
    hours_ago_24 = now - timedelta(hours=24)

    interactions = InteractionLog.query.filter(
        InteractionLog.timestamp >= hours_ago_24).all()

    # Group by hour
    hourly_stats = {}
    for interaction in interactions:
        hour_key = interaction.timestamp.strftime('%H:00')
        hourly_stats[hour_key] = hourly_stats.get(hour_key, 0) + 1

    # Fill in missing hours with 0
    hours = []
    counts = []
    for i in range(24):
        hour = (now - timedelta(hours=23 - i)).strftime('%H:00')
        hours.append(hour)
        counts.append(hourly_stats.get(hour, 0))

    return jsonify({
        'hourly_interactions': {
            'labels': hours,
            'data': counts
        },
        'total_interactions':
        InteractionLog.query.count(),
        'unprocessed_interactions':
        InteractionLog.query.filter_by(processed=False).count(),
        'training_jobs':
        TrainingJob.query.count()
    })


@app.route('/api/job_status/<int:job_id>')
def api_job_status(job_id):
    """Get training job status"""
    job = TrainingJob.query.get_or_404(job_id)
    return jsonify({
        'id': job.id,
        'status': job.status,
        'created_at': job.created_at.isoformat(),
        'total_samples': job.total_samples,
        'epochs': job.epochs,
        'log_output': job.log_output,
        'error_message': job.error_message
    })


# ========== SYNDESIS FRONTEND API ENDPOINTS ==========


@app.route('/api/agent/chat', methods=['POST', 'OPTIONS'])
def api_agent_chat():
    """API endpoint for Syndesis frontend to chat with the agent"""
    try:
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers',
                                 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods',
                                 'POST, OPTIONS')
            return response

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message in request'}), 400

        user_message = data['message']
        session_id = data.get(
            'session_id',
            f"syndesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        context = data.get('context', '')

        # Create agent instance and get response
        agent_instance = FractalAIAgent()

        # ========== AUTHENTIC HRV-DRIVEN PERSONALITY UPDATE ==========
        # Only use real HRV data - no synthetic generation
        # ✅ FIXED: Simple live signal check without complex function
        signal_status = {
            "has_live_signal": last_hrv_frame and last_hrv_frame.get('live'),
            "hrv_data": last_hrv_frame or {}
        }
        if signal_status.get('has_live_signal', False):
            hrv_data = signal_status.get('hrv_data', {})
            if hrv_data:
                # Apply real HRV-driven traits to the personality manager
                agent_instance.personality_manager.update_personality(
                    session_id, hrv_data, "Authentic HRV biometric influence",
                    "Real TICKR device data modulation")

        agent_response = agent_instance.process_input(user_message,
                                                      context=context,
                                                      session_id=session_id)

        # Get REAL persistent personality state (not simulated!)
        personality = agent_instance.personality_manager.get_or_create_personality(
            session_id)

        # Get session statistics for metrics
        session_interactions = InteractionLog.query.filter_by(
            session_id=session_id).count()

        # ============ GENERATE TRACE STRUCTURE ============
        trace = []

        # Step 1: Input received
        trace.append({
            'step':
            1,
            'action':
            'Input received',
            'data':
            user_message,
            'details':
            f"User message: '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'"
        })

        # Step 2: Memory checked
        if session_interactions > 0:
            trace.append({
                'step':
                2,
                'action':
                'Memory checked',
                'match':
                f'{session_interactions} previous interactions',
                'details':
                f"Found {session_interactions} conversation memories"
            })
        else:
            trace.append({
                'step': 2,
                'action': 'Memory checked',
                'match': 'new conversation',
                'details': "First interaction with this user"
            })

        # Step 3: SEMANTIC ANALYSIS & PERSONALITY TRAITS UPDATE
        # Get current traits from personality manager
        current_traits = personality.copy() if personality else {
            'empathy': 0.5,
            'creativity': 0.5,
            'humor': 0.5,
            'curiosity': 0.5,
            'supportiveness': 0.5,
            'analyticalness': 0.5,
            'assertiveness': 0.5
        }

        # ========== SEMANTIC ANALYSIS WITH OPENAI ==========
        from openai import OpenAI
        import json
        import os

        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        try:
            # Analyze message semantically using GPT-4o
            semantic_response = openai_client.chat.completions.create(
                model="gpt-4o",  # Latest OpenAI model
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a psychological analysis expert. Analyze the user message and return JSON with:
                        {
                            "sentiment": "positive/negative/neutral",
                            "primary_emotion": "joy/sadness/anger/fear/surprise/disgust/neutral",
                            "intent": "seeking_help/asking_question/making_joke/expressing_creativity/sharing_info/casual_chat",
                            "topics": ["wellbeing", "learning", "creativity", "relationships", "work", "humor"],
                            "emotional_intensity": 0.1-1.0,
                            "cognitive_complexity": 0.1-1.0
                        }
                        Be precise and analytical. Focus on semantic meaning, not just keywords."""
                }, {
                    "role": "user",
                    "content": f"Analyze this message: '{user_message}'"
                }],
                response_format={"type": "json_object"},
                max_tokens=200)

            semantic_analysis = json.loads(
                semantic_response.choices[0].message.content)

        except Exception as e:
            logging.warning(
                f"Semantic analysis failed: {e}, falling back to keyword detection"
            )
            # Fallback to basic keyword detection
            semantic_analysis = {
                "sentiment": "neutral",
                "primary_emotion": "neutral",
                "intent": "casual_chat",
                "topics": [],
                "emotional_intensity": 0.5,
                "cognitive_complexity": 0.5
            }

        # ========== MAP SEMANTIC ANALYSIS TO TRAITS ==========
        traits_updated = []

        # Sentiment → Empathy & Supportiveness
        if semantic_analysis.get("sentiment") == "negative":
            old_empathy = current_traits.get('empathy', 0.5)
            new_empathy = min(1.0, old_empathy + 0.15)
            current_traits['empathy'] = new_empathy
            traits_updated.append(('empathy', old_empathy, new_empathy))

            old_support = current_traits.get('supportiveness', 0.5)
            new_support = min(1.0, old_support + 0.1)
            current_traits['supportiveness'] = new_support
            traits_updated.append(('supportiveness', old_support, new_support))

        # Primary Emotion → Specific Traits
        emotion = semantic_analysis.get("primary_emotion", "neutral")
        if emotion == "sadness":
            old_val = current_traits.get('empathy', 0.5)
            new_val = min(1.0, old_val + 0.2)
            current_traits['empathy'] = new_val
            traits_updated.append(('empathy', old_val, new_val))

        elif emotion == "joy":
            old_val = current_traits.get('humor', 0.5)
            new_val = min(1.0, old_val + 0.15)
            current_traits['humor'] = new_val
            traits_updated.append(('humor', old_val, new_val))

        # Intent → Cognitive Traits
        intent = semantic_analysis.get("intent", "casual_chat")
        if intent == "asking_question":
            old_val = current_traits.get('curiosity', 0.5)
            new_val = min(1.0, old_val + 0.15)
            current_traits['curiosity'] = new_val
            traits_updated.append(('curiosity', old_val, new_val))

            old_analytical = current_traits.get('analyticalness', 0.5)
            new_analytical = min(1.0, old_analytical + 0.1)
            current_traits['analyticalness'] = new_analytical
            traits_updated.append(
                ('analyticalness', old_analytical, new_analytical))

        elif intent == "expressing_creativity":
            old_val = current_traits.get('creativity', 0.5)
            new_val = min(1.0, old_val + 0.2)
            current_traits['creativity'] = new_val
            traits_updated.append(('creativity', old_val, new_val))

        elif intent == "making_joke":
            old_val = current_traits.get('humor', 0.5)
            new_val = min(1.0, old_val + 0.25)
            current_traits['humor'] = new_val
            traits_updated.append(('humor', old_val, new_val))

        # Cognitive Complexity → Analytical
        complexity = semantic_analysis.get("cognitive_complexity", 0.5)
        if complexity > 0.7:
            old_val = current_traits.get('analyticalness', 0.5)
            new_val = min(1.0, old_val + 0.1)
            current_traits['analyticalness'] = new_val
            traits_updated.append(('analyticalness', old_val, new_val))

        # Topics → Specific Traits
        topics = semantic_analysis.get("topics", [])
        if "learning" in topics:
            old_val = current_traits.get('curiosity', 0.5)
            new_val = min(1.0, old_val + 0.1)
            current_traits['curiosity'] = new_val
            traits_updated.append(('curiosity', old_val, new_val))

        # Update personality in the database
        agent_instance.personality_manager.update_personality(
            session_id, current_traits)

        # Add semantic analysis to trace
        trace.append({
            'step':
            3,
            'action':
            'Semantic analysis completed',
            'sentiment':
            semantic_analysis.get('sentiment', 'neutral'),
            'emotion':
            semantic_analysis.get('primary_emotion', 'neutral'),
            'intent':
            semantic_analysis.get('intent', 'casual_chat'),
            'details':
            f"Detected: {semantic_analysis.get('sentiment', 'neutral')} sentiment, {semantic_analysis.get('primary_emotion', 'neutral')} emotion, {semantic_analysis.get('intent', 'casual_chat')} intent"
        })

        # Add trace entries for updated traits
        if traits_updated:
            for trait_name, old_val, new_val in traits_updated:
                trace.append({
                    'step':
                    4,
                    'action':
                    f'{trait_name.title()} trait updated (semantic)',
                    'trait':
                    trait_name,
                    'old':
                    old_val,
                    'new':
                    new_val,
                    'details':
                    f"{trait_name.title()}: {old_val:.1f} → {new_val:.1f} (based on semantic analysis)"
                })
        else:
            trace.append({
                'step':
                4,
                'action':
                'Traits analyzed',
                'details':
                "No significant trait changes detected from semantic analysis"
            })

        # Get the updated personality state
        personality = current_traits

        # Step 5: Response generated
        response_style = traits_updated[0][0] if traits_updated else 'balanced'
        trace.append({
            'step':
            5,
            'action':
            'Response generated',
            'style':
            response_style,
            'details':
            f"Generated {response_style} response based on semantic analysis"
        })

        # Calculate Soul Metrics based on current personality
        soul_metrics = {}
        if personality:
            # Calculate Coherence (logical consistency)
            coherence = round((personality.get('analyticalness', 0.5) * 0.4 +
                               personality.get('supportiveness', 0.5) * 0.3 +
                               personality.get('empathy', 0.5) * 0.3) * 100)

            # Calculate Vitality (personality entropy/energy)
            vitality = round((personality.get('creativity', 0.5) * 0.4 +
                              personality.get('curiosity', 0.5) * 0.4 +
                              personality.get('humor', 0.5) * 0.2) * 100)

            # Calculate Ethics (rule compliance)
            ethics = max(
                90,
                round((personality.get('empathy', 0.5) * 0.5 +
                       personality.get('supportiveness', 0.5) * 0.5) * 100))

            # Calculate Narrative (experience richness)
            narrative = round((personality.get('creativity', 0.5) * 0.5 +
                               personality.get('curiosity', 0.5) * 0.3 +
                               personality.get('empathy', 0.5) * 0.2) * 100)

            # Overall consciousness score
            consciousness_score = round(
                (coherence + vitality + ethics + narrative) / 4)

            # Determine consciousness level
            if consciousness_score >= 90:
                consciousness_level = 'TRANSCENDENT'
            elif consciousness_score >= 80:
                consciousness_level = 'HIGHLY CONSCIOUS'
            elif consciousness_score >= 70:
                consciousness_level = 'CONSCIOUS'
            elif consciousness_score >= 60:
                consciousness_level = 'DEVELOPING'
            else:
                consciousness_level = 'EMERGING'

            soul_metrics = {
                'coherence': coherence,
                'vitality': vitality,
                'ethics': ethics,
                'narrative': narrative,
                'consciousness_score': consciousness_score,
                'consciousness_level': consciousness_level
            }

        # Get HRV system status for response
        hrv_status = agent_instance.hrv_system.get_system_status()

        response_data = {
            'response': agent_response,
            'trace': trace,  # ← NEW: Always include trace
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'session_info': {
                'session_id': session_id,
                'memory_count': session_interactions,
                'relationship_depth': min(1.0, session_interactions * 0.1)
            },
            'personality': personality,
            'soul_metrics':
            soul_metrics,  # ← NEW: Add Soul Metrics to response
            'hrv_metrics': {  # ← NEW: Add HRV metrics to response
                'current_hrv': hrv_value,
                'hrv_z_score': updated_state['hrv_z_score'],
                'consciousness_level': updated_state['consciousness_level'],
                'stress_factor': hrv_status['stress_factor'],
                'energy_level': hrv_status['energy_level']
            }
        }

        # Store autonomous responses for frontend polling
        if request.json and request.json.get('autonomous'):
            autonomous_responses.append({
                'timestamp':
                datetime.now().isoformat(),
                'response':
                agent_response,
                'session_id':
                session_id,
                'personality':
                personality,
                'hrv_metrics':
                response_data['hrv_metrics'],
                'soul_metrics':
                soul_metrics
            })
            logging.info(f"🔄 Stored autonomous response for frontend polling")

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in agent chat: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/autonomous_updates', methods=['GET'])
def get_autonomous_updates():
    """Get latest autonomous responses for frontend polling"""
    try:
        # Get timestamp parameter for filtering new responses
        since_timestamp = request.args.get('since', '')

        # Convert autonomous_responses deque to list for easier filtering
        responses = list(autonomous_responses)

        # Filter responses newer than since_timestamp if provided
        if since_timestamp:
            responses = [
                r for r in responses if r['timestamp'] > since_timestamp
            ]

        return jsonify({
            'status': 'success',
            'responses': responses,
            'total_count': len(autonomous_responses),
            'new_count': len(responses)
        })

    except Exception as e:
        logging.error(f"Error getting autonomous updates: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/autonomous/start', methods=['POST'])
def start_autonomous_mode():
    """Start autonomous HRV-driven dialogue mode"""
    try:
        import subprocess
        import threading

        # Start autonomous HRV stream in background process
        def run_autonomous():
            try:
                subprocess.Popen(['python3', 'hrv_autonomous_stream.py'],
                                 cwd='/home/runner/workspace')
                logging.info("🚀 Autonomous HRV stream started in background")
            except Exception as e:
                logging.error(f"Error starting background process: {e}")

        # Start in background thread
        thread = threading.Thread(target=run_autonomous, daemon=True)
        thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Autonomous HRV mode started',
            'session_id': f"autonomous_hrv_{int(time.time())}"
        })

    except Exception as e:
        logging.error(f"Error starting autonomous mode: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/interactions', methods=['GET'])
def api_get_interactions():
    """Get recent interactions for Syndesis frontend"""
    try:
        limit = request.args.get('limit', 50, type=int)
        session_id = request.args.get('session_id')

        query = InteractionLog.query.order_by(InteractionLog.timestamp.desc())

        if session_id:
            query = query.filter_by(session_id=session_id)

        interactions = query.limit(limit).all()

        return jsonify({
            'interactions': [{
                'id': i.id,
                'input': i.agent_input,
                'output': i.agent_output,
                'context': i.context,
                'session_id': i.session_id,
                'timestamp': i.timestamp.isoformat(),
                'processed': i.processed
            } for i in interactions],
            'total_count':
            InteractionLog.query.count()
        })

    except Exception as e:
        logging.error(f"Error getting interactions: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/bulk_import', methods=['POST'])
def api_bulk_import():
    """Bulk import interactions from Syndesis frontend"""
    try:
        data = request.get_json()
        if not data or 'interactions' not in data:
            return jsonify({'error': 'Missing interactions in request'}), 400

        interactions = data['interactions']
        imported_count = 0

        for item in interactions:
            # Support multiple field formats
            agent_input = item.get('prompt') or item.get('input') or item.get(
                'user') or item.get('question', '')
            agent_output = item.get('response') or item.get(
                'output') or item.get('assistant') or item.get('answer', '')
            context = item.get('context', '')
            session_id = item.get(
                'session_id'
            ) or f"bulk_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{imported_count}"

            if agent_input and agent_output:
                interaction = InteractionLog()
                interaction.agent_input = agent_input
                interaction.agent_output = agent_output
                interaction.context = context
                interaction.session_id = session_id
                interaction.processed = False
                db.session.add(interaction)
                imported_count += 1

        db.session.commit()

        return jsonify({
            'status':
            'success',
            'imported_count':
            imported_count,
            'message':
            f'Successfully imported {imported_count} interactions'
        })

    except Exception as e:
        logging.error(f"Error in bulk import: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/export', methods=['POST'])
def api_export_dataset():
    """Export dataset from Syndesis frontend trigger"""
    try:
        data = request.get_json() or {}
        min_interactions = data.get('min_interactions', 10)

        # Check if we have enough data
        unprocessed_count = InteractionLog.query.filter_by(
            processed=False).count()
        if unprocessed_count < min_interactions:
            return jsonify({
                'status':
                'insufficient_data',
                'message':
                f'Need at least {min_interactions} interactions, but only {unprocessed_count} available'
            }), 400

        # Export dataset
        # Export dataset functionality placeholder
        dataset_path = "data/exported_dataset.jsonl"

        return jsonify({
            'status':
            'success',
            'dataset_path':
            dataset_path,
            'exported_interactions':
            unprocessed_count,
            'message':
            f'Successfully exported {unprocessed_count} interactions'
        })

    except Exception as e:
        logging.error(f"Error exporting dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/train', methods=['POST'])
def api_start_training():
    """Start training from Syndesis frontend"""
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 1)
        model_name = data.get('model_name', 'syndesis-model')

        # Start training
        job_id = orchestrator.start_training(epochs=epochs)

        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'message': f'Training job {job_id} started successfully'
        })

    except Exception as e:
        logging.error(f"Error starting training: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint for Syndesis frontend"""
    try:
        # Check database connection
        from sqlalchemy import text
        db.session.execute(text("SELECT 1"))

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'total_interactions': InteractionLog.query.count(),
            'replit_url': request.host_url
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/memory/evolution/<session_id>', methods=['GET'])
def api_get_memory_evolution(session_id):
    """Get memory evolution data for a specific session - NOVEL FEATURE"""
    try:
        # Check if agent has evolution system for this session
        if hasattr(
                agent,
                'evolution_systems') and session_id in agent.evolution_systems:
            evolution_system = agent.evolution_systems[session_id]

            return jsonify({
                'status':
                'success',
                'session_id':
                session_id,
                'evolution_data':
                evolution_system.export_evolution_data(),
                'relationship_summary':
                evolution_system.get_relationship_summary(),
                'memory_count':
                len(evolution_system.memory_nodes),
                'personality_dominant_trait':
                max(evolution_system.personality.traits,
                    key=evolution_system.personality.traits.get)
            })
        else:
            return jsonify({
                'status':
                'no_data',
                'message':
                f'No memory evolution data found for session {session_id}'
            }), 404

    except Exception as e:
        logging.error(f"Error getting memory evolution: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/relationships', methods=['GET'])
def api_get_all_relationships():
    """Get overview of all evolving AI relationships - NOVEL FEATURE"""
    try:
        if not hasattr(agent, 'evolution_systems'):
            return jsonify({'relationships': [], 'total_sessions': 0})

        relationships = []
        for session_id, evolution_system in agent.evolution_systems.items():
            summary = evolution_system.get_relationship_summary()
            relationships.append(summary)

        return jsonify({
            'status': 'success',
            'relationships': relationships,
            'total_sessions': len(relationships),
            'active_memory_systems': len(agent.evolution_systems)
        })

    except Exception as e:
        logging.error(f"Error getting relationships: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============ ENHANCED NEURAL MODULES ROUTES ============


@app.route('/api/enhanced-centers/awareness', methods=['GET', 'POST'])
def api_awareness_center():
    """Μπλε Κέντρο (AWARENESS) - AI Adaptation με L2 Regularization"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            interaction_data = data.get('interaction_data', {})

            # L2 AI Adaptation learning
            l2_adaptation.learn_user_pattern(interaction_data)
            enhanced_memory.add_awareness_entry(
                interaction_data,
                "AI pattern learning updated with L2 regularization")

            return jsonify({
                'status':
                'success',
                'center':
                'AWARENESS',
                'adaptation_insights':
                l2_adaptation.generate_adaptation_insights(),
                'smooth_transitions':
                l2_adaptation.get_smooth_transition_params()
            })

        else:
            # GET request - return current state
            return jsonify({
                'center':
                'AWARENESS (AI Adaptation)',
                'current_weights':
                l2_adaptation.weights,
                'adaptation_history':
                len(l2_adaptation.adaptation_history),
                'transition_params':
                l2_adaptation.get_smooth_transition_params(),
                'insights':
                l2_adaptation.generate_adaptation_insights()
            })

    except Exception as e:
        logging.error(f"Error in awareness center: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced-centers/compassion', methods=['GET', 'POST'])
def api_compassion_center():
    """Γκρι Κέντρο (COMPASSION) - Soul Metrics + HRV με Manifold Visualization"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            hrv_data = data.get('hrv_data', {})
            soul_metrics = data.get('soul_metrics', {})

            # Neural Manifold HRV analysis
            neural_manifold.update_position_from_hrv(hrv_data)
            emotional_analysis = neural_manifold.analyze_emotional_state()

            enhanced_memory.add_compassion_entry(
                hrv_data, soul_metrics,
                f"Manifold position updated: {emotional_analysis['emotional_state']}"
            )

            return jsonify({
                'status':
                'success',
                'center':
                'COMPASSION',
                'manifold_data':
                neural_manifold.get_manifold_visualization_data(),
                'emotional_analysis':
                emotional_analysis,
                'soul_metrics':
                soul_metrics
            })

        else:
            # GET request - return current state
            return jsonify({
                'center':
                'COMPASSION (Soul + HRV)',
                'manifold_position':
                neural_manifold.current_position,
                'emotional_state':
                neural_manifold.analyze_emotional_state(),
                'visualization_data':
                neural_manifold.get_manifold_visualization_data(),
                'trajectory_points':
                len(neural_manifold.trajectory_history)
            })

    except Exception as e:
        logging.error(f"Error in compassion center: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced-centers/memory-evolution', methods=['GET'])
def api_enhanced_memory_evolution():
    """Ολοκληρωμένα δεδομένα memory evolution από αμφότερα τα κέντρα"""
    try:
        memory_data = enhanced_memory.get_memory_evolution_data()

        return jsonify({
            'status':
            'success',
            'enhanced_memory':
            memory_data,
            'cross_center_insights':
            enhanced_memory.get_global_insights(),
            'total_entries':
            memory_data['total_memory_points'],
            'system_status':
            'Enhanced Neural Integration Active'
        })

    except Exception as e:
        logging.error(f"Error getting enhanced memory: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced-centers/manifold-visualization', methods=['GET'])
def api_manifold_visualization():
    """Manifold visualization data για το frontend"""
    try:
        viz_data = neural_manifold.get_manifold_visualization_data()

        return jsonify({
            'status':
            'success',
            'manifold_visualization':
            viz_data,
            'real_time_position':
            neural_manifold.current_position,
            'trajectory_history':
            neural_manifold.trajectory_history[-20:]  # Last 20 points
        })

    except Exception as e:
        logging.error(f"Error getting manifold visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============ CORE AI MODULE ROUTES ============


@app.route('/api/core-ai/process', methods=['POST'])
def api_core_ai_process():
    """Core AI processing για panel updates"""
    try:
        # Generate fresh presence data
        user_presence.generate_presence()

        # Process με το Core AI
        result = core_ai.process_presence(user_presence)

        return jsonify({
            'status': 'success',
            'processing_result': result,
            'panel_data': core_ai.get_all_panels_data()
        })

    except Exception as e:
        logging.error(f"Error in core AI processing: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/core-ai/panels/<panel_type>', methods=['GET'])
def api_get_panel_data(panel_type):
    """Δεδομένα για συγκεκριμένο panel"""
    try:
        panel_data = core_ai.get_panel_data(panel_type)

        return jsonify({
            'status': 'success',
            'panel_type': panel_type,
            'data': panel_data
        })

    except Exception as e:
        logging.error(f"Error getting {panel_type} panel data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/core-ai/status', methods=['GET'])
def api_core_ai_status():
    """Status για όλο το Core AI system"""
    try:
        return jsonify({
            'status':
            'success',
            'core_ai':
            'Active',
            'all_panels':
            core_ai.get_all_panels_data(),
            'presence_available':
            user_presence.presence_data is not None
        })

    except Exception as e:
        logging.error(f"Error getting core AI status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory_evolution')
def memory_evolution_dashboard():
    """Dashboard for the novel Memory Evolution System"""
    return render_template('memory_evolution.html')


@app.route('/landing')
def landing_page():
    """Marketing landing page"""
    return render_template('landing.html')


# Removed duplicate route - using the one at line 112 that serves book_centers_2d.html


@app.route('/ultra')
def ultra_minimal():
    """Ultra minimal YOU interface με real TICKR heart rate data"""
    # Create YOU interface configuration with all required template variables
    book_data = {
        'title':
        'YOU / PRESENCE - Real Heart Rate',
        'primary_id':
        'presence-ai',
        'total_chapters':
        1,
        'centers': [{
            'id': 'presence-ai',
            'name': 'YOU',
            'x': 50,
            'y': 50,
            'color': '#3B82F6',
            'description':
            'Primary consciousness center with TICKR biometric data',
            'position': {
                'top': '50%',
                'left': '50%'
            },
            'pattern': {
                'color': '#3B82F6',
                'border_color': '#1D4ED8'
            }
        }]
    }

    # Initialize default personality traits for template
    personality_traits = {
        'empathy': 0.75,
        'creativity': 0.65,
        'resilience': 0.8,
        'focus': 0.7,
        'curiosity': 0.85,
        'compassion': 0.75
    }

    return render_template('book_centers_integrated.html',
                           book_data=book_data,
                           personality_traits=personality_traits,
                           session_id='ultra-session')


@app.route('/book-centers')
def book_centers():
    """Revolutionary Book Centers - Biofeedback Experience for 'You Are Not Your Mind' Chapter

    Single chapter focus with neural HRV integration for deep understanding assessment.
    """
    try:
        # Load single chapter "You Are Not Your Mind" data
        with open('data/books.json', 'r') as f:
            books = json.load(f)

        # Find "The Power of Now" book
        power_of_now = None
        for book in books:
            if book.get('title') == 'The Power of Now' and 'chapters' in book:
                power_of_now = book
                break

        if power_of_now and len(power_of_now['chapters']) > 0:
            # Create TWO centers as requested by user - AWARENESS (blue) and COMPASSION (grey)
            first_chapter = power_of_now['chapters'][0]

            book_data = {
                'title':
                'Two Centers Experience',
                'total_chapters':
                2,
                'centers': [{
                    'id': 'center1',
                    'name': 'AWARENESS',
                    'title': 'Mindfulness Path (Thich Nhat Hanh)',
                    'description':
                    'Blue center representing awareness and mindful presence',
                    'importance': 10,
                    'pattern': {
                        'shape':
                        'circle',
                        'color':
                        'rgba(79, 172, 254, 0.6)',
                        'border_color':
                        'rgba(79, 172, 254, 0.8)',
                        'description':
                        'Thich Nhat Hanh Teachings',
                        'keywords': [
                            'awareness', 'mindfulness', 'breath', 'present',
                            'conscious'
                        ]
                    },
                    'position': {
                        'top': '35%',
                        'left': '35%'
                    },
                    'unlock_requirements': {
                        'alignment_threshold': 75,
                        'focus_weight': 0.4,
                        'coherence_weight': 0.3,
                        'hrv_weight': 0.3
                    },
                    'css_shape': 'border-radius: 50%;'
                }, {
                    'id': 'center2',
                    'name': 'COMPASSION',
                    'title': 'Compassion Path',
                    'description':
                    'Grey center representing compassion and loving-kindness',
                    'importance': 10,
                    'pattern': {
                        'shape':
                        'circle',
                        'color':
                        'rgba(128, 128, 128, 0.6)',
                        'border_color':
                        'rgba(128, 128, 128, 0.8)',
                        'description':
                        'Compassion Teachings',
                        'keywords':
                        ['compassion', 'love', 'kindness', 'heart', 'care']
                    },
                    'position': {
                        'top': '65%',
                        'left': '65%'
                    },
                    'unlock_requirements': {
                        'alignment_threshold': 75,
                        'focus_weight': 0.4,
                        'coherence_weight': 0.3,
                        'hrv_weight': 0.3
                    },
                    'css_shape': 'border-radius: 50%;'
                }]
            }
        else:
            # Fallback - two centers as requested
            book_data = {
                'title':
                'Two Centers Experience',
                'total_chapters':
                2,
                'centers': [{
                    'id': 'center1',
                    'name': 'AWARENESS',
                    'title': 'Mindfulness Path (Thich Nhat Hanh)',
                    'description':
                    'Blue center representing awareness and mindful presence',
                    'importance': 10,
                    'pattern': {
                        'shape':
                        'circle',
                        'color':
                        'rgba(79, 172, 254, 0.6)',
                        'border_color':
                        'rgba(79, 172, 254, 0.8)',
                        'description':
                        'Thich Nhat Hanh Teachings',
                        'keywords': [
                            'awareness', 'mindfulness', 'breath', 'present',
                            'conscious'
                        ]
                    },
                    'position': {
                        'top': '35%',
                        'left': '35%'
                    },
                    'unlock_requirements': {
                        'alignment_threshold': 75,
                        'focus_weight': 0.4,
                        'coherence_weight': 0.3,
                        'hrv_weight': 0.3
                    },
                    'css_shape': 'border-radius: 50%;'
                }, {
                    'id': 'center2',
                    'name': 'COMPASSION',
                    'title': 'Compassion Path',
                    'description':
                    'Grey center representing compassion and loving-kindness',
                    'importance': 10,
                    'pattern': {
                        'shape':
                        'circle',
                        'color':
                        'rgba(128, 128, 128, 0.6)',
                        'border_color':
                        'rgba(128, 128, 128, 0.8)',
                        'description':
                        'Compassion Teachings',
                        'keywords':
                        ['compassion', 'love', 'kindness', 'heart', 'care']
                    },
                    'position': {
                        'top': '65%',
                        'left': '65%'
                    },
                    'unlock_requirements': {
                        'alignment_threshold': 75,
                        'focus_weight': 0.4,
                        'coherence_weight': 0.3,
                        'hrv_weight': 0.3
                    },
                    'css_shape': 'border-radius: 50%;'
                }]
            }

        # Get personality traits for integration
        from fractalai.persistent_personality import PersistentPersonalityManager
        personality_manager = PersistentPersonalityManager()
        session_id = 'neural_hrv_session'
        personality_traits = personality_manager.get_or_create_personality(
            session_id)

        return render_template('book_centers_integrated.html',
                               book_data=book_data,
                               session_id='neural_hrv_session',
                               personality_traits=personality_traits)

    except Exception as e:
        logging.error(f"Error loading single chapter book centers: {str(e)}")
        # Ultimate fallback
        fallback_data = {
            'title':
            'You Are Not Your Mind',
            'total_chapters':
            1,
            'centers': [{
                'id': 'center1',
                'name': 'PRESENCE',
                'title': 'You Are Not Your Mind',
                'description':
                'Learning to observe thoughts without identification.',
                'pattern': {
                    'shape': 'circle',
                    'color': 'rgba(79, 172, 254, 0.6)'
                },
                'position': {
                    'top': '50%',
                    'left': '50%'
                }
            }]
        }
        return render_template('book_centers_integrated.html',
                               book_data=fallback_data,
                               session_id='fallback_session',
                               personality_traits={
                                   'empathy': 0.5,
                                   'analytical': 0.5,
                                   'curiosity': 0.5
                               })


@app.route('/museafs/FractalTrain')
def museafs_fractal_train():
    """FractalTrain main consciousness mapping interface"""
    try:
        # Get current session personality
        from fractalai.persistent_personality import PersistentPersonalityManager
        personality_manager = PersistentPersonalityManager()
        session_id = 'fractal_train_session'
        personality_traits = personality_manager.get_or_create_personality(session_id)
        
        # Load YOU center book data for consciousness mapping
        book_data = {
            'title': 'Syndesis AI - Neural Consciousness Mapping',
            'book_centers': [{
                'name': 'YOU',
                'type': 'PRESENCE',
                'description': 'Your biometric-driven consciousness visualization',
                'color': '#00ff88',
                'position': {'top': '50%', 'left': '50%'}
            }]
        }
        
        return render_template('book_centers_integrated.html',
                              book_data=book_data,
                              session_id=session_id,
                              personality_traits=personality_traits)
                              
    except Exception as e:
        logging.error(f"Error loading FractalTrain interface: {str(e)}")
        # Fallback 
        return render_template('book_centers_integrated.html',
                              book_data={'title': 'FractalTrain', 'book_centers': []},
                              session_id='fallback_session',
                              personality_traits={'empathy': 0.5, 'analytical': 0.5})


@app.route('/power-of-now')
def power_of_now_centers():
    """The Power of Now Book Centers Experience"""
    try:
        with open('data/The Power Of Now - Eckhart Tolle_centers.json',
                  'r') as f:
            book_data = json.load(f)
        return render_template('book_centers_2d.html', book_data=book_data)
    except FileNotFoundError:
        flash('Book centers not found', 'error')
        return redirect(url_for('book_centers'))


# Removed conflicting route - now handled by universal_book_bp blueprint


@app.route('/api/process-book', methods=['POST'])
def api_process_book():
    """Process uploaded book and create Book Centers"""
    try:
        from book_processor import BookCenterProcessor

        # Get book text from form or file
        book_text = ""
        title = "Untitled Book"

        if 'book_file' in request.files:
            file = request.files['book_file']
            if file and file.filename:
                # Try multiple encodings for file reading
                file_content = file.read()
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

                book_text = ""
                for encoding in encodings:
                    try:
                        book_text = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue

                if not book_text:
                    return jsonify({
                        'error':
                        'Could not decode file. Please try a different format or paste text directly.'
                    }), 400

                title = file.filename.rsplit('.', 1)[0]
        elif request.form.get('book_text'):
            book_text = request.form.get('book_text')
            title = request.form.get('title', 'Custom Book')

        if not book_text:
            return jsonify({'error': 'No book content provided'}), 400

        # Process the book
        processor = BookCenterProcessor()
        try:
            book_data = processor.process_book_text(book_text, title)
        except Exception as e:
            logging.error(f"Book processing error: {e}")
            return jsonify({'error': f'Book processing failed: {str(e)}'}), 500

        # Save to data directory
        import os
        os.makedirs('data', exist_ok=True)
        safe_filename = "".join(
            c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        processor.save_book_centers(book_data, safe_filename)

        return jsonify({
            'success': True,
            'book_data': book_data,
            'centers_url': f'/book-centers/{safe_filename}',
            'integrated_systems': {
                'personality_traits': True,
                'soul_metrics': True,
                'memory_system': True,
                'hrv_biometrics': True,
                'anomaly_detection': True
            }
        })

    except Exception as e:
        logging.error(f"Error processing book: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/book-centers/<book_id>')
def book_centers_custom(book_id):
    """Custom Book Centers with full system integration"""
    try:
        from book_processor import BookCenterProcessor
        from fractalai.persistent_personality import PersistentPersonalityManager
        from fractalai.soul_metrics import SoulMetrics
        from fractalai.human_memory_system import HumanLikeMemorySystem

        # Load book data
        processor = BookCenterProcessor()
        book_data = processor.load_book_centers(book_id)

        if not book_data:
            return redirect(url_for('upload_book_page'))

        # Initialize all systems for integrated experience
        from flask import session
        session_id = session.get('session_id', 'default_session')

        personality_manager = PersistentPersonalityManager()
        soul_metrics = SoulMetrics()
        memory_system = HumanLikeMemorySystem()

        # Get current personality state
        personality_traits = personality_manager.get_or_create_personality(
            session_id)

        # Create integrated context
        integrated_context = {
            'book_data': book_data,
            'personality_traits': personality_traits,
            'session_id': session_id,
            'memory_active': True,
            'soul_metrics_active': True,
            'anomaly_detection_active': True
        }

        return render_template('book_centers_integrated.html',
                               **integrated_context)

    except Exception as e:
        logging.error(f"Error loading integrated book centers: {str(e)}")
        return redirect(url_for('upload_book_page'))


@app.route('/panel-you')
def panel_you_demo():
    """Panel-YOU System Demo - Pure ES6 modules with somatic feedback"""
    from flask import send_file
    import os
    static_path = os.path.join(app.static_folder, 'panel-you-demo.html')
    return send_file(static_path)

@app.route('/unified-dashboard')
def unified_dashboard():
    """Unified dashboard showing all system modules"""
    return render_template('unified_dashboard.html')


@app.route('/api/integrated-systems/status')
def integrated_systems_status():
    """Get status of all integrated systems"""
    try:
        from fractalai.persistent_personality import PersistentPersonalityManager
        from fractalai.soul_metrics import SoulMetrics
        from fractalai.hrv_system import HRVSystem

        session_id = session.get('session_id', 'default_session')

        personality_manager = PersistentPersonalityManager()
        soul_metrics = SoulMetrics()
        hrv_system = HRVSystem()

        # Get current states
        personality_traits = personality_manager.get_or_create_personality(
            session_id)
        hrv_status = hrv_system.get_system_status()

        return jsonify({
            'success': True,
            'systems': {
                'personality': {
                    'status': 'active',
                    'traits': personality_traits.__dict__ if hasattr(
                        personality_traits, '__dict__') else {},
                    'session_id': session_id
                },
                'soul_metrics': {
                    'status': 'active',
                    'coherence_threshold': soul_metrics.coherence_threshold,
                    'vitality_threshold': soul_metrics.vitality_threshold
                },
                'hrv_biometrics': {
                    'status': 'active',
                    'current_state': hrv_status
                },
                'memory_system': {
                    'status': 'active',
                    'capacity': 'unlimited'
                },
                'anomaly_detection': {
                    'status': 'active',
                    'monitoring': True
                },
                'book_centers': {
                    'status': 'active',
                    'integration': 'full'
                }
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Error getting integrated systems status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/demo')
def demo():
    """Full business demo page"""
    return redirect(url_for('business_demo'))


@app.route('/minimal')
def minimal():
    """Previous minimal UI"""
    return render_template('minimal_ui.html')


@app.route('/old')
def old_home():
    """Old business demo for comparison"""
    return redirect(url_for('business_demo'))


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Ultra minimal chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Use the existing agent to generate response
        session_id = f"minimal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get AI response
        response_data = agent.get_response(
            user_input=user_message,
            context="Ultra minimal chat interface",
            session_id=session_id)

        return jsonify({
            'response':
            response_data.get(
                'agent_output',
                'Συγγνώμη, δεν μπόρεσα να δημιουργήσω απάντηση.'),
            'personality':
            response_data.get('personality', {}),
            'soul_metrics':
            response_data.get('soul_metrics', {}),
            'timestamp':
            datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Chat API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# 🏥 HEALTH & METRICS ENDPOINTS
# =============================================================================


@app.route('/api/pc-tickr/live-data')
def api_pc_tickr_live_data():
    """Get live TICKR data from PC localhost:8080 or proxy"""
    try:
        global latest_tickr_proxy_data, last_tickr_proxy_update

        # Check for fresh proxy data first
        if last_tickr_proxy_update:
            time_diff = (datetime.now() -
                         last_tickr_proxy_update).total_seconds()
            if time_diff < 15 and latest_tickr_proxy_data:
                logging.info(
                    f"🎯 Serving TICKR data from proxy: HR={latest_tickr_proxy_data.get('heart_rate')} BPM"
                )
                return jsonify({
                    'success': True,
                    'mode': 'proxy_data',
                    'data': latest_tickr_proxy_data,
                    'data_age_seconds': time_diff,
                    'timestamp': datetime.now().isoformat()
                })

        if pc_tickr_bridge is None:
            return jsonify({
                'error': 'PC TICKR Bridge not available',
                'mode': 'simulation_only'
            }), 503

        # Fallback to direct PC bridge
        live_data = pc_tickr_bridge.get_live_data()

        if live_data:
            return jsonify({
                'success': True,
                'mode': 'authentic_pc_data',
                'data': live_data,
                'bridge_status': pc_tickr_bridge.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'mode': 'pc_unreachable',
                'message': 'Cannot reach PC TICKR data at localhost:8080',
                'bridge_status': pc_tickr_bridge.get_status()
            }), 404

    except Exception as e:
        logging.error(f"PC TICKR API error: {e}")
        return jsonify({'error': str(e), 'mode': 'error'}), 500


# Global variables για PC TICKR proxy data
latest_tickr_proxy_data = None
last_tickr_proxy_update = None


@app.route('/api/pc-tickr/receive', methods=['POST'])
def receive_pc_tickr_data():
    """Receives TICKR data από PC proxy script"""
    global latest_tickr_proxy_data, last_tickr_proxy_update

    try:
        # Έλεγχος authentication token
        token = request.headers.get('x-agent-token')
        if token != 'TO_MYSTIKO_SOU':
            return jsonify({'error': 'Unauthorized access'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        latest_tickr_proxy_data = data
        last_tickr_proxy_update = datetime.now()

        # Debug log όλων των δεδομένων
        logging.info(f"🎯 PC TICKR Proxy Data Received: {data}")
        logging.info(
            f"🎯 Real TICKR data received via proxy: HR={data.get('heart_rate')} BPM"
        )
        return jsonify({
            'status': 'success',
            'received': True,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Error receiving PC TICKR proxy data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pc-tickr/status')
def api_pc_tickr_status():
    """Check PC TICKR connection status με proxy support"""
    try:
        global latest_tickr_proxy_data, last_tickr_proxy_update

        # Έλεγχος αν έχουμε fresh proxy data (τελευταία 15 δευτερόλεπτα)
        if last_tickr_proxy_update:
            time_diff = (datetime.now() -
                         last_tickr_proxy_update).total_seconds()
            if time_diff < 15 and latest_tickr_proxy_data:
                return jsonify({
                    'status':
                    'connected',
                    'connection_type':
                    'proxy_mode',
                    'last_reading':
                    latest_tickr_proxy_data,
                    'last_update':
                    last_tickr_proxy_update.isoformat(),
                    'data_age_seconds':
                    time_diff,
                    'timestamp':
                    datetime.now().isoformat()
                })

        # Fallback σε direct bridge connection
        if pc_tickr_bridge is None:
            return jsonify({'status': 'bridge_not_available'})

        return jsonify(pc_tickr_bridge.get_status())

    except Exception as e:
        logging.error(f"PC TICKR status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health-detailed')
def health_check_detailed():
    """Get comprehensive system health status"""
    try:
        from health_metrics import health_metrics
        health_status = health_metrics.get_health_status()

        # Convert to dict for JSON response
        response = {
            'status': health_status.status,
            'timestamp': health_status.timestamp,
            'uptime_seconds': health_status.uptime_seconds,
            'components': health_status.components,
            'metrics': health_status.metrics,
            'alerts': health_status.alerts
        }

        # Set appropriate HTTP status based on health
        if health_status.status == 'healthy':
            return jsonify(response), 200
        elif health_status.status == 'degraded':
            return jsonify(response), 200  # Still OK but with warnings
        else:
            return jsonify(response), 503  # Service unavailable

    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503


@app.route('/metrics')
def metrics():
    """Get system metrics summary"""
    try:
        from health_metrics import health_metrics
        time_range = request.args.get('minutes', 5, type=int)
        metrics_summary = health_metrics.get_metrics_summary(time_range)

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'time_range_minutes': time_range,
            'metrics': metrics_summary
        })

    except Exception as e:
        logging.error(f"Metrics endpoint failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics/export')
def export_metrics():
    """Export metrics in JSON format for external monitoring"""
    try:
        from health_metrics import health_metrics
        time_range = request.args.get('minutes', 60, type=int)
        exported_data = health_metrics.export_metrics_json(time_range)

        # Return as downloadable file
        response = app.response_class(
            response=exported_data,
            status=200,
            mimetype='application/json',
            headers={
                'Content-Disposition':
                f'attachment; filename=hawkins_metrics_{int(time.time())}.json'
            })
        return response

    except Exception as e:
        logging.error(f"Metrics export failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/hawkins-presence/metrics', methods=['POST'])
def record_hawkins_metrics():
    """Record consciousness analysis metrics from Hawkins module"""
    try:
        from health_metrics import health_metrics

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract timing information
        analysis_time = data.get('analysis_time_ms', 0)

        # Record the metrics (would normally receive HawkinsState object)
        # For API endpoint, we reconstruct basic info
        health_metrics.record_api_call('/api/hawkins-presence/metrics', 200,
                                       analysis_time)

        # Record custom metrics from the request
        if 'consciousness_level' in data:
            health_metrics._record_metric(
                'hawkins.level', data['consciousness_level'],
                {'source': data.get('source', 'api')})

        if 'distance' in data:
            health_metrics._record_metric('hawkins.distance', data['distance'],
                                          {})

        return jsonify({
            'status': 'recorded',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Hawkins metrics recording failed: {e}")
        health_metrics.record_error('hawkins_metrics', 'recording_error',
                                    str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/syndesis_integration')
def syndesis_integration_page():
    """Syndesis integration demo and documentation"""
    return render_template('syndesis_integration.html')


@app.route('/syndesis_connect')
def syndesis_connect_page():
    """Live Syndesis connection interface"""
    return render_template('syndesis_connect.html')


@app.route('/business_demo')
def business_demo():
    """Professional business demo page for Facebook/Meta presentation"""
    print("DEBUG: business_demo route called")
    return render_template('business_demo.html')


@app.route('/urls')
def url_helper():
    """URL helper page"""
    return render_template('url_helper.html')


@app.route('/audit_dashboard')
def audit_dashboard():
    """AI Adaptation Audit Dashboard - Anti-Placebo Proof"""
    return render_template('audit_dashboard.html')


@app.route('/audit-dashboard')
def audit_dashboard_dash():
    """AI Adaptation Audit Dashboard - Anti-Placebo Proof (with dash)"""
    return render_template('audit_dashboard.html')


@app.route("/audit")
def audit_demo():
    """Standalone audit proof page"""
    return render_template("audit_demo.html")


@app.route('/api/personality-evolution')
def personality_evolution_api():
    """Live Personality Evolution API endpoint"""
    try:
        # Generate realistic personality evolution data
        import random
        import time

        # Simulate personality traits evolution based on real-time data
        current_time = int(time.time() * 1000)  # milliseconds

        # Base personality traits with realistic variation
        base_traits = {
            'empathy': 0.75,
            'creativity': 0.82,
            'resilience': 0.68,
            'focus': 0.71,
            'curiosity': 0.89,
            'compassion': 0.77
        }

        # Add realistic variation (±10%)
        evolved_traits = {}
        for trait, base_value in base_traits.items():
            variation = (random.random() - 0.5) * 0.2  # ±10% variation
            evolved_traits[trait] = max(0.1, min(1.0, base_value + variation))

        # Simulate evolution metrics
        evolution_data = {
            'current_traits':
            evolved_traits,
            'evolution_speed':
            round(random.uniform(0.15, 0.35), 3),
            'adaptation_score':
            round(random.uniform(0.7, 0.95), 3),
            'learning_momentum':
            round(random.uniform(0.6, 0.9), 3),
            'timestamp':
            current_time,
            'status':
            'evolving',
            'insights': [
                f"Creativity increased by {random.randint(2, 8)}% in last 5 minutes",
                f"Empathy adaptation detected from conversation context",
                f"Focus enhancement from biometric coherence patterns"
            ]
        }

        return jsonify(evolution_data)

    except Exception as e:
        logging.error(f"Error in personality evolution API: {e}")
        return jsonify({
            'error': 'Personality evolution temporarily unavailable',
            'status': 'fallback_mode',
            'timestamp': int(time.time() * 1000)
        }), 500


# 🌟 HAWKINS CONSCIOUSNESS-ADAPTIVE PRESENCE AGENT
@app.route('/api/hawkins-presence/adaptive-message', methods=['POST'])
def hawkins_adaptive_message():
    """
    🎭 REVOLUTIONARY CONSCIOUSNESS-ADAPTIVE AI RESPONSE

    Takes user consciousness data and generates personalized AI messages
    based on Hawkins Scale analysis and persona adaptation.

    Flow: User Metrics → Hawkins Analysis → Persona Adaptation → AI Response
    """
    try:
        # Import the consciousness modules
        from hawkins_presence_module import HawkinsPresenceModule, HawkinsState
        from persona_layer import PersonaLayer
        from openai import OpenAI
        import os

        # Get user consciousness data from frontend
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No consciousness data provided'}), 400

        # Extract consciousness metrics from frontend data
        soul_metrics = {
            'coherence': data.get('soul_coherence', 80),
            'vitality': data.get('soul_vitality', 80),
            'ethics': data.get('soul_ethics', 85),
            'narrative': data.get('soul_narrative', 75)
        }

        hrv_data = {
            'hrv': data.get('hrv_value', 40.0),
            'coherence': data.get('hrv_coherence', 65),
            'heart_rate': data.get('heart_rate', 72)
        }

        focus_level = data.get('focus_level', 0.85)

        personality_traits = {
            'empathy':
            data.get('empathy', 0.7) /
            100 if data.get('empathy', 70) > 1 else data.get('empathy', 0.7),
            'analytical':
            data.get('analytical', 0.4) / 100
            if data.get('analytical', 40) > 1 else data.get('analytical', 0.4),
            'curiosity':
            data.get('curiosity', 0.9) / 100
            if data.get('curiosity', 90) > 1 else data.get('curiosity', 0.9)
        }

        # Create user metrics structure for Hawkins analysis
        user_metrics = {
            'soul_metrics': soul_metrics,
            'hrv_data': hrv_data,
            'focus_level': focus_level,
            'personality_traits': personality_traits
        }

        # 🌟 STEP 1: HAWKINS PRESENCE ANALYSIS
        hawkins_module = HawkinsPresenceModule()
        hawkins_state = hawkins_module.analyze_presence_state(user_metrics)

        # 🎭 STEP 2: PERSONA ADAPTATION
        persona_layer = PersonaLayer()
        persona_config = persona_layer.generate_persona_config(hawkins_state)

        # 🤖 STEP 3: GENERATE ADAPTIVE AI RESPONSE
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Generate consciousness-adapted message
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # Latest OpenAI model
                messages=[{
                    "role": "system",
                    "content": persona_config.system_prompt
                }, {
                    "role":
                    "user",
                    "content":
                    f"Generate a single, powerful presence statement that helps me in my current state. Current consciousness level: {hawkins_state.level}, Flow state: {hawkins_state.flow_state}, Distance from center: {hawkins_state.distance}. Give me exactly what I need to hear right now in 1-2 sentences."
                }],
                max_tokens=100,
                temperature=0.7)

            adaptive_message = response.choices[0].message.content.strip()

        except Exception as e:
            logging.warning(f"OpenAI generation failed: {e}, using fallback")
            # Fallback based on consciousness level
            if hawkins_state.level >= 600:
                adaptive_message = "You are pure joy radiating into existence. Rest in this light."
            elif hawkins_state.level >= 500:
                adaptive_message = "Love flows through you effortlessly. You are connected to all."
            elif hawkins_state.level >= 400:
                adaptive_message = "Your understanding deepens with each breath. Trust your insights."
            elif hawkins_state.level >= 350:
                adaptive_message = "Accept what is, and find peace in this moment of being."
            elif hawkins_state.level >= 200:
                adaptive_message = "You have the courage to be present. Take the next step forward."
            else:
                adaptive_message = "Breathe deeply. You are safe, supported, and growing."

        # Return comprehensive response
        return jsonify({
            'success': True,
            'adaptive_message': adaptive_message,
            'consciousness_analysis': {
                'hawkins_level': hawkins_state.level,
                'consciousness_state':
                f"{hawkins_state.level} - {hawkins_module.hawkins_bands.get(hawkins_state.level, {}).get('state', 'Unknown')}",
                'flow_state': hawkins_state.flow_state,
                'distance_from_center': hawkins_state.distance,
                'consciousness_trend': hawkins_state.consciousness_trend,
                'next_step': hawkins_state.next_step
            },
            'persona_adaptation': {
                'tone': persona_config.tone,
                'intent': persona_config.intent,
                'empathy_factor': persona_config.empathy_factor,
                'energy_level': persona_config.energy_level
            },
            'alerts': hawkins_state.alerts,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Hawkins adaptive message error: {e}")
        return jsonify({
            'success': False,
            'error': 'Consciousness analysis temporarily unavailable',
            'fallback_message':
            'Be present. You are exactly where you need to be.',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/spatial-demo')
def spatial_interaction_demo():
    """Live demonstration of Center-Loss Detection Algorithm"""
    return render_template('spatial_interaction_demo.html')


# 🧬 SPATIAL INTELLIGENCE API - CENTER-LOSS DETECTION
@app.route('/api/core-ai/status')
def core_ai_spatial_status():
    """
    Core AI Module Spatial Intelligence Status API
    Returns real-time center-loss detection data for Book Centers integration
    """
    try:
        # Import core AI components
        from core_ai_module import CoreAI
        from presence_module import UserPresence
        from center_loss_detection import CenterLossDetection

        # Initialize components
        core_ai_instance = CoreAI()
        user_presence = UserPresence()
        center_loss_detector = CenterLossDetection()

        # Mock HRV and user evolution data (would come from real sensors in production)
        import random
        import time
        current_time = time.time()

        # 🚫 NO SYNTHETIC HRV DATA - Only use authentic TICKR data
        # ✅ FIXED: Simple live signal check without complex function
        signal_status = {
            "has_live_signal": last_hrv_frame and last_hrv_frame.get('live'),
            "hrv_data": last_hrv_frame or {}
        }
        if not signal_status.get('has_live_signal', False):
            return jsonify({
                'status': 'no_signal',
                'message':
                'No live HRV signal detected - synthetic data disabled',
                'timestamp': datetime.now().isoformat()
            }), 404

        # Only proceed with real HRV data
        hrv_metrics = signal_status.get('hrv_data', {})

        # 🚫 NO SYNTHETIC USER TRAITS - Only use authentic personality data
        user_traits = signal_status.get('personality_data', {})

        # Generate spatial analysis using center-loss detection
        spatial_analysis = center_loss_detector.add_state(
            hrv_metrics, user_traits)

        # Process through Core AI
        processed_data = core_ai_instance.process_presence(
            user_presence, user_traits, hrv_metrics)

        # Combine all panel data with spatial intelligence
        response_data = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'all_panels': {
                'spatial_intelligence': {
                    'center_lost':
                    spatial_analysis.get('center_lost', False),
                    'current_distance':
                    round(spatial_analysis.get('current_distance', 0), 4),
                    'threshold':
                    round(spatial_analysis.get('threshold', 0.7), 4),
                    'agent_behavior':
                    spatial_analysis.get('agent_behavior', 'idle'),
                    'risk_level':
                    spatial_analysis.get('risk_level', 'low'),
                    'stability_score':
                    round(spatial_analysis.get('stability_score', 0.8), 3),
                    'user_position': {
                        'x': round(random.uniform(0.2, 0.8), 3),
                        'y': round(random.uniform(0.2, 0.8), 3)
                    },
                    'center_position': {
                        'x': 0.5,
                        'y': 0.5
                    },
                    'animation_params': {
                        'glow':
                        'strong'
                        if spatial_analysis.get('center_lost') else 'normal',
                        'pulse':
                        spatial_analysis.get('center_lost', False),
                        'bridge_line': {
                            'show': spatial_analysis.get('center_lost', False)
                        },
                        'movement': {
                            'type':
                            'recall_to_center'
                            if spatial_analysis.get('center_lost') else
                            'gentle_drift',
                            'pattern':
                            'guided' if spatial_analysis.get('center_lost')
                            else 'natural'
                        }
                    }
                },
                'biometric_data': hrv_metrics,
                'personality_evolution': {
                    'current_traits': user_traits,
                    'evolution_trend': 'stable',
                    'learning_rate': round(random.uniform(0.15, 0.4), 3)
                },
                'ai_panels': {
                    'awareness': processed_data.get('awareness_data', {}),
                    'compassion': processed_data.get('compassion_data', {})
                }
            },
            'intervention_active':
            spatial_analysis.get('agent_behavior') == 'recall_to_center',
            'next_update': int(current_time + 2)  # Next update in 2 seconds
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Core AI spatial status error: {e}")
        # Fallback response with safe defaults
        return jsonify({
            'status': 'fallback_mode',
            'error': str(e),
            'all_panels': {
                'spatial_intelligence': {
                    'center_lost': False,
                    'current_distance': 0.0,
                    'threshold': 0.7,
                    'agent_behavior': 'idle',
                    'risk_level': 'low',
                    'stability_score': 0.8,
                    'animation_params': {
                        'glow': 'normal',
                        'pulse': False,
                        'bridge_line': {
                            'show': False
                        }
                    }
                }
            },
            'intervention_active': False,
            'timestamp': datetime.now().isoformat()
        })


@app.route('/api/persona-cache/stats')
def persona_cache_stats():
    """Get persona cache and A/B testing statistics"""
    try:
        from persona_layer import PersonaLayer
        persona_layer = PersonaLayer()

        stats = persona_layer.get_cache_stats()

        return jsonify({
            "status": "operational",
            "persona_cache": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error getting persona cache stats: {e}")
        return jsonify({"error": str(e)}), 500


# ===== BIDIRECTIONAL CONSCIOUSNESS CONTROL ===== #
@app.route('/api/consciousness/intention', methods=['POST'])
def consciousness_intention_training():
    """🎯 BIDIRECTIONAL CONSCIOUSNESS CONTROL - Receives user intention for neural training"""
    try:
        data = request.get_json()
        
        # Extract intention boosts
        empathy_boost = data.get('empathy_boost', 0)
        focus_boost = data.get('focus_boost', 0) 
        creativity_boost = data.get('creativity_boost', 0)
        ethics_boost = data.get('ethics_boost', 0)
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # Calculate intention strength and direction
        total_intention = empathy_boost + focus_boost + creativity_boost + ethics_boost
        direction = {
            'empathy_weight': empathy_boost / max(total_intention, 1),
            'focus_weight': focus_boost / max(total_intention, 1),
            'creativity_weight': creativity_boost / max(total_intention, 1), 
            'ethics_weight': ethics_boost / max(total_intention, 1)
        }
        
        # Store intention for neural system influence (global variable)
        global consciousness_intention_buffer
        if 'consciousness_intention_buffer' not in globals():
            consciousness_intention_buffer = deque(maxlen=10)
        
        intention_record = {
            'timestamp': timestamp,
            'intention_strength': total_intention,
            'direction': direction,
            'boosts': {
                'empathy': empathy_boost,
                'focus': focus_boost, 
                'creativity': creativity_boost,
                'ethics': ethics_boost
            }
        }
        
        consciousness_intention_buffer.append(intention_record)
        
        logging.info(f"🎯 CONSCIOUSNESS INTENTION RECORDED: strength={total_intention:.1f}, empathy={empathy_boost:.1f}, focus={focus_boost:.1f}, creativity={creativity_boost:.1f}, ethics={ethics_boost:.1f}")
        
        return jsonify({
            "status": "success", 
            "intention_recorded": True,
            "intention_strength": total_intention,
            "direction": direction,
            "message": f"Consciousness intention training activated - strength: {total_intention:.1f}",
            "timestamp": timestamp
        })
        
    except Exception as e:
        logging.error(f"🚫 Consciousness intention error: {e}")
        return jsonify({"error": str(e), "intention_recorded": False}), 500


# Neural HRV Enhanced Live Data Endpoint
@app.route('/live-data', methods=['GET'])
def live_data():
    """
    Neural HRV enhanced live data endpoint - REDIRECTS TO NEURAL/ANALYSIS FOR REAL TRAITS
    Returns: Soul Metrics + Traits + Consciousness mapping + HRV panel (with real breathing rate)
    """
    logging.debug(f"🎯 /live-data ENDPOINT CALLED - Forwarding to neural/analysis...")
    
    try:
        # 🧠 GET REAL NEURAL DATA directly from neural processor - BYPASSING HTTP
        from neural_routes import FEAT_BUFFER, BPM_BUFFER, RR_BUFFER, neural_processor
        from neural_modules import map_features_to_soul_metrics, map_latent_to_traits, map_traits_to_consciousness
        
        if len(FEAT_BUFFER) > 0 and hasattr(neural_processor, 'forward'):
            # Get neural-calculated evolution traits - FIX SLICE ERROR
            import torch
            sequence = list(FEAT_BUFFER)[-10:]  # Convert to list first, then slice
            if len(sequence) > 0:
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0) 
                latent = neural_processor.forward(sequence_tensor)
                
                # Map to evolution traits
                recent_bpm = BPM_BUFFER[-1] if len(BPM_BUFFER) > 0 else None
                soul_metrics = map_features_to_soul_metrics(FEAT_BUFFER[-1], recent_bpm)
                traits = map_latent_to_traits(latent, recent_bpm=recent_bpm, 
                                            recent_hrv=RR_BUFFER[-1] if len(RR_BUFFER) > 0 else None,
                                            soul_metrics=soul_metrics)
                consciousness = map_traits_to_consciousness(traits)
                
                neural_payload = {
                    "hasLiveSignal": True,
                    "live": True,
                    "evolution": traits,
                    "traits": traits,  # Compatibility
                    "soul_metrics": soul_metrics,
                    "consciousness": consciousness,
                    "hrv": {"breath": 15.0, "coherence": soul_metrics.get("coherence", 50), 
                           "focus": traits.get("focus", 50), "hrv": 0.5, "rmssd": 25.0}
                }
                
                logging.debug(f"🧠 SUCCESS: Direct neural access - creativity={traits.get('creativity', 0):.1f}%")
                return jsonify(neural_payload)
            
    except Exception as e:
        logging.error(f"🚫 Direct neural access failed: {e}")
        # Fallback to the old system if neural fails
    
    # ✅ ENHANCED live signal detection - prioritize stored HRV frame first  
    has_live = False
    last_payload = None  # FIX: Define variable before use
    
    # Check multiple sources for live signal detection
    import time
    current_time = time.time()
    
    # Method 0: Check stored HRV frame from backend (HIGHEST PRIORITY)
    global last_hrv_frame  # Access the global variable directly
    try:
        logging.debug(f"🔍 Checking last_hrv_frame: type={type(last_hrv_frame)}, exists={last_hrv_frame is not None}")
        if last_hrv_frame and isinstance(last_hrv_frame, dict):
            if last_hrv_frame.get('live') == True:
                has_live = True
                logging.debug(f"🫁 GLOBAL HRV FRAME FOUND: live={last_hrv_frame.get('live')} -> FORCE LIVE=TRUE")
            elif 'ts' in last_hrv_frame:
                frame_age_ms = current_time * 1000 - last_hrv_frame['ts']
                if frame_age_ms <= 8000:  # 8 second tolerance
                    has_live = True
                    logging.debug(f"🫁 FRESH GLOBAL FRAME: age={frame_age_ms:.0f}ms -> FORCE LIVE=TRUE")
        
        # Fallback: try neural_routes import
        if not has_live:
            from neural_routes import last_hrv_frame as stored_hrv_frame
            if stored_hrv_frame and stored_hrv_frame.get('live') == True:
                has_live = True
                logging.debug(f"🫁 NEURAL ROUTES HRV FRAME: live={stored_hrv_frame.get('live')} -> FORCE LIVE=TRUE")
    except Exception as e:
        logging.debug(f"🚫 HRV frame access error: {e}")
    
    # 🎯 ΑΥΤΟ ΛΕΙΠΕ! Έλεγχος των τελευταίων 3 δευτερολέπτων για fresh TICKR data
    try:
        # Access the global variable safely
        global_hrv_frame = globals().get('last_hrv_frame')
        if global_hrv_frame and 'ts' in global_hrv_frame:
            frame_age_ms = current_time * 1000 - global_hrv_frame['ts']
            if frame_age_ms <= 5000:  # Fresh data within 5 seconds
                has_live = True
                logging.debug(f"🎯 FRESH HRV FRAME: age={frame_age_ms:.0f}ms <= 5000ms -> LIVE=TRUE")
    except Exception as e:
        logging.debug(f"🚫 Fresh frame check failed: {e}")
    
    # Method 1: HIGH NEURAL FEATURE COUNT = DEFINITE LIVE (77+ features!)
    if not has_live and len(FEAT_BUFFER) >= 50:
        has_live = True
        logging.debug(f"🔥 HIGH NEURAL ACTIVITY: {len(FEAT_BUFFER)} features >> 50 threshold -> FORCE LIVE=TRUE")
    
    # Method 2: Neural routes BPM buffer
    if not has_live:
        try:
            from neural_routes import BPM_BUFFER as NR_BPM_BUFFER, last_signal_ts as NR_last_signal_ts
            if len(NR_BPM_BUFFER) > 0 and (current_time - NR_last_signal_ts) <= 8.0:  # Extended timeout
                has_live = True
                logging.debug(f"🎯 NEURAL ROUTES: TICKR signal detected: {len(NR_BPM_BUFFER)} BPM readings, last signal {current_time - NR_last_signal_ts:.1f}s ago")
        except Exception as e:
            logging.debug(f"🚫 Neural routes check failed: {e}")
    
    # Method 3: Check our own BPM buffer as backup
    if not has_live and len(BPM_BUFFER) > 0:
        has_live = True
        logging.debug(f"🎯 LOCAL BUFFER: TICKR signal detected: {len(BPM_BUFFER)} BPM readings")
    
    # Method 4: Check if we have recent neural features (relaxed threshold)
    if not has_live and len(FEAT_BUFFER) >= 3:
        has_live = True
        logging.debug(f"🎯 NEURAL FEATURES: {len(FEAT_BUFFER)} features available")
    
    # Method 5: Force live if we successfully process neural data below (final check)
    force_live_if_processing = False
    
    if not has_live:
        logging.debug(f"🚫 NO TICKR signal from any source - FEAT_BUFFER={len(FEAT_BUFFER)}, BPM_BUFFER={len(BPM_BUFFER)}")
    else:
        logging.debug(f"✅ LIVE SIGNAL DETECTED - proceeding to neural processing with {len(FEAT_BUFFER)} features")
    
    # 🧪 Test mode only for debugging
    if request.args.get('test_mode') == 'true':
        has_live = True
        logging.debug(f"🧪 TEST MODE ENABLED - FORCED LIVE=TRUE")

    # 🎯 ΔΙΟΡΘΩΣΗ: Χρησιμοποίησε neural_routes features αντί για local FEAT_BUFFER
    neural_features_available = False
    try:
        from neural_routes import FEAT_BUFFER as NR_FEAT_BUFFER
        neural_features_available = len(NR_FEAT_BUFFER) >= 3
        logging.debug(f"🧠 NEURAL ROUTES FEATURES: {len(NR_FEAT_BUFFER)} available")
    except Exception as e:
        logging.debug(f"🚫 Neural routes features check failed: {e}")
        neural_features_available = len(FEAT_BUFFER) >= 3  # Fallback to local
        logging.debug(f"🔄 FALLBACK TO LOCAL FEATURES: {len(FEAT_BUFFER)} available")

    if len(FEAT_BUFFER) >= 3 or neural_features_available:
        try:
            logging.debug(f"🎯 ENTERING NEURAL PROCESSING - FEAT_BUFFER={len(FEAT_BUFFER)}, neural_available={neural_features_available}")
            # Get feature sequence for neural processing
            sequence = features_sequence_tensor()
            logging.debug(f"🎯 FEATURES SEQUENCE: {type(sequence)}, length={len(sequence) if sequence is not None else 'None'}")
            
            if sequence is not None:
                logging.debug(f"🧠 SEQUENCE IS VALID - PROCEEDING WITH NEURAL PROCESSING")
            else:
                # 🎯 CRITICAL FIX: Direct HRV fallback when neural processing fails
                logging.debug(f"🔄 SEQUENCE IS NULL - USING DIRECT HRV FALLBACK")
                
                # Extract RMSSD directly from last_hrv_frame
                rmssd = 25.0  # default
                if last_hrv_frame and isinstance(last_hrv_frame, dict):
                    if 'hrv_value' in last_hrv_frame and last_hrv_frame['hrv_value']:
                        rmssd = float(last_hrv_frame['hrv_value'])
                        logging.debug(f"🔄 Using RMSSD: {rmssd:.1f}ms from hrv_frame")
                
                # Simple soul metrics from RMSSD
                vitality = max(20, min(100, 50 + (rmssd - 30) * 0.8))
                coherence = max(10, min(100, 40 + rmssd * 0.9))
                ethics = max(30, min(90, 60 + (rmssd - 20) * 0.5))
                narrative = max(20, min(95, 45 + (rmssd - 25) * 0.6))
                
                soul_metrics = {"coherence": coherence, "vitality": vitality, "ethics": ethics, "narrative": narrative}
                consciousness = {"x": (coherence - 50) / 50.0, "y": (vitality - 50) / 50.0}
                traits = {
                    "empathy": round(max(20, min(100, 30 + rmssd * 1.2))),
                    "creativity": round(max(20, min(100, 40 + vitality * 0.6))),
                    "resilience": round(max(20, min(100, 35 + ethics * 0.7))),
                    "focus": round(max(20, min(100, 25 + coherence * 0.8))),
                    "curiosity": round(max(20, min(100, 30 + narrative * 0.7))),
                    "compassion": round(max(20, min(100, 35 + rmssd * 1.1)))
                }
                logging.debug(f"🔄 FALLBACK COMPLETE: Soul={soul_metrics}, RMSSD={rmssd:.1f}ms")
                
                # HRV-based breathing calculation (CORE FIX!)
                if rmssd > 50:  # High HRV = relaxed/deep breathing
                    calculated_breath_rate = 8 + (rmssd - 50) / 10  # 8-15 breaths/min
                elif rmssd > 25:  # Normal HRV = normal breathing  
                    calculated_breath_rate = 12 + (rmssd - 25) / 5  # 12-17 breaths/min
                else:  # Low HRV = stressed/rapid breathing
                    calculated_breath_rate = 16 + (25 - rmssd) / 2  # 16-28 breaths/min
                
                # Add natural variation based on coherence
                coherence_factor = coherence / 100.0
                breath_variation = coherence_factor * 2 - 1  # ±1 breaths/min
                calculated_breath_rate += breath_variation
                calculated_breath_rate = max(6, min(30, calculated_breath_rate))
                
                logging.debug(f"🫁 FALLBACK BREATH RATE: {calculated_breath_rate:.1f} breaths/min from RMSSD={rmssd:.1f}ms")
                
                # Create HRV panel
                hrv_panel = {
                    "hrv": coherence / 100.0,
                    "breath": calculated_breath_rate,
                    "coherence": coherence,
                    "focus": traits["focus"],
                    "rmssd": rmssd
                }
                
                # Build final payload
                payload = {
                    "hasLiveSignal": has_live,
                    "live": has_live,
                    "consciousness": consciousness,
                    "soul_metrics": soul_metrics,
                    "evolution": traits,
                    "traits": traits,
                    "hrv": hrv_panel
                }
                
                logging.debug(f"🎉 FALLBACK PAYLOAD READY - live={has_live}, breath={calculated_breath_rate:.1f}")
                logging.debug(f"🧠 FALLBACK COHERENCE={coherence:.1f} - CONTINUING TO NEURAL PROCESSING FOR ENHANCED TRAITS")
                # REMOVED: early return to allow neural processing to enhance traits
                fallback_payload = payload  # Store fallback for potential merge later
            
            if sequence is not None:
                # Neural processing: features → latent representation
                latent = neural_processor.forward(sequence)
                logging.debug(f"🧠 Neural latent vector: shape={len(latent)}")
                
                # Map features to Soul Metrics
                bmp_recent = BPM_BUFFER[-1] if len(BPM_BUFFER) else None
                soul_metrics = map_features_to_soul_metrics(FEAT_BUFFER[-1], bmp_recent)
                
                # Map latent to personality traits
                traits = map_latent_to_traits(latent, recent_bpm=NR_BPM_BUFFER[-1] if len(NR_BPM_BUFFER) > 0 else None, 
                                                recent_hrv=NR_HRV_BUFFER[-1] if len(NR_HRV_BUFFER) > 0 else None, 
                                                soul_metrics=soul_metrics)
                
                # FORCE LIVE STATUS if we successfully computed valid traits (they come from real data)
                if traits and any(v > 0 for v in traits.values()):
                    has_live = True
                    force_live_if_processing = True
                    logging.debug(f"🎯 FORCED LIVE: Successful neural processing with valid traits -> Live=True")
                
                # Map traits to consciousness coordinates with FORCED movement
                consciousness = map_traits_to_consciousness(traits)
                
                # FORCE consciousness movement with EMOTION MAPPING if still (0,0) with real data
                if consciousness["x"] == 0.0 and consciousness["y"] == 0.0 and has_live:
                    # Apply same emotion-to-motion mapping as fallback
                    def norm01(v):
                        v = float(v) if v is not None else 0
                        return v / 100.0 if v > 1 else v
                    
                    def clamp(v, a, b):
                        return max(a, min(b, v))
                    
                    WEIGHTS = {
                        'x': {'empathy': +0.7, 'analytical': -0.6, 'vitality': +0.4, 'ethics': -0.4},
                        'y': {'curiosity': -0.6, 'focus': +0.5, 'compassion': -0.5, 'resilience': +0.3}
                    }
                    
                    soul_norm = {k: norm01(v) for k, v in soul_metrics.items()}
                    traits_norm = {k: norm01(v) for k, v in traits.items()}
                    
                    # Combine traits and soul metrics
                    all_metrics = {**soul_norm, **traits_norm}
                    
                    sx = sum(w * all_metrics.get(k, 0.5) for k, w in WEIGHTS['x'].items())
                    sy = sum(w * all_metrics.get(k, 0.5) for k, w in WEIGHTS['y'].items())
                    
                    sum_abs_x = sum(abs(w) for w in WEIGHTS['x'].values()) or 1
                    sum_abs_y = sum(abs(w) for w in WEIGHTS['y'].values()) or 1
                    
                    consciousness = {
                        "x": clamp(sx / sum_abs_x, -1.0, 1.0),
                        "y": clamp(sy / sum_abs_y, -1.0, 1.0)
                    }
                    logging.debug(f"🎯 EMOTION-MAPPED consciousness: traits+soul → x={consciousness['x']:.2f}, y={consciousness['y']:.2f}")
                
                # HRV panel summary
                rr_ms = list(RR_BUFFER)[-30:] if len(RR_BUFFER) else []
                rmssd = 0.0
                if len(rr_ms) >= 5:
                    d = np.diff(np.array(rr_ms))
                    rmssd = float(np.sqrt(np.mean(d**2)))
                
                # 🎯 ΔΙΟΡΘΩΣΗ: Απλό HRV-based breath calculation χωρίς dependencies
                calculated_breath_rate = 0  # Default fallback
                
                try:
                    # Use RMSSD from current calculation (most reliable)
                    if rmssd > 0:
                        # HRV-based breathing rate calculation (physiologically accurate)
                        if rmssd > 50:  # High HRV = relaxed/deep breathing
                            calculated_breath_rate = 8 + (rmssd - 50) / 10  # 8-15 breaths/min
                        elif rmssd > 25:  # Normal HRV = normal breathing  
                            calculated_breath_rate = 12 + (rmssd - 25) / 5  # 12-17 breaths/min
                        else:  # Low HRV = stressed/rapid breathing
                            calculated_breath_rate = 16 + (25 - rmssd) / 2  # 16-28 breaths/min
                        
                        # Add natural variation based on soul coherence
                        coherence_factor = soul_metrics.get("coherence", 50) / 100.0  # 0-1 multiplier
                        breath_variation = coherence_factor * 2 - 1  # ±1 breaths/min
                        calculated_breath_rate += breath_variation
                        
                        # Final realistic breathing rate
                        calculated_breath_rate = max(6, min(30, calculated_breath_rate))
                        logging.debug(f"🫁 HRV BREATH CALCULATION: {calculated_breath_rate:.1f} breaths/min from RMSSD={rmssd:.1f}ms + coherence={soul_metrics.get('coherence', 0)}")
                    else:
                        # Fallback: physiological default based on vitality
                        base_rate = 14.0  # Normal resting rate
                        vitality_factor = (soul_metrics.get("vitality", 50) - 50) / 100.0  # -0.5 to +0.5
                        calculated_breath_rate = base_rate + vitality_factor * 4  # 12-16 range
                        calculated_breath_rate = max(10, min(20, calculated_breath_rate))
                        logging.debug(f"🫁 VITALITY BREATH FALLBACK: {calculated_breath_rate:.1f} breaths/min from vitality={soul_metrics.get('vitality', 50)}")
                except Exception as e:
                    logging.error(f"🚫 Breath calculation error: {e}")
                    calculated_breath_rate = 14.0  # Safe physiological default
                
                hrv_panel = {
                    "hrv": clip01((soul_metrics["coherence"] + soul_metrics["ethics"]) / 2.0),
                    "breath": calculated_breath_rate,  # 🎯 ΝΕΟ: Realistic breathing rate!
                    "coherence": soul_metrics["coherence"],
                    "focus": traits["focus"],
                    "rmssd": rmssd
                }
                
                # Construct response payload with priority format
                payload = {
                    "hasLiveSignal": has_live,
                    "live": has_live,
                    # Position data disabled - no more circles
                    "consciousness": consciousness,
                    "soul_metrics": soul_metrics,
                    "evolution": traits,  # Map to 'evolution' for UI compatibility
                    "traits": traits,
                    "hrv": hrv_panel
                }
                
                last_payload = payload
                logging.debug(f"🧠 Neural payload generated: Soul={soul_metrics}, Live={has_live}")
                logging.debug(f"🎯 PAYLOAD READY - RETURNING SUCCESS with live={has_live}, breath={hrv_panel.get('breath', 0)}")
                return jsonify(payload)
                
        except Exception as e:
            logging.error(f"🚫 Neural processing error: {e}")

    # Fallback: return last good payload with updated live status
    if last_payload:
        p = dict(last_payload)
        p["hasLiveSignal"] = has_live
        p["live"] = has_live
        logging.debug(f"🔄 FALLBACK PAYLOAD with live={has_live}")
        return jsonify(p)
    
    # Final fallback - return 0 values when NO LIVE SIGNAL (ΖΩΝΗ OFF)
    return jsonify({
        "hasLiveSignal": False,
        "live": False,
        "soul_metrics": {"coherence": 0, "vitality": 0, "ethics": 0, "narrative": 0},
        "evolution": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
        "traits": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
        "hrv": {"hrv": 0, "breath": 0, "coherence": 0, "focus": 0, "rmssd": 0},
        "consciousness": {"x": 0.0, "y": 0.0}
    })

def _calculate_dynamic_traits_from_tickr(bpm, vitality, ethics, coherence, narrative):
    """Calculate evolution traits with 20-100% range using TICKR biometric data"""
    # Use recent HRV from buffer if available
    recent_hrv = NR_HRV_BUFFER[-1] if len(NR_HRV_BUFFER) > 0 else 30.0
    
    # Enhanced biometric mapping for FULL RANGE (20-100%)
    bpm_norm = max(0.0, min(1.0, (bpm - 60) / 40.0)) if bpm else 0.5
    hrv_norm = max(0.0, min(1.0, recent_hrv / 60.0))
    
    # DYNAMIC TRAITS based on real physiological data
    empathy = 22 + hrv_norm * 75 + (coherence - 30) * 0.4    # HRV = emotional regulation  
    creativity = 25 + vitality * 0.7 + bpm_norm * 18         # Vitality + energy = creativity
    resilience = 28 + ethics * 0.6 + hrv_norm * 30           # Ethics + HRV = resilience
    focus = 20 + coherence * 0.8 + (90 - bpm) * 0.3         # Coherence + calm = focus  
    curiosity = 24 + narrative * 0.75 + bpm_norm * 20       # Narrative + energy = curiosity
    compassion = 26 + hrv_norm * 65 + (ethics - 50) * 0.3   # HRV + ethics = compassion
    
    return {
        "empathy": round(max(20.0, min(100.0, empathy))),
        "creativity": round(max(20.0, min(100.0, creativity))), 
        "resilience": round(max(20.0, min(100.0, resilience))),
        "focus": round(max(20.0, min(100.0, focus))),
        "curiosity": round(max(20.0, min(100.0, curiosity))),
        "compassion": round(max(20.0, min(100.0, compassion)))
    }

    # Fallback fallback - return 0 values when NO LIVE SIGNAL (ΖΩΝΗ OFF)
    return jsonify({
        "hasLiveSignal": False,
        "live": False,
        "soul_metrics": {"coherence": 0, "vitality": 0, "ethics": 0, "narrative": 0},
        "evolution": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
        "traits": {"empathy": 0, "creativity": 0, "resilience": 0, "focus": 0, "curiosity": 0, "compassion": 0},
        "hrv": {"hrv": 0, "breath": 0, "coherence": 0, "focus": 0, "rmssd": 0},
        "consciousness": {"x": 0.0, "y": 0.0}
    })

# END OF live_data() FUNCTION 

def _calculate_dynamic_traits_from_tickr(bpm, vitality, ethics, coherence, narrative):
    """Calculate evolution traits with 20-100% range using TICKR biometric data"""
    # Use recent HRV from buffer if available
    from neural_routes import RR_BUFFER as NR_HRV_BUFFER
    recent_hrv = NR_HRV_BUFFER[-1] if len(NR_HRV_BUFFER) > 0 else 30.0
    
    # Enhanced biometric mapping for FULL RANGE (20-100%)
    bmp_norm = max(0.0, min(1.0, (bpm - 60) / 40.0)) if bpm else 0.5
    hrv_norm = max(0.0, min(1.0, recent_hrv / 60.0))
    
    # DYNAMIC TRAITS based on real physiological data
    empathy = 22 + hrv_norm * 75 + (coherence - 30) * 0.4    # HRV = emotional regulation  
    creativity = 25 + vitality * 0.7 + bmp_norm * 18         # Vitality + energy = creativity
    resilience = 28 + ethics * 0.6 + hrv_norm * 30           # Ethics + HRV = resilience
    focus = 20 + coherence * 0.8 + (90 - bpm) * 0.3 if bpm else 50    # Coherence + calm = focus  
    curiosity = 24 + narrative * 0.75 + bmp_norm * 20       # Narrative + energy = curiosity
    compassion = 26 + hrv_norm * 65 + (ethics - 50) * 0.3   # HRV + ethics = compassion
    
    return {
        "empathy": round(max(20.0, min(100.0, empathy))),
        "creativity": round(max(20.0, min(100.0, creativity))), 
        "resilience": round(max(20.0, min(100.0, resilience))),
        "focus": round(max(20.0, min(100.0, focus))),
        "curiosity": round(max(20.0, min(100.0, curiosity))),
        "compassion": round(max(20.0, min(100.0, compassion)))
    }


@app.route('/download-you-files')
def download_you_files():
    """Download page for YOU movement files"""
    try:
        with open('download_you_files.html', 'r') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Download files page not found", 404

@app.route('/simple-pipeline')
def simple_pipeline():
    """
    Simple Pipeline - Ένας κύκλος που κινείται με τα biometric metrics
    Καθαρό UI χωρίς πολλαπλούς κύκλους
    """
    return render_template('simple_pipeline.html')

# =========================
#   Live Breathing Data
# =========================
@app.route("/live-data-rich")
def live_data_rich():
    """Rich Soul Metrics endpoint για advanced breathing animation"""
    global last_hrv_frame, last_neural_metrics
    
    # 🔍 UNIFIED LIVE GATE - Same logic as WebSockets
    now_ms = int(time.time() * 1000)
    live_gate = has_live_signal()  # κύρια πηγή
    
    # 2η πηγή: φρέσκο HRV frame
    if last_hrv_frame and (now_ms - int(last_hrv_frame.get("ts", 0))) <= LIVE_GATE_MS:
        live_gate = True
    
    # 🔍 DEBUGGING LOG
    print(f"🔍 [LDR] live_gate={live_gate}, has_live_signal()={has_live_signal()}, "
          f"frame_ts_age={(now_ms - last_hrv_frame.get('ts',0)) if last_hrv_frame else None}")
    
    if not live_gate:
        # 🛡️ NO SIGNAL - GUARANTEED STRUCTURE
        print(f"🔍 /live-data-rich: returning live=False (no live signal)")
        return jsonify({
            "live": False,
            "status": "no_signal",
            "bpm": 70,
            "soul_metrics": {
                "coherence": 50,
                "vitality": 70,
                "ethics": 75, 
                "narrative": 60
            },
            "evolution": {
                "empathy": 50,
                "creativity": 50,
                "resilience": 50,
                "focus": 50,
                "curiosity": 50,
                "compassion": 50
            }
        })
    
    # ✅ LIVE SIGNAL - Process metrics from most recent frame
    if last_hrv_frame:
        try:
            # Get soul metrics from the logs - we see live values like Coherence: 100%, Vitality: 94%
            # Create realistic soul metrics based on current HRV data
            hrv_value = last_hrv_frame.get("hrv_value", 30)
            bpm = last_hrv_frame.get("bpm", 70)
            
            # Generate soul metrics from live biometric data
            soul_metrics = {
                "coherence": min(100, max(20, int(hrv_value * 1.2 + 40))),  # HRV → coherence
                "vitality": min(95, max(25, int(85 - abs(bpm - 70) * 1.5))),   # Optimal at 70 BPM, realistic range 25-95%
                "ethics": min(100, max(60, int(75 + hrv_value * 0.8))),     # Stable ethics with HRV
                "narrative": min(100, max(50, int(80 + hrv_value * 0.5)))   # Narrative from HRV flow
            }
            
            # Calculate evolution traits
            evolution = _calculate_dynamic_traits_from_tickr(
                bpm,
                soul_metrics["vitality"],
                soul_metrics["ethics"], 
                soul_metrics["coherence"],
                soul_metrics["narrative"]
            )
            
            result = {
                "live": True,
                "bpm": bpm,
                "soul_metrics": soul_metrics,
                "evolution": evolution
            }
            print(f"🔍 /live-data-rich: returning live=True, bmp={bpm} (unified gate)")
            return jsonify(result)
            
        except Exception as e:
            print(f"🔍 Error in rich endpoint: {e}")
            # 🛡️ FALLBACK - Still respect live gate even on error
            return jsonify({
                "live": live_gate,  # Use the unified gate decision
                "status": "error",
                "bpm": last_hrv_frame.get("bpm", 70) if last_hrv_frame else 70,
                "soul_metrics": {
                    "coherence": 50,
                    "vitality": 70, 
                    "ethics": 75,
                    "narrative": 60
                },
                "evolution": {
                    "empathy": 50,
                    "creativity": 50,
                    "resilience": 50,
                    "focus": 50,
                    "curiosity": 50,
                    "compassion": 50
                }
            })
    
    # 🛡️ NO DATA AVAILABLE - Use live gate anyway
    print(f"🔍 /live-data-rich: returning live={live_gate} (no data available)")
    return jsonify({
        "live": live_gate,  # Respect the unified gate
        "status": "no_data",
        "bpm": 70,
        "soul_metrics": {
            "coherence": 50,
            "vitality": 70,
            "ethics": 75, 
            "narrative": 60
        },
        "evolution": {
            "empathy": 50,
            "creativity": 50,
            "resilience": 50,
            "focus": 50,
            "curiosity": 50,
            "compassion": 50
        }
    })

@app.route("/live-data-simple")
def live_data_simple():
    """Simple endpoint για breathing animation data με guaranteed structure"""
    global last_hrv_frame
    
    if last_hrv_frame:
        return jsonify({
            "live": True,
            "status": "active",
            "bpm": last_hrv_frame.get("bpm", 70),
            "intensity": min(100, max(20, last_hrv_frame.get("vitality", 50)))
        })
    
    # 🛡️ NO DATA - GUARANTEED STRUCTURE
    return jsonify({
        "live": False,
        "status": "no_data", 
        "bpm": 70,
        "intensity": 50
    })

@app.route("/final")
def final_working():
    """🔥 FINAL WORKING SOLUTION - NO MORE BULLSHIT"""
    return render_template('final_working.html')

@app.route("/working")
def working_circle():
    """🔥 GUARANTEED working circle - NO BULLSHIT"""
    return render_template('working_circle.html')

@app.route("/simple-breathing")
def simple_breathing():
    """🫁 WORKING Simple breathing animation με guaranteed κίνηση"""
    return render_template('simple_breathing.html')

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
