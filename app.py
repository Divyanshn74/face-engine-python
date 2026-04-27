from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import base64
import numpy as np
import cv2
import time
import logging
import faiss
import json
import os
import mysql.connector
import mediapipe as mp

from insightface.app import FaceAnalysis


IS_GPU_NVIDIA = os.environ.get("IS_GPU_NVIDIA", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if allowed_origins == [""]:
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

API_KEY = os.environ.get("FACE_ENGINE_API_KEY", "default-secret-key")

@app.before_request
def require_api_key():
    if request.path in ['/', '/health', '/gpu_status', '/index_status']:
        return
    
    if request.method == 'OPTIONS':
        return
        
    auth_header = request.headers.get('X-API-Key')
    if auth_header != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "D1v2a3s4@"),
    "database": os.environ.get("DB_NAME", "attendance_db")
}

faiss_index = None
student_mapping = []
gpu_resources = None


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def build_faiss_index():
    global faiss_index, student_mapping
    
    logger.info("Building FAISS index from database...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, roll_no, face_embedding 
            FROM students 
            WHERE face_embedding IS NOT NULL
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            logger.warning("No students with embeddings found in database")
            faiss_index = None
            student_mapping = []
            return
        
        embeddings_list = []
        student_mapping = []
        
        for row in rows:
            try:
                embedding_str = row['face_embedding']
                
                if embedding_str.startswith('['):
                    embedding = json.loads(embedding_str)
                else:
                    embedding = [float(x) for x in embedding_str.split(',')]
                
                embeddings_list.append(np.array(embedding, dtype=np.float32))
                student_mapping.append({
                    "id": row['id'],
                    "roll_no": row['roll_no']
                })
            except Exception as e:
                logger.warning("Failed to parse embedding for student %d: %s", row['id'], e)
                continue
        
        if not embeddings_list:
            logger.warning("No valid embeddings found")
            faiss_index = None
            student_mapping = []
            return
        
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        
        dimension = 512
        
        if IS_GPU_NVIDIA:
            try:
                gpu_resources = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatL2(dimension)
                cpu_index.add(embeddings_matrix)
                faiss_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
                logger.info("✓ FAISS GPU index built: %d students indexed", len(student_mapping))
            except Exception as gpu_err:
                logger.warning("FAISS GPU failed (%s), falling back to CPU", gpu_err)
                faiss_index = faiss.IndexFlatL2(dimension)
                faiss_index.add(embeddings_matrix)
                logger.info("✓ FAISS CPU index built (fallback): %d students indexed", len(student_mapping))
        else:
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(embeddings_matrix)
            logger.info("✓ FAISS index built: %d students indexed", len(student_mapping))
        
    except Exception as e:
        logger.error("Failed to build FAISS index: %s", e)
        faiss_index = None
        student_mapping = []


def rebuild_index():
    build_faiss_index()


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception occurred")
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "status": 500,
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
        "path": request.path
    }
    
    return jsonify(response), 500

try:
    logger.info("Loading InsightFace ArcFace model (buffalo_l)...")
    
    if IS_GPU_NVIDIA:
        insightface_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("GPU Mode: Requesting CUDAExecutionProvider for InsightFace")
    else:
        insightface_providers = ["CPUExecutionProvider"]
        logger.info("CPU Mode: Using CPUExecutionProvider for InsightFace")
    
    face_app = FaceAnalysis(name="buffalo_l", providers=insightface_providers)
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("InsightFace ArcFace model loaded successfully.")
except Exception as e:
    face_app = None
    logger.error("Failed to load InsightFace model: %s", str(e))


def decode_base64_image(b64_string):
    try:
        img_bytes = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error("decode_base64_image: %s", e)
        return None


def get_embedding_from_image(img):
    if face_app is None:
        return None, "Model not loaded"
    
    try:
        start_time = time.time()
        
        faces = face_app.get(img)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        num_faces = len(faces)
        
        if num_faces == 0:
            logger.warning("No face detected in image")
            return None, "No face detected"
        
        if num_faces > 1:
            logger.warning("Multiple faces detected: %d", num_faces)
            return None, f"Multiple faces detected ({num_faces})"
        
        embedding = faces[0].embedding.astype(np.float32)
        
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
        
        logger.info("Inference time: %.2f ms | Embedding length: %d | Norm after L2: %.4f", 
                    inference_time_ms, len(embedding), np.linalg.norm(embedding))
        
        return embedding, None
        
    except Exception as e:
        logger.error("get_embedding_from_image: %s", e)
        return None, str(e)


def cosine_distance(a, b):
    a = np.array(a, dtype=np.float64).flatten()
    b = np.array(b, dtype=np.float64).flatten()
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if norma == 0 or normb == 0:
        return 1.0
    cosine = dot / (norma * normb)
    return float(1.0 - cosine)


@app.route("/get_embedding", methods=["POST"])
def get_embedding():
    try:
        data = request.get_json(force=True)
        base64_image = data.get("image")
        if not base64_image:
            return jsonify({"error": "Missing 'image' field"}), 400

        img = decode_base64_image(base64_image)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        embedding, error = get_embedding_from_image(img)
        
        if error:
            if "No face" in error or "Multiple faces" in error:
                return jsonify({"error": error}), 400
            return jsonify({"error": "Failed to compute embedding"}), 500

        return jsonify({"embedding": embedding.tolist()}), 200
    except Exception as e:
        logger.error("/get_embedding: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/compare", methods=["POST"])
def compare_faces():
    try:
        data = request.get_json(force=True)
        known_embedding = data.get("known_embedding")
        unknown_b64 = data.get("unknown_image")

        if known_embedding is None or unknown_b64 is None:
            return jsonify({"error": "Both 'known_embedding' and 'unknown_image' are required"}), 400

        img = decode_base64_image(unknown_b64)
        if img is None:
            return jsonify({"error": "Failed to decode unknown_image"}), 400

        live_embedding, error = get_embedding_from_image(img)
        
        if error:
            if "No face" in error or "Multiple faces" in error:
                return jsonify({"error": error}), 400
            return jsonify({"error": "Failed to compute embedding for unknown_image"}), 500

        dist = cosine_distance(known_embedding, live_embedding)
        threshold = 0.5
        match = dist <= threshold

        return jsonify({"match": bool(match), "distance": round(dist, 4)}), 200
    except Exception as e:
        logger.error("/compare: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/identify", methods=["POST"])
def identify():
    global faiss_index, student_mapping
    
    try:
        data = request.get_json(force=True)
        base64_image = data.get("image")
        
        if not base64_image:
            return jsonify({"error": "Missing 'image' field"}), 400
        
        if faiss_index is None or len(student_mapping) == 0:
            return jsonify({"error": "FAISS index not initialized. No students in database."}), 503
        
        img = decode_base64_image(base64_image)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        live_embedding, error = get_embedding_from_image(img)
        if error:
            if "No face" in error or "Multiple faces" in error:
                return jsonify({"error": error}), 400
            return jsonify({"error": "Failed to compute embedding"}), 500
        
        query_vector = live_embedding.astype(np.float32).reshape(1, -1)
        
        start = time.time()
        D, I = faiss_index.search(query_vector, 1)
        search_time_ms = (time.time() - start) * 1000
        
        best_idx = int(I[0][0])
        l2_distance = float(D[0][0])
        
        cosine_distance_val = l2_distance / 2
        
        best_student = student_mapping[best_idx]
        
        logger.info("FAISS Identify: %d students | search: %.3f ms | cosine_dist: %.4f",
                    len(student_mapping), search_time_ms, cosine_distance_val)
        
        threshold = 0.5
        if cosine_distance_val <= threshold:
            return jsonify({
                "match": True,
                "student_id": best_student["id"],
                "roll_no": best_student["roll_no"],
                "distance": round(cosine_distance_val, 4)
            }), 200
        else:
            return jsonify({
                "match": False,
                "distance": round(cosine_distance_val, 4),
                "message": "Unknown person"
            }), 200
            
    except Exception as e:
        logger.error("/identify: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/rebuild_index", methods=["POST"])
def api_rebuild_index():
    try:
        rebuild_index()
        return jsonify({
            "success": True,
            "students_indexed": len(student_mapping)
        }), 200
    except Exception as e:
        logger.error("/rebuild_index: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/index_status", methods=["GET"])
def index_status():
    return jsonify({
        "indexed": faiss_index is not None,
        "students_count": len(student_mapping)
    }), 200


try:
    print("Initializing MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    print("✓ MediaPipe FaceMesh initialized successfully.")
    logger.info("MediaPipe FaceMesh initialized successfully.")
except Exception as e:
    face_mesh = None
    print(f"✗ Failed to initialize MediaPipe FaceMesh: {e}")
    logger.error("Failed to initialize MediaPipe FaceMesh: %s", e)


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.2


def calculate_ear(landmarks, eye_indices, img_width, img_height):
    try:
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            x = int(lm.x * img_width)
            y = int(lm.y * img_height)
            points.append((x, y))

        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

        if h == 0:
            return 0.0

        return (v1 + v2) / (2.0 * h)

    except Exception as e:
        logger.error("EAR calculation error: %s", e)
        return 0.0


def detect_blink_in_frame(img):
    if face_mesh is None:
        return None, False

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        return None, False

    landmarks = results.multi_face_landmarks[0].landmark

    left_ear = calculate_ear(landmarks, LEFT_EYE, img_width, img_height)
    right_ear = calculate_ear(landmarks, RIGHT_EYE, img_width, img_height)

    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, True


@app.route("/liveness_check", methods=["POST"])
def liveness_check():
    try:
        data = request.get_json(force=True)
        frames_b64 = data.get("frames", [])

        if not frames_b64:
            return jsonify({"error": "Missing 'frames' field"}), 400

        if len(frames_b64) < 2:
            return jsonify({"error": "At least 2 frames required"}), 400

        ear_values = []
        blink_detected = False
        frames_with_face = 0

        for i, frame_b64 in enumerate(frames_b64):

            img = decode_base64_image(frame_b64)
            if img is None:
                logger.warning("Frame %d: decode failed", i)
                continue

            ear, face_found = detect_blink_in_frame(img)

            if not face_found:
                logger.warning("Frame %d: No face detected", i)
                continue

            frames_with_face += 1
            ear_values.append(ear)

            logger.info("Frame %d: EAR = %.4f", i, ear)

            if ear < EAR_THRESHOLD:
                blink_detected = True
                logger.info("Frame %d: BLINK detected (EAR=%.4f)", i, ear)

        if frames_with_face == 0:
            return jsonify({"error": "No face detected in any frame"}), 400

        logger.info(
            "Liveness summary | frames: %d | with_face: %d | blink: %s",
            len(frames_b64),
            frames_with_face,
            blink_detected
        )

        return jsonify({
            "liveness": blink_detected,
            "frames_analyzed": frames_with_face,
            "ear_values": [round(e, 4) for e in ear_values]
        }), 200

    except Exception as e:
        import traceback
        import sys
        print("=" * 60)
        print("LIVENESS CHECK ERROR:")
        print(f"Exception type: {type(e)}")
        print(f"Exception: {repr(e)}")
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        print("=" * 60)
        logger.error("/liveness_check: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/secure_identify", methods=["POST"])
def secure_identify():
    global faiss_index, student_mapping
    
    start_total_time = time.time()
    
    try:
        data = request.get_json(force=True)
        frames_b64 = data.get("frames", [])
        
        if not frames_b64:
            return jsonify({"error": "Missing 'frames' field"}), 400
        
        if len(frames_b64) < 2:
            return jsonify({"error": "At least 2 frames required"}), 400
            
        if faiss_index is None or len(student_mapping) == 0:
             return jsonify({"error": "FAISS index not initialized"}), 503

        blink_detected = False
        frames_with_face = 0
        last_valid_frame_img = None
        
        for i, frame_b64 in enumerate(frames_b64):
            img = decode_base64_image(frame_b64)
            if img is None:
                continue
                
            ear, face_found = detect_blink_in_frame(img)
            
            if face_found:
                frames_with_face += 1
                last_valid_frame_img = img
                
                if ear < EAR_THRESHOLD:
                    blink_detected = True
        
        if frames_with_face == 0:
            return jsonify({"error": "No face detected in any frame"}), 400
            
        if not blink_detected:
            logger.info("Secure Identify: Liveness FAILED (No blink detected)")
            return jsonify({
                "match": False,
                "liveness": False,
                "message": "Liveness check failed (blink required)"
            }), 200
            
        if last_valid_frame_img is None:
             return jsonify({"error": "Unexpected error: No valid frame for identification"}), 500

        live_embedding, error = get_embedding_from_image(last_valid_frame_img)
        if error:
            logger.error("Secure Identify: Failed to extract embedding from last frame: %s", error)
            return jsonify({"error": "Failed to extract face features from last frame"}), 500
            
        query_vector = live_embedding.astype(np.float32).reshape(1, -1)
        
        start_search = time.time()
        D, I = faiss_index.search(query_vector, 1)
        search_time_ms = (time.time() - start_search) * 1000
        
        best_idx = int(I[0][0])
        l2_distance = float(D[0][0])
        cosine_distance_val = l2_distance / 2
        
        best_student = student_mapping[best_idx]
        threshold = 0.5
        
        is_match = cosine_distance_val <= threshold
        
        total_time_ms = (time.time() - start_total_time) * 1000
        
        logger.info(
            "Secure Identify: Liveness=TRUE | Match=%s | Dist=%.4f | Student=%s | Search=%.2f ms | Total=%.2f ms",
            is_match, cosine_distance_val, best_student["roll_no"] if is_match else "Unknown", search_time_ms, total_time_ms
        )
        
        if is_match:
            return jsonify({
                "match": True,
                "liveness": True,
                "student_id": best_student["id"],
                "roll_no": best_student["roll_no"],
                "distance": round(cosine_distance_val, 4)
            }), 200
        else:
            return jsonify({
                "match": False,
                "liveness": True,
                "distance": round(cosine_distance_val, 4),
                "message": "Unknown person"
            }), 200
            
    except Exception as e:
        logger.error("/secure_identify: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "message": "🚀 Face Engine API by Divyansh ❤️ is running perfectly!",
        "developer": "Divyansh",
        "powered_by": ["Flask", "InsightFace", "ArcFace", "FAISS"],
        "gpu_enabled": IS_GPU_NVIDIA,
        "faiss_index": {
            "active": faiss_index is not None,
            "students_indexed": len(student_mapping)
        }
    }, 200


@app.route("/gpu_status", methods=["GET"])
def gpu_status():
    return jsonify({
        "gpu_enabled": IS_GPU_NVIDIA,
        "components": {
            "insightface": {
                "gpu_active": IS_GPU_NVIDIA,
                "provider": "CUDAExecutionProvider" if IS_GPU_NVIDIA else "CPUExecutionProvider",
                "impact": "5-10x faster face detection and embedding extraction"
            },
            "faiss": {
                "gpu_active": IS_GPU_NVIDIA,
                "index_type": "GpuIndexFlatL2" if IS_GPU_NVIDIA else "IndexFlatL2",
                "impact": "2-5x faster nearest-neighbor search at scale (>1000 students)"
            },
            "opencv": {
                "gpu_active": False,
                "reason": "Image decode/cvtColor already <1ms on CPU, GPU transfer overhead negates gain",
                "impact": "No change — CPU is optimal for these operations"
            },
            "mediapipe": {
                "gpu_active": False,
                "reason": "MediaPipe Python API does not expose GPU provider toggle",
                "impact": "No change — uses internal TFLite acceleration"
            }
        }
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "face-engine",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200


if __name__ == "__main__":
    port = 5001
    logger.info("🚀 Face Engine API is starting...")
    logger.info("🖥️  GPU Mode: %s", "ENABLED (NVIDIA CUDA)" if IS_GPU_NVIDIA else "DISABLED (CPU only)")
    
    build_faiss_index()
    
    logger.info("➡️  Running on http://localhost:%d", port)
    logger.info("🧠  Model: InsightFace ArcFace (buffalo_l) [%s]", "CUDA" if IS_GPU_NVIDIA else "CPU")
    logger.info("📊  FAISS: %d students indexed [%s]", len(student_mapping), "GPU" if IS_GPU_NVIDIA else "CPU")
    logger.info("Press CTRL+C to stop.")
    app.run(host="0.0.0.0", port=port, debug=False)
