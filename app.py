from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from deepface import DeepFace
import traceback
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load Facenet model
try:
    model = DeepFace.build_model("Facenet")
    print("[INFO] Facenet model loaded successfully.")
except Exception as e:
    model = None
    print("[ERROR] Failed to load Facenet model:", str(e))
    traceback.print_exc()

def decode_base64_image(b64_string):
    try:
        img_bytes = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("[ERROR] decode_base64_image:", e)
        return None

def get_embedding_from_image(img):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, img)

        embedding_data = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=False
        )

        if isinstance(embedding_data, list):
            embedding_data = embedding_data[0]["embedding"]

        return np.array(embedding_data).flatten()
    except Exception as e:
        print("[ERROR] get_embedding_from_image:", e)
        traceback.print_exc()
        return None
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

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

        embedding = get_embedding_from_image(img)
        if embedding is None:
            return jsonify({"error": "Failed to compute embedding"}), 500

        return jsonify({"embedding": embedding.tolist()}), 200
    except Exception as e:
        print("[ERROR] /get_embedding:", e)
        traceback.print_exc()
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

        live_embedding = get_embedding_from_image(img)
        if live_embedding is None:
            return jsonify({"error": "Failed to compute embedding for unknown_image"}), 500

        dist = cosine_distance(known_embedding, live_embedding)
        threshold = 0.4
        match = dist <= threshold

        return jsonify({"match": bool(match), "distance": round(dist, 4)}), 200
    except Exception as e:
        print("[ERROR] /compare:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# JSON root route
@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "message": "🚀 Face Engine API by Divyansh ❤️ is running perfectly!",
        "developer": "Divyansh",
        "powered_by": ["Flask", "DeepFace", "TensorFlow"],
    }, 200

if __name__ == "__main__":
    port = 5001
    print(f"\\n🚀 Face Engine API is starting...")
    print(f"➡️  Running on http://localhost:{port}")
    print(f"🧠  Model: Facenet")
    print(f"Press CTRL+C to stop.\\n")
    app.run(host="0.0.0.0", port=port, debug=False)