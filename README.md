# 🧠 Face Engine API (Python)

> The dedicated **AI Microservice** for the Face Recognition Attendance System. This Python backend handles real-time face detection, liveness verification, and sub-millisecond vector similarity search.

---

## 🧩 Overview

The **Face Engine API** is a highly optimized **Flask** service that acts as the core biometric engine for the Java Spring Boot backend. 
It leverages state-of-the-art deep learning models to:
1. **Extract 512-dimensional facial embeddings** from images.
2. **Verify Liveness** by analyzing sequential frames for blink detection (preventing photo spoofing).
3. **Perform instantaneous face matching** against thousands of registered students using a loaded FAISS index.

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|-------------|
| **Framework** | Python 3.11, Flask |
| **Face Recognition** | InsightFace (ArcFace `buffalo_l`) |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Liveness Detection** | MediaPipe FaceMesh (Eye Aspect Ratio tracking) |
| **Image Processing** | OpenCV, NumPy |
| **Database Connector** | MySQL Connector (direct FAISS sync) |
| **Acceleration** | Optional NVIDIA CUDA GPU support |

---

## 🏗️ Project Structure

```
face-engine-python/
├── app.py             # Main Flask AI Engine (handles routes, FAISS, InsightFace)
├── requirements.txt   # Python dependencies
├── Procfile           # Configuration for cloud deployments (e.g., Render)
└── runtime.txt        # Specifies Python version for cloud buildpacks
```

---

## 🔒 Security

This microservice is strictly isolated and **must not be exposed to the public internet** without protection.
- **API Key Authentication:** All sensitive endpoints require an `X-API-Key` header matching the `FACE_ENGINE_API_KEY` environment variable.
- **CORS:** Cross-Origin Resource Sharing is strictly controlled via the `ALLOWED_ORIGINS` environment variable.

---

## 🚀 Setup & Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Divyanshn74/face-engine-python.git
cd face-engine-python
```

### 2️⃣ Configure Environment Variables
Set the following environment variables (or rely on the defaults if running locally alongside the Java app):

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=YOUR_MYSQL_PASSWORD
DB_NAME=attendance_db
FACE_ENGINE_API_KEY=default-secret-key
IS_GPU_NVIDIA=false # Set to 'true' if you have CUDA drivers installed
```

### 3️⃣ Install dependencies
*Note: We highly recommend using a virtual environment (`venv` or `conda`).*
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Run the API
```bash
python app.py
```
The API will initialize the models, build the FAISS index from the MySQL database, and run on: **http://127.0.0.1:5001**

---

## 🔌 Core API Endpoints

*Note: All POST endpoints require the `X-API-Key` header.*

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/secure_identify` | `POST` | Evaluates liveness across multiple frames and identifies the face via FAISS. |
| `/get_embedding` | `POST` | Extracts and returns the 512-d ArcFace embedding from a single image. |
| `/identify` | `POST` | Performs a pure FAISS nearest-neighbor search (no liveness check). |
| `/compare` | `POST` | Compares a known embedding against a live image using cosine distance. |
| `/liveness_check` | `POST` | Analyzes a sequence of frames for blink detection (EAR tracking). |
| `/rebuild_index` | `POST` | Forces the engine to re-query the MySQL DB and rebuild the FAISS index. |
| `/gpu_status` | `GET` | Returns hardware acceleration diagnostic info. |
| `/health` | `GET` | Standard health check endpoint. |

---

## 🌐 Integration with Java Backend

The main [Java Spring Boot Repository](https://github.com/Divyanshn74/face-recog-regular) communicates with this engine. In the Java app's `application.properties` or `.env` file, ensure these variables are mapped:

```properties
FACE_ENGINE_URL=http://localhost:5001
FACE_ENGINE_API_KEY=default-secret-key
```

---

## 🧑‍💻 Author

**Divyansh Namdev** ([Divyanshn74](https://github.com/Divyanshn74))

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.