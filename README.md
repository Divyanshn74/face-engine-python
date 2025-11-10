# Face Engine API (Python)

This repository provides the **face recognition engine** for the Face Recognition Attendance System.  
It is a lightweight **FastAPI** service that performs face detection, feature extraction, and recognition using **OpenCV** and **NumPy**.  
The Java Spring Boot backend communicates with this API to register and recognize student faces.

---

## 🧩 Overview

The **Face Engine API** exposes REST endpoints to:
- Register a student's face (store facial encodings).
- Recognize a face by comparing it with existing encodings.
- Return results (match / not match) to the Java backend.

---

## 🏗️ Project Structure

```
face-engine-python/
├── app.py             # Main FastAPI app
├── requirements.txt   # Dependencies list
├── Procfile           # For deployment (e.g., Heroku/Render)
├── runtime.txt        # Python runtime version
└── (data storage / encodings as implemented in app.py)
```

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|-------------|
| **Framework** | FastAPI |
| **Face Recognition** | OpenCV |
| **Language** | Python 3.10+ |
| **Libraries** | NumPy, uvicorn |
| **Deployment** | Local / Render / Railway / Heroku |

---

## 🚀 Setup & Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Divyanshn74/face-engine-python.git
cd face-engine-python
```

### 2️⃣ Install dependencies
It’s recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # (on Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 3️⃣ Run the API
```bash
python app.py
```

By default, the API will run on port **5001**:
```
http://127.0.0.1:5001
```

You can access the built-in documentation at:
```
http://127.0.0.1:5001/docs
```

---

## 🔗 Java Backend Integration

Your **Java Spring Boot** application should reference this API in its `application.properties` file:

```properties
face.engine.url=http://127.0.0.1:5001
```

The Java backend will send image data to this API via endpoints like:

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/register-face` | `POST` | Save face encoding for a student |
| `/recognize` | `POST` | Recognize and match an uploaded face |

---

## 🌐 Deploying Online

You can host this API on any Python-friendly platform such as:

### 🟣 Render
1. Create a new Web Service.  
2. Connect your GitHub repo (`Divyanshn74/face-engine-python`).  
3. Use the following settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
4. After deployment, note your service URL (e.g. `https://face-engine.onrender.com`).

### 🟢 Railway
1. Connect your GitHub repository.  
2. Railway automatically detects and builds FastAPI projects.  
3. After deploy, get your public URL.

### 🔵 Heroku (legacy method)
1. Install Heroku CLI and log in.  
2. Run:
   ```bash
   heroku create face-engine-api
   git push heroku main
   ```
3. Your API will be live at something like:  
   `https://face-engine-api.herokuapp.com`

Update your Java backend with the new public URL.

---

## 🧠 API Example (FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np

app = FastAPI()

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # TODO: Implement recognition logic here
    return {"match": True, "student_id": 1}
```

---

## 🧑‍💻 Authors & Contributors

**Project Developed By:**  
- 👨‍💻 **Divyansh Namdev** (Divyanshn74)

**Team Members:**  
- Gopal Kumar Saw  
- Vivek Pushptode  
- Neha Rathor  

Java backend repository: *(to be linked once public)*

---

## 🪪 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
