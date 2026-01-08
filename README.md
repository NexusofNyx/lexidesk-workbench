# LeXIDesk Workbench

LeXIDesk is a legal text analysis platform that uses advanced NLP models (CNN-CRF) for sentence boundary detection and summarization. This project consists of a React frontend and a Python FastAPI backend.

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js** (v18+)
- **Python** (v3.9+)

### 2. Setup Frontend
The frontend is built with Vite + React + TypeScript.

```bash
# Install dependencies
npm install

# Start the frontend development server
npm run dev
```
The frontend will run at `http://localhost:5173`.

### 3. Setup Backend
The backend is built with FastAPI and handles the ML inference.

#### Windows (PowerShell)
```powershell
# 1. Create a virtual environment (if not already created)
python -m venv backend/venv

# 2. Activate the virtual environment
.\backend\venv\Scripts\Activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start the backend server
python -m uvicorn backend.main:app --reload --port 8000
```

#### macOS / Linux
```bash
# 1. Create a virtual environment
python3 -m venv backend/venv

# 2. Activate
source backend/venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start server
uvicorn backend.main:app --reload --port 8000
```

The backend API will run at `http://localhost:8000`.
API Documentation (Swagger UI) is available at: `http://localhost:8000/docs`

---

## 🛠️ Project Structure

```
lexidesk-workbench/
├── backend/               # Python Backend
│   ├── models/            # ML Models (CNN, CRF)
│   ├── src/               # Model source code
│   ├── main.py            # FastAPI Entry point
│   └── requirements.txt   # Python Dependencies
├── src/                   # React Frontend
│   ├── components/        # UI Components
│   ├── pages/             # Application Pages
│   └── lib/               # API Clients
└── package.json           # Frontend Dependencies
```

## ✨ helper Commands

We have added a helper script in `package.json` to run the backend easily on Windows:

```bash
# Run the backend server (Windows)
npm run backend
```
