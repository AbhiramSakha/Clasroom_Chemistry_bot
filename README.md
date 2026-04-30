<<<<<<< HEAD
# ⚗️ Chemistry AI Assistant
**KIET University · JNTU Kakinada Affiliated**
B.Tech / B.Sc Chemistry AI Platform — Academic Project

---

## 📋 Project Overview

An AI-powered chemistry learning platform that combines:
- **FLAN-T5 + LoRA** fine-tuned language model for chemistry Q&A
- **Claude AI** for enhanced chat, quiz generation, and notes
- **RDKit** for 2D molecular structure visualization
- **PyMuPDF** for PDF text extraction and analysis
- **MongoDB** for query history persistence
- **FastAPI** backend + **React** frontend

---

## 🚀 Features

| Module | Technology | Description |
|--------|-----------|-------------|
| Ask Chemistry | FLAN-T5 + Wikipedia | Q&A with periodic table + molar mass |
| Formulas | FLAN-T5 | Formula sheets by topic |
| Structures | RDKit + AI SVG | 2D molecular diagrams (50+ compounds) |
| PDF Analyzer | PyMuPDF + FLAN-T5 | Summary, Quiz, Video Script from PDFs |
| Smart Notes | Claude AI | Exam-ready structured notes |
| Video Script | FLAN-T5 | Narrated video explanation scripts |
| AI Quiz | Claude AI | Interactive MCQ with explanations |
| Chemistry Chat | Claude AI | Context-aware tutor (no repetition) |

---

## 🛠️ Tech Stack

**Backend:**
- Python 3.10+
- FastAPI 0.104+
- PyTorch (CPU)
- HuggingFace Transformers (FLAN-T5)
- PEFT (LoRA adapter)
- RDKit (molecular rendering)
- PyMuPDF (PDF extraction)
- MongoDB + pymongo
- googletrans (multi-language)

**Frontend:**
- React 18 + Vite
- Anthropic Claude API (chat, quiz, notes)
- CSS-in-JS (no external CSS framework)

---

## 📦 Installation

### Backend Setup
```bash
# 1. Create virtual environment
python -m venv chemenv
source chemenv/bin/activate  # Linux/Mac
chemenv\Scripts\activate     # Windows

# 2. Install dependencies
pip install fastapi uvicorn pymongo python-multipart
pip install torch transformers peft
pip install rdkit pymupdf wikipedia googletrans==4.0.0rc1

# 3. Start MongoDB (ensure MongoDB is running)
mongod --dbpath ./data/db

# 4. Run FastAPI server
uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
# 1. Create React app with Vite
npm create vite@latest chemistry-ai -- --template react
cd chemistry-ai

# 2. Install dependencies
npm install

# 3. Copy Dashboard.jsx to src/components/
# 4. Start development server
npm run dev
# Frontend available at http://localhost:5173
```

---

## 📁 Project Structure

```
chemistry-ai/
├── backend/
│   ├── main.py          ← FastAPI routes
│   ├── model.py         ← FLAN-T5 + LoRA inference
│   ├── auth.py          ← JWT authentication
│   ├── schemas.py       ← Pydantic models
│   ├── database.py      ← MongoDB connection
│   └── MyFinetunedModel/ ← LoRA adapter weights
│
└── frontend/
    └── src/
        └── components/
            └── Dashboard.jsx  ← Main dashboard UI
```

---

## 🔑 Key Improvements (v3.0)

### 1. Fixed Chemistry Chat (No Repetition)
- Now uses **Claude AI** directly with proper system prompt
- Context-aware multi-turn conversation
- Chemistry-only filtering with polite redirect

### 2. AI Molecular Image Generation
- New **AI SVG mode** can draw any compound (glucose, aspirin, caffeine, etc.)
- Original **RDKit mode** for publication-grade SMILES-based structures
- 50+ compounds in SMILES library (up from 6)

### 3. Better FLAN-T5 Output
- `repetition_penalty=2.0` (was 1.5)
- `no_repeat_ngram_size=4` (was 3)
- Post-processing removes repeated sentences
- Cleans model artifacts from output

### 4. Enhanced UI
- System status indicators in sidebar
- Quick example chips in all sections
- Better error messages
- Animated chat typing indicator
- Score visualization in Quiz section

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /predict | Chemistry Q&A |
| GET | /history | Query history |
| POST | /structure | Molecular structure image |
| POST | /pdf-analyze | PDF analysis |
| POST | /translate | Text translation |

---

## 📊 Supported Languages

English, Telugu, Hindi, Tamil, Kannada, French, German, Spanish, Chinese, Japanese, Arabic

---

## 👨‍💻 Development Team

- **University**: KIET College, Korangi
- **Affiliation**: Jawaharlal Nehru Technological University Kakinada (JNTU-K)
- **Department**: Computer Science / Chemistry

---

## 📝 Academic Disclaimer

This tool is developed for educational purposes as part of the JNTU Kakinada curriculum project. AI-generated answers should be verified against standard chemistry textbooks and journals.

---

*Chemistry AI v3.0 · KIET University · JNTU Kakinada*
=======
# KIET_AID_TEAM_6
Classroom Chemistry Bot
📌 Project Overview

Classroom Chemistry Bot is an AI-powered educational assistant designed to help students understand chemistry concepts in an interactive and simple way. The bot acts as a virtual classroom companion that answers chemistry-related questions, explains topics, and supports learning through instant responses.

It is especially useful for school and college students who need quick clarification on chemical concepts, formulas, reactions, and definitions without depending on textbooks or constant teacher assistance.

🎯 Objectives

Simplify complex chemistry concepts

Provide instant academic support to students

Improve classroom learning using AI technology

Encourage self-learning and curiosity

⚙️ Key Features

🔬 Answers chemistry-related questions in real time

📘 Explains chemical formulas, reactions, and theories

🧠 Supports basic to intermediate chemistry topics

💬 User-friendly conversational interface

⏱️ Saves time by providing quick explanations

🛠️ Technologies Used

Python

Natural Language Processing (NLP)

AI / Machine Learning concepts

Flask (for web-based interface – optional)

👨‍🎓 Use Cases

Classroom doubt clarification

Exam preparation support

Self-study assistance

Chemistry revision tool

🚀 Future Enhancements

Voice-based interaction

Image-based chemical equation recognition

Multi-language support

Student performance tracking

📚 Conclusion

The Classroom Chemistry Bot demonstrates how AI can be effectively used in education to make learning more interactive, accessible, and engaging. It bridges the gap between students and complex chemistry topics through smart automation.

# Model
Download Model's from https://drive.google.com/drive/folders/1d-JBh6k-ArhK1ltgs-lwuAcO8G-kvVum?usp=sharing & https://drive.google.com/drive/folders/1Ctf_nnNlio91wJ01xB-L4C98FniZPSAL?usp=sharing
>>>>>>> f5fee160c8126f9036f92e993ca565b0fdc307b9
