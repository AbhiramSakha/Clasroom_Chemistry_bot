"""
main.py — Chemistry AI FastAPI Backend
KIET University · JNTU Kakinada
----------------------------------------

Endpoints:
  GET  /              → Health check
  POST /predict       → FLAN-T5 Q&A
  GET  /history       → Query history (last 30)
  POST /structure     → RDKit molecular structure image
  POST /pdf-analyze   → PDF/image analysis
  POST /translate     → Text translation
"""

# ================================
# IMPORTS
# ================================
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from auth import router
from schemas import Query
from model import (
    generate_answer,
    analyze_pdf_text,
    translate_text,
    generate_structure_image
)
from database import history_col

from datetime import datetime
import os
import tempfile
import shutil
import fitz  # PyMuPDF


# ================================
# APP INITIALIZATION
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://classroomchemistrybot.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes
app.include_router(router)

# Static folder for molecular images
os.makedirs("structures", exist_ok=True)
app.mount("/structures", StaticFiles(directory="structures"), name="structures")


# ================================
# 1. HEALTH CHECK
# ================================
@app.get("/")
def root():
    return {
        "status": "✅ Chemistry AI API Running",
        "version": "3.0",
        "university": "KIET College, JNTU Kakinada",
        "model": "FLAN-T5 Base + LoRA Adapter",
        "endpoints": {
            "predict":     "POST /predict",
            "history":     "GET /history",
            "structure":   "POST /structure",
            "pdf_analyze": "POST /pdf-analyze",
            "translate":   "POST /translate"
        }
    }


# ================================
# 2. PREDICT — Q&A
# ================================
@app.post("/predict")
def predict(q: Query):
    if not q.text or not q.text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question text is required."}
        )

    language = getattr(q, "language", "en") or "en"

    output = generate_answer(q.text.strip(), language=language)

    # Save history
    try:
        history_col.insert_one({
            "input": q.text,
            "output": output,
            "language": language,
            "time": datetime.utcnow()
        })
    except Exception as e:
        print(f"[MongoDB Error] {e}")

    return {"output": output}


# ================================
# 3. HISTORY
# ================================
@app.get("/history")
def history():
    try:
        data = list(
            history_col.find()
            .sort("time", -1)
            .limit(30)
        )

        return [
            {"input": d["input"], "output": d["output"]}
            for d in data
        ]

    except Exception as e:
        print(f"[MongoDB Error] {e}")
        return []


# ================================
# 4. STRUCTURE — RDKit
# ================================
@app.post("/structure")
def structure(q: Query):
    if not q.text:
        return {
            "image_url": None,
            "message": "Please provide a compound name."
        }

    compound = q.text.strip().lower()
    file_path = generate_structure_image(compound)

    if file_path is None:
        return {
            "image_url": None,
            "message": f"No structure available for '{compound}'."
        }

    dest = f"structures/{os.path.basename(file_path)}"

    if os.path.exists(file_path) and file_path != dest:
        try:
            os.rename(file_path, dest)
        except Exception:
            shutil.copy(file_path, dest)
            os.remove(file_path)

    return {
        "image_url": f"/structures/{os.path.basename(dest)}",
        "message": f"✅ Structure of {compound.title()} generated."
    }


# ================================
# 5. PDF ANALYZE
# ================================
@app.post("/pdf-analyze")
async def pdf_analyze(
    file: UploadFile = File(...),
    language: str = Form(default="en")
):
    content = await file.read()

    # -------- PDF --------
    if file.content_type == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc = fitz.open(tmp_path)
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()

        except Exception as e:
            return {"error": f"PDF read error: {str(e)}"}

        finally:
            os.unlink(tmp_path)

        if not text.strip():
            return {"error": "Empty or scanned PDF."}

    # -------- IMAGE --------
    elif file.content_type.startswith("image/"):
        name = file.filename.replace("_", " ").replace("-", " ")
        name = name.rsplit(".", 1)[0]

        text = f"Explain chemistry topic: {name}"

    else:
        return {"error": "Unsupported file type."}

    # AI processing
    result = analyze_pdf_text(text[:2000])

    # Translation
    if language != "en":
        try:
            result["summary"] = translate_text(result["summary"], language)
            result["quiz"] = translate_text(result["quiz"], language)
            result["video_script"] = translate_text(result["video_script"], language)
        except Exception as e:
            print(f"[Translation Error] {e}")

    return result


# ================================
# 6. TRANSLATE
# ================================
@app.post("/translate")
def translate(q: Query):
    lang = getattr(q, "language", "en") or "en"

    if lang == "en":
        return {"output": q.text}

    result = translate_text(q.text, lang)
    return {"output": result}
