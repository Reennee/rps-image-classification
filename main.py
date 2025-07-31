from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from src.prediction import predict_image, retrain_model
from src.preprocessing import load_preprocess_data
from src.model import create_model, save_model
from keras.models import load_model

app = FastAPI(
    title="Rock-Paper-Scissors Classifier API",
    description="API for image prediction and model retraining",
    version="1.0.0"
)

# Allow CORS for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Directory for retraining uploads
RETRAIN_DIR = "data/retrain_uploads"
MODEL_PATH = "models/best_model.h5"

os.makedirs(RETRAIN_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Accept image file, save temporarily
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format")
    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        label, confidence = predict_image(tmp_path, MODEL_PATH)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return JSONResponse({
        "predicted_class": label,
        "confidence": float(confidence)
    })

@app.post("/retrain")
async def retrain(
    file: UploadFile = File(...),
    label: str = Form(...)
):
    # Create directory if not exists
    label_dir = os.path.join("data/custom", label)
    os.makedirs(label_dir, exist_ok=True)

    file_path = os.path.join(label_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # retrain model
    new_model = retrain_model()
    save_model(new_model, MODEL_PATH)

    return {"message": f"Model retrained with image: {file.filename}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
