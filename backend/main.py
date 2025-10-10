from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn
import os

from model_training.dataset_preprocess import preprocess_image
from model_training.train_model import FashionModel


app = FastAPI(
    title="Fashion AI API",
    description="AI-powered fashion style scoring and color analysis",
    version="1.0.0"
)

# Enable CORS for frontend (React, etc.)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this later to your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

MODEL_PATH = "models/fashion_model_v1.h5"
model = None


@app.on_event("startup")
def load_model():
    global model
    model = FashionModel()
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading existing model from {MODEL_PATH}")
        model.load_model(MODEL_PATH)
    else:
        print(f"⚠️ Model not found. Building new untrained model...")
        model.build_model()


@app.get("/", tags=["Health"])
def health_check():
    return {"message": "✅ Fashion AI backend is running!"}


@app.post("/predict/", tags=["Fashion Analysis"])
async def predict_fashion_score(file: UploadFile = File(...)):
    if not model or not model.model:
        raise HTTPException(status_code=503, detail="Model not loaded properly.")

    try:
        # Read uploaded file
        image_bytes = await file.read()

        # Preprocess (runs in threadpool to avoid blocking)
        processed_img, color_palette = await run_in_threadpool(preprocess_image, image_bytes)
        if processed_img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupt image.")

        # Model Prediction
        prediction_result = await run_in_threadpool(model.predict, processed_img)
        if prediction_result is None:
            raise HTTPException(status_code=500, detail="Model prediction failed.")

        final_score = prediction_result.get("style_match", 0) * 10

        return {
            "filename": file.filename,
            "style_score": round(final_score, 2),
            "attributes": prediction_result,
            "color_palette": color_palette,
        }

    except Exception as e:
        print(f"🔥 Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
