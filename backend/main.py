from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from model_training.train_model import FashionModel
from utils.model_predict import predict_fashion_score
from utils.preprocess import preprocess_image
from utils.score_logic import calculate_fashion_score

app = FastAPI(
    title="Fashion Analyzer API",
    description="AI-powered fashion scoring and recommendation system for men",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
fashion_model = None

@app.on_event("startup")
async def startup_event():
    """Load the pre-trained model on startup"""
    global fashion_model
    try:
        fashion_model = FashionModel()
        fashion_model.load_model("models/fashion_model.h5")
        print("✓ Fashion model loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Model not found. Train first using setup_and_train.py. Error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fashion Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze (POST)",
            "train": "/train (POST)",
            "health": "/health (GET)",
            "docs": "/docs (Interactive documentation)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if fashion_model else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }

@app.post("/analyze")
async def analyze_fashion(file: UploadFile = File(...)):
    """
    Analyze fashion in uploaded image
    Returns: score (0-10), recommendations (including skin tone advice), and color palette
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        processed_img, color_palette, skin_tone = preprocess_image(image_bytes)
        
        if processed_img is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Get ML predictions
        ml_features = fashion_model.predict(processed_img) if fashion_model else None
        
        # Calculate final score with skin tone
        score, recommendations = calculate_fashion_score(
            ml_features=ml_features,
            color_palette=color_palette,
            image=processed_img,
            skin_tone=skin_tone
        )
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "score": round(score, 1),
            "recommendations": recommendations,
            "color_palette": color_palette,
            "skin_tone": skin_tone,
            "feedback": "Fashion analysis complete!"
        })
    
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={
            "success": False,
            "error": e.detail
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"Server error: {str(e)}"
        })

@app.post("/train")
async def train_model():
    """
    Endpoint to trigger model training
    (In production, use Celery for async tasks)
    """
    try:
        model = FashionModel()
        history = model.train()
        model.save_model("models/fashion_model.h5")
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Model trained successfully",
            "epochs": len(history.history['loss']) if history else 0
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )