from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys
import os
from pathlib import Path

# Add backend to path - CRITICAL FIX
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

try:
    from model_training.train_model import FashionModel
    from utils.model_predict import predict_fashion_score
    from utils.preprocess import preprocess_image
    from utils.score_logic import calculate_fashion_score
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Backend path: {backend_path}")
    raise

app = FastAPI(
    title="Fashion Analyzer API",
    description="AI-powered fashion scoring and recommendation system for men",
    version="1.0.0"
)

# CORS middleware - MORE PERMISSIVE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
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
        print("📦 Loading model...")
        fashion_model = FashionModel()
        model_path = str(backend_path / "models" / "fashion_model.h5")
        
        if os.path.exists(model_path):
            fashion_model.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}")
            print("⚠ Model will use default/synthetic predictions")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fashion Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "model_status": "loaded" if fashion_model and fashion_model.model else "not loaded",
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
    model_status = "loaded" if (fashion_model and fashion_model.model) else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "message": "API is running"
    }

@app.post("/analyze")
async def analyze_fashion(file: UploadFile = File(...)):
    """
    Analyze fashion in uploaded image
    Returns: score (0-10), recommendations, color palette, and skin tone
    """
    try:
        print(f"📸 Analyzing image: {file.filename}")
        
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        print(f"📦 Image size: {len(image_bytes)} bytes")
        
        processed_img, color_palette, skin_tone = preprocess_image(image_bytes)
        
        if processed_img is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        print("✓ Image preprocessed")
        
        # Get ML predictions
        ml_features = None
        if fashion_model and fashion_model.model:
            try:
                ml_features = fashion_model.predict(processed_img)
                print("✓ ML prediction done")
            except Exception as e:
                print(f"⚠ ML prediction error: {e}")
        
        # Calculate final score with skin tone
        score, recommendations = calculate_fashion_score(
            ml_features=ml_features,
            color_palette=color_palette,
            image=processed_img,
            skin_tone=skin_tone
        )
        
        print(f"✓ Score calculated: {score:.1f}/10")
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "score": round(score, 1),
            "recommendations": recommendations,
            "color_palette": color_palette,
            "skin_tone": skin_tone,
            "feedback": "Fashion analysis complete!"
        })
    
    except HTTPException as e:
        print(f"✗ HTTP Error: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={
            "success": False,
            "error": e.detail
        })
    except Exception as e:
        print(f"✗ Server error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"Server error: {str(e)}"
        })

@app.post("/train")
async def train_model():
    """Endpoint to trigger model training"""
    try:
        print("🚀 Training model...")
        model = FashionModel()
        history = model.train()
        model.save_model("models/fashion_model.h5")
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Model trained successfully",
            "epochs": len(history.history['loss']) if history else 0
        })
    except Exception as e:
        print(f"✗ Training error: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "test": "success",
        "backend_path": str(backend_path),
        "models_exist": os.path.exists(str(backend_path / "models" / "fashion_model.h5"))
    }

if __name__ == "__main__":
    print("="*60)
    print("🎨 FASHION ANALYZER API")
    print("="*60)
    print(f"Backend path: {backend_path}")
    print(f"Starting server at http://0.0.0.0:8000")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Changed to False for stability
        log_level="info"
    )