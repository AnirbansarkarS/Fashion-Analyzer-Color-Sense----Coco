from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from services.gemini_service import analyze_outfit_colors
from services.fashion_rules import get_fallback_analysis
import os

app = FastAPI(title="Fashion Analyzer API")

# editable for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Fashion Analyzer API is running! 🎨"}

@app.post("/analyze")
async def analyze_fashion(file: UploadFile = File(...)):
    """
    Analyze outfit colors from uploaded image
    """
    try:
        
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        
        try:
            result = await analyze_outfit_colors(base64_image)
            return JSONResponse(content=result)
        except Exception as gemini_error:
            print(f"Gemini API failed: {gemini_error}")
            # Fallback to rule-based system
            fallback_result = get_fallback_analysis()
            return JSONResponse(content=fallback_result)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)