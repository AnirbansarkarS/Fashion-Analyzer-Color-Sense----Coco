import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

async def analyze_outfit_colors(base64_image: str):
    """
    Use Gemini Vision API to analyze outfit colors and give fashion advice
    """
    
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    
    prompt = """Analyze this outfit image and identify the clothing items and their colors.

Focus on identifying:
1. Shirt/Top color (name and approximate hex code)
2. Pants/Bottom color (name and approximate hex code)

Then evaluate the color combination:
- Give a score from 0-100 (how well the colors match)
- Provide a rating (e.g., "Perfect Match!", "Good Combo", "Needs Improvement")
- Explain why this combination works or doesn't work
- Give 2-3 specific suggestions for better color combinations

Return your response in this EXACT JSON format:
{
    "shirtColor": {
        "name": "color name",
        "hex": "#hexcode"
    },
    "pantsColor": {
        "name": "color name",
        "hex": "#hexcode"
    },
    "score": 85,
    "rating": "Great Match!",
    "explanation": "Brief explanation of why this works or doesn't",
    "suggestions": [
        "Suggestion 1",
        "Suggestion 2",
        "Suggestion 3"
    ]
}

Be specific with color names (e.g., "Navy Blue" not just "Blue", "Olive Green" not just "Green").
Make hex codes realistic approximations."""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 1024,
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    
    try:
        text_response = result['candidates'][0]['content']['parts'][0]['text']
        

        if "```json" in text_response:
            json_str = text_response.split("```json")[1].split("```")[0].strip()
        elif "```" in text_response:
            json_str = text_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = text_response.strip()
        
        fashion_data = json.loads(json_str)
        return fashion_data
        
    except Exception as e:
        raise Exception(f"Failed to parse Gemini response: {str(e)}")
