from pydantic import BaseModel
from typing import List

class ColorInfo(BaseModel):
    name: str
    hex: str

class FashionAnalysisResponse(BaseModel):
    shirtColor: ColorInfo
    pantsColor: ColorInfo
    score: int
    rating: str
    explanation: str
    suggestions: List[str]