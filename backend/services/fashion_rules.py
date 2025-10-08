import random

def get_fallback_analysis():
    """
    Fallback rule-based color analysis when API fails
    Returns a sample analysis
    """
    
    samples = [
        {
            "shirtColor": {"name": "White", "hex": "#FFFFFF"},
            "pantsColor": {"name": "Navy Blue", "hex": "#1e3a8a"},
            "score": 95,
            "rating": "Classic Combo! 🔥",
            "explanation": "White and navy blue create a timeless, professional look. This is one of the most versatile combinations in menswear.",
            "suggestions": [
                "This combination is already excellent!",
                "Try adding brown leather accessories for warmth",
                "Consider a light blue shirt as a softer alternative"
            ]
        },
        {
            "shirtColor": {"name": "Light Blue", "hex": "#87CEEB"},
            "pantsColor": {"name": "Khaki", "hex": "#d4a574"},
            "score": 88,
            "rating": "Great Match!",
            "explanation": "Light blue and khaki create a relaxed, approachable look. The cool blue balances the warm neutral khaki perfectly.",
            "suggestions": [
                "This is a solid combination for casual or business casual",
                "White sneakers or brown loafers would complete this look",
                "Try pairing the khaki with white or mint green for variety"
            ]
        },
        {
            "shirtColor": {"name": "Black", "hex": "#000000"},
            "pantsColor": {"name": "Gray", "hex": "#808080"},
            "score": 82,
            "rating": "Good Combo",
            "explanation": "Black and gray create a sleek, modern monochromatic look. It's safe but can appear somber without the right accessories.",
            "suggestions": [
                "Add a pop of color with accessories (watch, shoes, belt)",
                "Consider lighter gray for more contrast",
                "Charcoal pants with a burgundy or navy shirt would add more personality"
            ]
        }

        # More samples can be added here    
        {
            "shirtColor": {"name": "Red", "hex": "#FF0000"},
            "pantsColor": {"name": "Black", "hex": "#000000"},
            "score": 75,
            "rating": "Needs Improvement",
            "explanation": "Red and black can be a bold combination but may come off as too intense or aggressive for everyday wear.",
            "suggestions": [
                "Try pairing red with navy or gray for a more balanced look",
                "Black pants work better with neutral or cool-toned shirts",
                "Consider a patterned shirt that incorporates red for a subtler effect"
            ]
        }
    ]
    
    return random.choice(samples)


# Color theory rules (next time , willbe edited more well)
COLOR_COMPATIBILITY = {
    "complementary": [
        ("blue", "orange"),
        ("red", "green"),
        ("yellow", "purple")
    ],
    "analogous": [
        ("blue", "green"),
        ("red", "orange"),
        ("yellow", "green")
    ],
    "neutral_safe": [
        "white", "black", "gray", "beige", "khaki", "navy"
    ],
    "avoid": [
        ("brown", "black"), 
        ("navy", "black"),  
    ]
}
