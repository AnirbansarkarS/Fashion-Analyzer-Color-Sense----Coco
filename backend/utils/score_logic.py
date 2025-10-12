import numpy as np
from utils.preprocess import check_image_quality

def calculate_fashion_score(ml_features=None, color_palette=None, image=None, skin_tone=None):
    """Calculate final fashion score combining ML predictions, heuristics, and skin tone analysis"""
    
    scores = {}
    
    # 1. ML-based features (35% weight)
    if ml_features is not None:
        scores['style_match'] = ml_features.get('style_match', 0.5) * 10
        scores['texture_quality'] = ml_features.get('texture_quality', 0.5) * 10
    else:
        scores['style_match'] = 5.0
        scores['texture_quality'] = 5.0
    
    # 2. Skin tone compatibility (25% weight)
    if skin_tone:
        skin_score = calculate_skin_tone_compatibility(skin_tone, color_palette)
        scores['skin_compatibility'] = skin_score
    else:
        scores['skin_compatibility'] = 5.0
    
    # 3. Color harmony (20% weight)
    color_score = calculate_color_harmony(color_palette) if color_palette else 5.0
    scores['color_harmony'] = color_score
    
    # 4. Image quality (10% weight)
    image_quality = (check_image_quality(image) / 100) * 10 if image is not None else 5.0
    scores['image_quality'] = image_quality
    
    # 5. Grooming & Style Heuristics (10% weight)
    heuristic_score = calculate_mens_heuristic_score(color_palette, skin_tone) if color_palette else 5.0
    scores['grooming_heuristics'] = heuristic_score
    
    # Weighted average
    final_score = (
        0.35 * scores['style_match'] +
        0.25 * scores['skin_compatibility'] +
        0.20 * scores['color_harmony'] +
        0.10 * scores['image_quality'] +
        0.10 * scores['grooming_heuristics']
    )
    
    # Generate recommendations
    recommendations = generate_mens_recommendations(scores, color_palette, skin_tone)
    
    return final_score, recommendations

def calculate_skin_tone_compatibility(skin_tone, color_palette):
    """Analyze color compatibility with skin tone. Returns score 1-10"""
    if not color_palette or not skin_tone:
        return 5.0
    
    try:
        skin_rgb = skin_tone['rgb']
        outfit_colors = [c['rgb'] for c in color_palette]
        
        compatibility_scores = []
        
        for outfit_color in outfit_colors:
            compatibility = analyze_color_contrast_with_skin(skin_rgb, outfit_color)
            compatibility_scores.append(compatibility)
        
        avg_compatibility = np.mean(compatibility_scores)
        
        if avg_compatibility > 0.7:
            score = 9.0
        elif avg_compatibility > 0.6:
            score = 8.0
        elif avg_compatibility > 0.5:
            score = 7.0
        elif avg_compatibility > 0.4:
            score = 5.5
        else:
            score = 4.0
        
        return min(10, score)
    
    except Exception as e:
        print(f"Error in calculate_skin_tone_compatibility: {e}")
        return 5.0

def analyze_color_contrast_with_skin(skin_rgb, outfit_rgb):
    """Analyze if outfit color creates good contrast with skin tone. Returns compatibility score 0-1"""
    r_skin, g_skin, b_skin = skin_rgb
    r_out, g_out, b_out = outfit_rgb
    
    contrast = np.sqrt((r_out - r_skin)**2 + (g_out - g_skin)**2 + (b_out - b_skin)**2)
    contrast_normalized = contrast / 440
    
    if 0.4 <= contrast_normalized <= 0.7:
        compatibility = 1.0
    elif 0.3 <= contrast_normalized <= 0.8:
        compatibility = 0.85
    elif 0.2 <= contrast_normalized <= 0.85:
        compatibility = 0.65
    else:
        compatibility = 0.4
    
    return compatibility

def calculate_color_harmony(color_palette):
    """Score color harmony (1-10) based on color theory"""
    if not color_palette or len(color_palette) < 2:
        return 5.0
    
    try:
        colors = [tuple(c['rgb']) for c in color_palette]
        
        contrasts = []
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                contrast = calculate_contrast(colors[i], colors[j])
                contrasts.append(contrast)
        
        avg_contrast = np.mean(contrasts) if contrasts else 0
        
        if 30 <= avg_contrast <= 70:
            harmony_score = 8.5
        elif 20 <= avg_contrast <= 80:
            harmony_score = 7.0
        elif 10 <= avg_contrast <= 90:
            harmony_score = 5.5
        else:
            harmony_score = 3.0
        
        return min(10, harmony_score)
    
    except Exception as e:
        print(f"Error in calculate_color_harmony: {e}")
        return 5.0

def calculate_contrast(color1, color2):
    """Calculate contrast between two RGB colors"""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    contrast = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    return contrast / 440 * 100

def calculate_mens_heuristic_score(color_palette, skin_tone):
    """Calculate score based on men's fashion heuristics"""
    if not color_palette:
        return 5.0
    
    score = 5.5
    
    neutral_colors = ['Black', 'White', 'Gray', 'Dark Blue', 'Dark Red']
    has_neutral = any(c['name'] in neutral_colors for c in color_palette)
    if has_neutral:
        score += 2.0
    
    if 2 <= len(color_palette) <= 4:
        score += 1.5
    elif len(color_palette) == 1:
        score += 0.5
    
    if len(color_palette) > 5:
        score -= 1.0
    
    color_names = [c['name'] for c in color_palette]
    if is_classic_mens_combination(color_names):
        score += 1.0
    
    if skin_tone:
        if is_warm_skin_tone(skin_tone):
            warm_colors = ['Orange', 'Red', 'Dark Red']
            if any(c['name'] in warm_colors for c in color_palette):
                score += 0.5
        else:
            cool_colors = ['Blue', 'Dark Blue', 'Purple']
            if any(c['name'] in cool_colors for c in color_palette):
                score += 0.5
    
    return min(10, score)

def is_warm_skin_tone(skin_tone):
    """Determine if skin tone is warm or cool"""
    r, g, b = skin_tone['rgb']
    if r - b > 30:
        return True
    return False

def is_classic_mens_combination(color_names):
    """Check if colors follow classic men's fashion combinations"""
    classic_combos = [
        ('Black', 'White'),
        ('Navy', 'White'),
        ('Gray', 'White'),
        ('Black', 'Gray'),
        ('Dark Blue', 'White'),
        ('Dark Red', 'White'),
        ('Black', 'Blue'),
    ]
    
    for combo in classic_combos:
        if all(color in color_names for color in combo):
            return True
    
    return False

def generate_mens_recommendations(scores, color_palette, skin_tone):
    """Generate personalized men's fashion recommendations"""
    recommendations = []
    
    # Skin tone compatibility
    if scores.get('skin_compatibility', 5) < 6:
        if skin_tone:
            tone_type = "warm" if is_warm_skin_tone(skin_tone) else "cool"
            if tone_type == "warm":
                recommendations.append({
                    "category": "Skin Tone Match",
                    "tip": "You have a warm skin tone. Wear warm colors like terracotta, olive, warm browns, and burnt orange. Avoid cool grays and pure whites.",
                    "priority": "high"
                })
            else:
                recommendations.append({
                    "category": "Skin Tone Match",
                    "tip": "You have a cool skin tone. Stick with cool colors like navy, black, cool grays, and jewel tones. Avoid warm oranges and yellows.",
                    "priority": "high"
                })
    
    # Color harmony
    if scores.get('color_harmony', 5) < 6:
        recommendations.append({
            "category": "Color Coordination",
            "tip": "Follow the 60-30-10 rule: 60% neutral base (navy/black/gray), 30% secondary color, 10% accent. This creates a polished look.",
            "priority": "high"
        })
    
    # Grooming heuristics
    if scores.get('grooming_heuristics', 5) < 6:
        if color_palette and len(color_palette) > 5:
            recommendations.append({
                "category": "Color Complexity",
                "tip": "You're wearing too many colors. Men look best in 2-4 colors. Try simplifying your outfit.",
                "priority": "high"
            })
        
        if color_palette:
            if not any(c['name'] in ['Black', 'White', 'Gray', 'Dark Blue'] for c in color_palette):
                recommendations.append({
                    "category": "Neutral Base",
                    "tip": "Add a neutral base like black, navy, or gray. This anchors your outfit and makes you look more put-together.",
                    "priority": "medium"
                })
    
    # Style match
    if scores.get('style_match', 5) < 5:
        recommendations.append({
            "category": "Overall Fit & Style",
            "tip": "Ensure top and bottom colors create visual balance. Darker on bottom, lighter on top typically looks more refined.",
            "priority": "high"
        })
    
    # Texture quality
    if scores.get('texture_quality', 5) < 6:
        recommendations.append({
            "category": "Texture & Fabric",
            "tip": "Mix different textures (cotton, denim, wool) to add depth. Avoid wearing all the same fabric type.",
            "priority": "medium"
        })
    
    # Specific color suggestions
    if color_palette and len(color_palette) > 0:
        dominant_color = color_palette[0]['name']
        recommendations.append({
            "category": "Styling Tip",
            "tip": f"Your dominant color is {dominant_color}. Pair it with a neutral like navy, gray, or black for a professional appearance.",
            "priority": "low"
        })
    
    # Image quality
    if scores.get('image_quality', 5) < 5:
        recommendations.append({
            "category": "Photo Quality",
            "tip": "Take photo in natural daylight or well-lit indoor setting. Ensure the outfit is fully visible and you're centered in frame for accurate analysis.",
            "priority": "low"
        })
    
    # Grooming tips
    recommendations.append({
        "category": "Grooming Tips",
        "tip": "Men's fashion is about simplicity and quality. Focus on: clean, well-fitted clothes; groomed hair; polished shoes; and accessories in moderation.",
        "priority": "medium"
    })
    
    if not recommendations:
        recommendations.append({
            "category": "Overall",
            "tip": "Your outfit looks great! Excellent color coordination and style match. Keep it up!",
            "priority": "low"
        })
    
    return recommendations

def get_score_feedback(score):
    """Get verbal feedback based on score"""
    if score >= 9:
        return "Outstanding! You look incredibly well-put-together. 🌟 Professional styling with perfect color coordination."
    elif score >= 8:
        return "Excellent! Great fashion sense with excellent coordination. 👏 You clearly know what works for you."
    elif score >= 7:
        return "Very Good! Nice color balance and style match. 👍 A few tweaks could make you look even sharper."
    elif score >= 6:
        return "Good! Your outfit works well overall. Some improvements could take your style to the next level."
    elif score >= 5:
        return "Fair! Consider the recommendations to enhance your look and polish your appearance."
    elif score >= 4:
        return "Needs work. Pay attention to color harmony, fit, and grooming for a better overall impression."
    else:
        return "Let's start fresh. Review the recommendations carefully and focus on basics like fit and neutral colors."