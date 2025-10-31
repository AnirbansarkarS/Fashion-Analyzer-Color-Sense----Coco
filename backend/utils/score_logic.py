import numpy as np
from utils.preprocess import check_image_quality

def calculate_fashion_score(ml_features=None, color_palette=None, image=None, skin_tone=None):
    """Calculate final fashion score combining ML predictions, heuristics, and skin tone analysis"""
    
    scores = {}
    
    # 1️⃣ ML-based features (35%)
    if ml_features is not None:
        scores['style_match'] = round((ml_features.get('style_match', 0.55) * 10) + np.random.uniform(-1, 1), 1)
        scores['texture_quality'] = round((ml_features.get('texture_quality', 0.55) * 10) + np.random.uniform(-1, 1), 1)
    else:
        scores['style_match'] = np.random.uniform(4.5, 6.5)
        scores['texture_quality'] = np.random.uniform(4.5, 6.5)
    
    # 2️⃣ Skin tone compatibility (25%)
    if skin_tone:
        scores['skin_compatibility'] = calculate_skin_tone_compatibility(skin_tone, color_palette)
    else:
        scores['skin_compatibility'] = np.random.uniform(4.5, 6.5)
    
    # 3️⃣ Color harmony (20%)
    scores['color_harmony'] = calculate_color_harmony(color_palette) if color_palette else np.random.uniform(4, 6.5)
    
    # 4️⃣ Image quality (10%)
    if image is not None:
        quality_raw = check_image_quality(image)
        scores['image_quality'] = np.clip((quality_raw / 100) * 10 + np.random.uniform(-0.5, 0.5), 3.5, 9.5)
    else:
        scores['image_quality'] = np.random.uniform(4, 6.5)
    
    # 5️⃣ Grooming / Style heuristics (10%)
    scores['grooming_heuristics'] = calculate_mens_heuristic_score(color_palette, skin_tone) if color_palette else np.random.uniform(4.5, 6.5)
    
    # Weighted final score
    final_score = (
        0.35 * scores['style_match'] +
        0.25 * scores['skin_compatibility'] +
        0.20 * scores['color_harmony'] +
        0.10 * scores['image_quality'] +
        0.10 * scores['grooming_heuristics']
    )
    final_score = np.clip(round(final_score, 1), 2.5, 9.8)
    
    recommendations = generate_mens_recommendations(scores, color_palette, skin_tone)
    return final_score, recommendations


# ✅ Skin tone compatibility
def calculate_skin_tone_compatibility(skin_tone, color_palette):
    if not color_palette or not skin_tone:
        return 5.0
    try:
        skin_rgb = skin_tone['rgb']
        outfit_colors = [c['rgb'] for c in color_palette]
        comp_scores = [
            analyze_color_contrast_with_skin(skin_rgb, outfit_rgb) for outfit_rgb in outfit_colors
        ]
        avg_comp = np.mean(comp_scores)
        score = 9.5 * avg_comp + np.random.uniform(-0.8, 0.8)
        return np.clip(score, 3.0, 9.5)
    except Exception as e:
        print(f"Skin tone compat error: {e}")
        return 5.0


# ✅ Contrast logic
def analyze_color_contrast_with_skin(skin_rgb, outfit_rgb):
    r1, g1, b1 = skin_rgb
    r2, g2, b2 = outfit_rgb
    contrast = np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2) / 440
    if contrast < 0.25:
        return 0.4
    elif contrast < 0.35:
        return 0.55
    elif contrast < 0.5:
        return 0.75
    elif contrast < 0.7:
        return 0.9
    else:
        return 0.8


# ✅ Color harmony
def calculate_color_harmony(color_palette):
    if not color_palette or len(color_palette) < 2:
        return 5.0
    try:
        colors = [tuple(c['rgb']) for c in color_palette]
        contrasts = [calculate_contrast(colors[i], colors[j])
                     for i in range(len(colors))
                     for j in range(i + 1, len(colors))]
        avg_contrast = np.mean(contrasts)
        harmony = 9.2 - abs(55 - avg_contrast) / 7
        return np.clip(harmony + np.random.uniform(-0.5, 0.5), 3.5, 9.5)
    except Exception as e:
        print(f"Color harmony error: {e}")
        return 5.0


def calculate_contrast(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2) / 440 * 100


# ✅ Heuristics (Fashion sense)
def calculate_mens_heuristic_score(color_palette, skin_tone):
    if not color_palette:
        return 5.0
    
    score = 5.0
    neutral_colors = ['Black', 'White', 'Gray', 'Dark Blue', 'Dark Red']
    color_names = [c['name'] for c in color_palette]

    if any(c in neutral_colors for c in color_names):
        score += 1.5
    if 2 <= len(color_palette) <= 4:
        score += 1.0
    elif len(color_palette) > 5:
        score -= 1.0
    if is_classic_mens_combination(color_names):
        score += 1.0
    if skin_tone:
        if is_warm_skin_tone(skin_tone):
            if any(c['name'] in ['Orange', 'Red', 'Dark Red'] for c in color_palette):
                score += 0.5
        else:
            if any(c['name'] in ['Blue', 'Dark Blue', 'Purple'] for c in color_palette):
                score += 0.5
    return np.clip(score + np.random.uniform(-0.3, 0.3), 3.5, 9.5)


def is_warm_skin_tone(skin_tone):
    r, g, b = skin_tone['rgb']
    return (r - b) > 30


def is_classic_mens_combination(color_names):
    classic_combos = [
        ('Black', 'White'),
        ('Navy', 'White'),
        ('Gray', 'White'),
        ('Black', 'Gray'),
        ('Dark Blue', 'White'),
        ('Dark Red', 'White'),
        ('Black', 'Blue'),
    ]
    return any(all(c in color_names for c in combo) for combo in classic_combos)


# ✅ Recommendations
def generate_mens_recommendations(scores, color_palette, skin_tone):
    recommendations = []
    if scores.get('skin_compatibility', 5) < 6:
        tone_type = "warm" if (skin_tone and is_warm_skin_tone(skin_tone)) else "cool"
        if tone_type == "warm":
            recommendations.append({
                "category": "Skin Tone Match",
                "tip": "You have a warm skin tone. Use earthy tones like olive, tan, rust, or burgundy.",
                "priority": "high"
            })
        else:
            recommendations.append({
                "category": "Skin Tone Match",
                "tip": "You have a cool skin tone. Stick to navy, gray, white, or icy blue shades.",
                "priority": "high"
            })
    if scores.get('color_harmony', 5) < 6:
        recommendations.append({
            "category": "Color Coordination",
            "tip": "Follow 60-30-10 rule: 60% base, 30% secondary, 10% accent. Creates sharp coordination.",
            "priority": "high"
        })
    if scores.get('grooming_heuristics', 5) < 6:
        recommendations.append({
            "category": "Grooming / Outfit Complexity",
            "tip": "Simplify outfit to 2–4 colors. Add one neutral like gray or navy for balance.",
            "priority": "medium"
        })
    if scores.get('style_match', 5) < 5:
        recommendations.append({
            "category": "Style Balance",
            "tip": "Ensure contrast between top and bottom. Light over dark works best for men.",
            "priority": "high"
        })
    if scores.get('texture_quality', 5) < 6:
        recommendations.append({
            "category": "Texture Variety",
            "tip": "Mix textures like denim, cotton, or wool to add dimension to your outfit.",
            "priority": "medium"
        })
    if not recommendations:
        recommendations.append({
            "category": "Overall",
            "tip": "Your outfit looks great! Balanced color and tone — stylish and confident. ✅",
            "priority": "low"
        })
    return recommendations


def get_score_feedback(score):
    if score >= 9:
        return "🔥 Elite styling! Effortlessly balanced and fashion-forward."
    elif score >= 8:
        return "👏 Excellent coordination! You’ve got a great sense of proportion and tone."
    elif score >= 7:
        return "👍 Very good! Solid outfit with minor tweaks possible."
    elif score >= 6:
        return "🙂 Decent look! With a few color or fabric adjustments, you’ll stand out more."
    elif score >= 5:
        return "🧩 Average match. Improve color pairing and outfit balance."
    else:
        return "🚧 Needs refinement — focus on neutrals, fit, and better lighting."
