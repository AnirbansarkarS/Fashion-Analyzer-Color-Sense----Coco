from google.cloud import vision
import io

def extract_colors(image_bytes: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    response = client.image_properties(image=image)
    colors = response.image_properties_annotation.dominant_colors.colors

    # Extract top 2 colors (shirt-pants)
    detected_colors = []
    for color_info in colors[:2]:
        rgb = color_info.color
        detected_colors.append(rgb_to_name(rgb.red, rgb.green, rgb.blue))

    return detected_colors

def rgb_to_name(r, g, b):
    
    if r > 200 and g < 100 and b < 100:
        return "red"
    elif g > 200 and r < 100:
        return "green"
    elif b > 200 and r < 100:
        return "blue"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    elif r > 200 and b > 200 and g < 100:
        return "purple"
    elif r > 200 and g < 100 and b > 200:
        return "pink"
    elif r > 150 and g > 100 and b < 100:
        return "orange"
    elif r > 100 and g > 150 and b < 100:
        return "lime"
    else:
        return f"rgb({int(r)}, {int(g)}, {int(b)})"
