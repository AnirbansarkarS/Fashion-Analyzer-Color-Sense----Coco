from PIL import Image
import numpy as np
import io
from sklearn.cluster import KMeans

def preprocess_image(image_bytes):
    try:
        # Convert bytes → PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to model input size (224x224)
        image = image.resize((224, 224))

        # Convert to numpy array and normalize (0–1)
        img_array = np.array(image) / 255.0

        # Extract color palette (optional)
        color_palette = extract_color_palette(img_array)

        return img_array.reshape(1, 224, 224, 3), color_palette
    except Exception as e:
        print(f"🔥 Error in preprocess_image: {e}")
        return None, None


def extract_color_palette(img_array, n_colors=5):
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10).fit(pixels)
    palette = (kmeans.cluster_centers_ * 255).astype(int).tolist()
    return palette
import cv2
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image: convert, resize, detect person, extract colors and skin tone
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None, None
        
        # Detect person in image (using Haar cascade)
        cropped_img = detect_and_crop_person(img)
        if cropped_img is None:
            cropped_img = img  # Fallback to full image
        
        # Resize to model input size
        resized_img = cv2.resize(cropped_img, target_size)
        
        # Normalize (0-1 range)
        normalized_img = resized_img.astype('float32') / 255.0
        
        # Extract color palette
        color_palette = extract_color_palette(resized_img, n_colors=5)
        
        # Extract skin tone from face/neck region
        skin_tone = extract_skin_tone(resized_img)
        
        return normalized_img, color_palette, skin_tone
    
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None, None, None

def detect_and_crop_person(img, scale=1.1, min_neighbors=5):
    """
    Detect human figure and crop to bounding box
    """
    try:
        # Load Haar cascade for full body
        cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bodies = cascade.detectMultiScale(
            gray, 
            scaleFactor=scale, 
            minNeighbors=min_neighbors
        )
        
        if len(bodies) > 0:
            # Get largest bounding box
            (x, y, w, h) = max(bodies, key=lambda b: b[2] * b[3])
            
            # Add padding (10% margin)
            padding = 0.1
            x = max(0, int(x - w * padding))
            y = max(0, int(y - h * padding))
            w = int(w * (1 + 2 * padding))
            h = int(h * (1 + 2 * padding))
            
            cropped = img[y:y+h, x:x+w]
            return cropped
        
        return img
    
    except Exception as e:
        print(f"Error in detect_and_crop_person: {e}")
        return img

def extract_color_palette(img, n_colors=5):
    """
    Extract dominant colors using KMeans clustering
    Returns color palette with RGB values and names
    """
    try:
        # Reshape image to list of pixels
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        colors = np.uint8(kmeans.cluster_centers_)
        
        # Convert BGR to RGB and create palette
        palette = []
        for color in colors:
            bgr = color
            rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]  # BGR to RGB
            hex_color = rgb_to_hex(rgb)
            color_name = get_color_name(rgb)
            
            palette.append({
                "rgb": rgb,
                "hex": hex_color,
                "name": color_name
            })
        
        return palette
    
    except Exception as e:
        print(f"Error in extract_color_palette: {e}")
        return None

def rgb_to_hex(rgb):
    """Convert RGB to hexadecimal"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def get_color_name(rgb):
    """Get color name from RGB values"""
    r, g, b = rgb
    
    # Simple color naming logic
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > g and r > b:
        if r > 150:
            return "Red"
        else:
            return "Dark Red"
    elif g > r and g > b:
        if g > 150:
            return "Green"
        else:
            return "Dark Green"
    elif b > r and b > g:
        if b > 150:
            return "Blue"
        else:
            return "Dark Blue"
    elif r > 150 and g > 100 and b < 100:
        return "Orange"
    elif r > 100 and g < 100 and b > 100:
        return "Purple"
    else:
        return "Gray"

def check_image_quality(img):
    """
    Check if image is clear and well-lit
    Returns quality score (0-100)
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(100, int(laplacian_var / 2))
        
        # Mean brightness
        brightness = int(np.mean(gray))
        brightness_score = 100 if 50 < brightness < 200 else 50
        
        # Contrast
        contrast = int(np.std(gray))
        contrast_score = min(100, int(contrast / 1.5))
        
        quality_score = int(0.4 * sharpness + 0.3 * brightness_score + 0.3 * contrast_score)
        
        return quality_score
    
    except Exception as e:
        print(f"Error in check_image_quality: {e}")
        return 50

def extract_skin_tone(img):
    """
    Extract dominant skin tone from face/upper body region
    Returns skin tone color info with RGB, hex, and name
    """
    try:
        # Convert BGR to HSV for better skin detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Skin tone range in HSV (approximate)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Get skin region from original image
        skin_region = cv2.bitwise_and(img, img, mask=mask)
        
        # Extract skin pixels
        skin_pixels = skin_region[mask > 0]
        
        if len(skin_pixels) < 100:
            # Fallback: sample from upper-middle region (face area)
            h, w = img.shape[:2]
            face_region = img[int(h*0.15):int(h*0.4), int(w*0.3):int(w*0.7)]
            skin_pixels = face_region.reshape((-1, 3))
        
        # KMeans to find dominant skin tone
        pixels = np.float32(skin_pixels)
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        skin_color_bgr = np.uint8(kmeans.cluster_centers_[0])
        
        # Convert BGR to RGB
        skin_color_rgb = [int(skin_color_bgr[2]), int(skin_color_bgr[1]), int(skin_color_bgr[0])]
        
        hex_color = rgb_to_hex(skin_color_rgb)
        color_name = get_skin_tone_name(skin_color_rgb)
        
        return {
            "rgb": skin_color_rgb,
            "hex": hex_color,
            "name": color_name,
            "tone_type": "warm" if skin_color_rgb[0] - skin_color_rgb[2] > 30 else "cool"
        }
    
    except Exception as e:
        print(f"Error in extract_skin_tone: {e}")
        return None

def get_skin_tone_name(rgb):
    """
    Get descriptive skin tone name from RGB values
    """
    r, g, b = rgb
    
    # Calculate lightness and saturation
    lightness = (max(r, g, b) + min(r, g, b)) / 2
    saturation = (max(r, g, b) - min(r, g, b)) / (max(r, g, b) + min(r, g, b)) if max(r, g, b) + min(r, g, b) > 0 else 0
    
    # Determine tone
    if lightness < 80:
        if r - b > 20:
            return "Deep Warm"
        else:
            return "Deep Cool"
    elif lightness < 130:
        if r - b > 20:
            return "Medium Warm"
        else:
            return "Medium Cool"
    elif lightness < 180:
        if r - b > 20:
            return "Light Warm"
        else:
            return "Light Cool"
    else:
        return "Fair"
