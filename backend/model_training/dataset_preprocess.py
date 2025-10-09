import cv2
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans

def preprocess_image(image_bytes, target_size=(224, 224)):

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None
        
      
        cropped_img = detect_and_crop_person(img)
        if cropped_img is None:
            cropped_img = img  
        
        resized_img = cv2.resize(cropped_img, target_size)
        # Normalize 
        normalized_img = resized_img.astype('float32') / 255.0
        
        color_palette = extract_color_palette(resized_img, n_colors=5)
        
        return normalized_img, color_palette
    
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None, None

def detect_and_crop_person(img, scale=1.1, min_neighbors=5):
##### commented ... will be testted......######

def extract_color_palette(img, n_colors=5):
    """
    Extract dominant colors using KMeans clustering
    Returns color palette with RGB values and names
    """


def rgb_to_hex(rgb):
    # RGB to hexadecimal
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def get_color_name(rgb):
    """Get color name from RGB values"""
    r, g, b = rgb
    
    # Will be edied ... ( trying to keep simple)
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


















"1"
    # try:
    #     # Load Haar cascade for full body
    #     cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    #     cascade = cv2.CascadeClassifier(cascade_path)
        
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     bodies = cascade.detectMultiScale(
    #         gray, 
    #         scaleFactor=scale, 
    #         minNeighbors=min_neighbors
    #     )
        
    #     if len(bodies) > 0:
    #         # Get largest bounding box
    #         (x, y, w, h) = max(bodies, key=lambda b: b[2] * b[3])
            
    #         # Add padding (10% margin)
    #         padding = 0.1
    #         x = max(0, int(x - w * padding))
    #         y = max(0, int(y - h * padding))
    #         w = int(w * (1 + 2 * padding))
    #         h = int(h * (1 + 2 * padding))
            
    #         cropped = img[y:y+h, x:x+w]
    #         return cropped
        
    #     return img
    
    # except Exception as e:
    #     print(f"Error in detect_and_crop_person: {e}")
    #     return img







"2"
    #     try:
    #     # Reshape image to list of pixels
    #     pixels = img.reshape((-1, 3))
    #     pixels = np.float32(pixels)
        
    #     # KMeans clustering
    #     kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    #     kmeans.fit(pixels)
        
    #     colors = np.uint8(kmeans.cluster_centers_)
        
    #     # Convert BGR to RGB and create palette
    #     palette = []
    #     for color in colors:
    #         bgr = color
    #         rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]  # BGR to RGB
    #         hex_color = rgb_to_hex(rgb)
    #         color_name = get_color_name(rgb)
            
    #         palette.append({
    #             "rgb": rgb,
    #             "hex": hex_color,
    #             "name": color_name
    #         })
        
    #     return palette
    
    # except Exception as e:
    #     print(f"Error in extract_color_palette: {e}")
    #     return None






# def check_image_quality(img):
#     """
#     Check if image is clear and well-lit
#     Returns quality score (0-100)
#     """
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Laplacian variance (sharpness)
#         laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         sharpness = min(100, int(laplacian_var / 2))
        
#         # Mean brightness
#         brightness = int(np.mean(gray))
#         brightness_score = 100 if 50 < brightness < 200 else 50
        
#         # Contrast
#         contrast = int(np.std(gray))
#         contrast_score = min(100, int(contrast / 1.5))
        
#         quality_score = int(0.4 * sharpness + 0.3 * brightness_score + 0.3 * contrast_score)
        
#         return quality_score
    
#     except Exception as e:
#         print(f"Error in check_image_quality: {e}")
#         return 50