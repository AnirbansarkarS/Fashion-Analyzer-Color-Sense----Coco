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
