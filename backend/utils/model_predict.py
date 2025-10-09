import numpy as np
from model_training.train_model import FashionModel

def predict_fashion_score(image, model=None):
    """
    Main prediction function
    Input: preprocessed image (224x224x3, normalized to 0-1)
    Output: fashion score details
    """
    
    if model is None:
        model = FashionModel()
        try:
            model.load_model("models/fashion_model.h5")
        except:
            print("Model not found. Using default scoring.")
            return get_default_score()
    
    # Make prediction
    prediction = model.predict(image)
    
    if prediction is None:
        return get_default_score()
    
    return prediction

def get_default_score():
    """Return default score structure when model unavailable"""
    return {
        'style_match': 0.5,
        'texture_quality': 0.45,
        'confidence': 0.3
    }

def batch_predict(images, model=None):
    """
    Predict scores for multiple images
    Input: list of preprocessed images
    Output: list of predictions
    """
    
    if model is None:
        model = FashionModel()
        model.load_model("models/fashion_model.h5")
    
    predictions = []
    for img in images:
        pred = model.predict(img)
        predictions.append(pred)
    
    return predictions

def get_feature_importance(prediction):
    """
    Analyze which features contributed most to the score
    """
    style_match = prediction.get('style_match', 0)
    texture_quality = prediction.get('texture_quality', 0)
    confidence = prediction.get('confidence', 0)
    
    features = {
        'style_match': {
            'score': style_match * 10,
            'weight': 0.4,
            'importance': style_match * 0.4
        },
        'texture_quality': {
            'score': texture_quality * 10,
            'weight': 0.15,
            'importance': texture_quality * 0.15
        },
        'color_harmony': {
            'score': 5.0,
            'weight': 0.3,
            'importance': 0.3
        },
        'confidence': {
            'score': confidence * 10,
            'weight': 0.15,
            'importance': confidence * 0.15
        }
    }
    
    return features