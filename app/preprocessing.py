import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Prétraite l'image pour le modèle
    """
    try:
        # Conversion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionnement
        image = image.resize(target_size)
        
        # Conversion en numpy array
        image_array = np.array(image)
        
        # Normalisation (0-1)
        image_array = image_array.astype('float32') / 255.0
        
        # Pour TensorFlow : reshape si nécessaire
        # image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image prétraitée - Shape: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {e}")
        raise