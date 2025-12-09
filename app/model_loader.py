import os
import numpy as np
import logging
from tensorflow.keras.models import load_model as tf_load_model
# Si vous utilisez PyTorch :
# import torch

logger = logging.getLogger(__name__)

# Variable globale pour stocker le modèle
_model = None
_model_path = "models/cataract_model.h5"  # Ajustez l'extension selon votre modèle

def load_model():
    """
    Charge le modèle en mémoire
    """
    global _model
    
    try:
        # Vérifie si le fichier existe
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Le modèle n'existe pas à l'emplacement: {_model_path}")
        
        logger.info(f"Chargement du modèle depuis: {_model_path}")
        
        # Pour TensorFlow/Keras
        _model = tf_load_model(_model_path)
        
        # Pour PyTorch (décommentez si nécessaire)
        # _model = torch.load(_model_path, map_location=torch.device('cpu'))
        # _model.eval()
        
        logger.info("✅ Modèle chargé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

def predict(image_array: np.ndarray):
    """
    Fait une prédiction sur l'image prétraitée
    """
    global _model
    
    if _model is None:
        raise RuntimeError("Le modèle n'est pas chargé")
    
    try:
        # Vérification de la forme de l'image
        expected_shape = (224, 224, 3)  # Ajustez selon votre modèle
        if image_array.shape != expected_shape:
            # Redimensionnement si nécessaire
            from tensorflow.image import resize
            image_array = resize(image_array, [expected_shape[0], expected_shape[1]])
        
        # Ajout de la dimension batch
        input_tensor = np.expand_dims(image_array, axis=0)
        
        # Prédiction
        prediction = _model.predict(input_tensor, verbose=0)
        
        # Extraction de la probabilité
        # Ajustez selon la sortie de votre modèle
        confidence = float(prediction[0][0]) if len(prediction[0]) > 1 else float(prediction[0])
        
        # Seuil de décision (ajustable)
        has_cataract = confidence > 0.5
        
        return {
            "confidence": confidence,
            "has_cataract": has_cataract,
            "raw_prediction": prediction.tolist()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise