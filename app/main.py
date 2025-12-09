from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime
import os

# Importez vos modules
from app.model_loader import load_model, predict
from app.preprocessing import preprocess_image

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = FastAPI(
    title="Cataract Detection API",
    description="API de détection de cataracte par IA",
    version="1.0.0"
)

# CORS - Autorise toutes les origines pour la démo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargez le modèle au démarrage
@app.on_event("startup")
async def startup_event():
    logger.info("Chargement du modèle IA...")
    try:
        load_model()
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
        raise e

# Route de santé
@app.get("/")
async def root():
    return {
        "message": "Cataract Detection API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "cataract-detection-api"
    }

# Route de prédiction principale
@app.post("/predict")
async def predict_cataract(file: UploadFile = File(...)):
    """
    Endpoint pour détecter la cataracte sur une image d'œil
    """
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        logger.info(f"Traitement de l'image: {file.filename}")
        
        # Lecture de l'image
        contents = await file.read()
        
        # Conversion en image PIL
        image = Image.open(io.BytesIO(contents))
        
        # Vérification de la taille
        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image trop petite")
        
        # Prétraitement
        processed_image = preprocess_image(image)
        
        # Prédiction
        prediction_result = predict(processed_image)
        
        # Formatage de la réponse
        confidence = float(prediction_result["confidence"])
        has_cataract = prediction_result["has_cataract"]
        
        # Détermination de la sévérité
        if confidence < 0.3:
            severity = "Aucune"
            recommendation = "Aucune action nécessaire"
            color = "green"
        elif confidence < 0.6:
            severity = "Léger"
            recommendation = "Surveillance recommandée"
            color = "orange"
        elif confidence < 0.8:
            severity = "Modéré"
            recommendation = "Consultation ophtalmologique recommandée"
            color = "darkorange"
        else:
            severity = "Sévère"
            recommendation = "Consultation ophtalmologique urgente"
            color = "red"
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "has_cataract": has_cataract,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.1f}%",
                "severity": severity,
                "severity_color": color,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            },
            "image_info": {
                "format": image.format,
                "size": image.size,
                "filename": file.filename
            }
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {str(e)}")

# Endpoint test avec exemple
@app.get("/test-prediction")
async def test_prediction():
    """
    Endpoint de test avec une réponse simulée
    """
    return {
        "success": True,
        "prediction": {
            "has_cataract": False,
            "confidence": 0.15,
            "confidence_percentage": "15.0%",
            "severity": "Aucune",
            "severity_color": "green",
            "recommendation": "Aucune action nécessaire",
            "timestamp": datetime.now().isoformat()
        },
        "note": "Ceci est une réponse de test. Utilisez POST /predict pour une vraie prédiction."
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)