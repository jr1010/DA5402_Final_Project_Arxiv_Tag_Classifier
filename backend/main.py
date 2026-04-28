import logging
from fastapi import FastAPI, HTTPException

from backend.schema import (
    PredictRequest,
    BulkPredictRequest,
    PredictResponse,
    BulkPredictResponse
)

import backend.inference as inference


# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Arxiv Classifier API")


# -------------------------
# Startup Event
# -------------------------
@app.on_event("startup")
def startup_event():
    logger.info("Starting API initialization")

    try:
        inference.initialize()
        logger.info("Initialization successful")

    except Exception:
        logger.exception("Startup initialization failed")


# -------------------------
# Health Check (Liveness)
# -------------------------
@app.get("/health")
def health():
    logger.info("Health check called")

    if inference.STATE == "error":
        logger.error(f"Health check failed | state=error")
        raise HTTPException(
            status_code=500,
            detail=f"System error: {inference.ERROR_MSG}"
        )

    if inference.vectorizer is None or inference.mlb is None:
        logger.error("Health check failed | artifacts not loaded")
        raise HTTPException(
            status_code=500,
            detail="Artifacts not loaded"
        )

    return {
        "status": "healthy",
        "state": inference.STATE
    }


# -------------------------
# Readiness Check
# -------------------------
@app.get("/ready")
def ready():
    logger.info("Readiness check called")

    if inference.STATE == "ready":
        return {"status": "ready"}

    elif inference.STATE == "starting":
        logger.warning("Model still starting")
        raise HTTPException(
            status_code=503,
            detail="Model is starting"
        )

    elif inference.STATE == "error":
        logger.error("Model in error state")
        raise HTTPException(
            status_code=503,
            detail=f"Model error: {inference.ERROR_MSG}"
        )


# -------------------------
# Detailed Status
# -------------------------
@app.get("/status")
def status():
    logger.info("Status endpoint called")

    return {
        "state": inference.STATE,
        "error": inference.ERROR_MSG
    }


# -------------------------
# Root
# -------------------------
@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {"message": "API is running"}


# -------------------------
# Single Prediction
# -------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Single prediction request received")

    try:
        labels = inference.predict_single(req.text)
        logger.info(f"Single prediction successful | labels_count={len(labels)}")

        return {"labels": labels}

    except Exception:
        logger.exception("Single prediction failed")

        raise HTTPException(
            status_code=503,
            detail="Prediction failed"
        )


# -------------------------
# Bulk Prediction
# -------------------------
@app.post("/predict-batch", response_model=BulkPredictResponse)
def predict_batch_endpoint(req: BulkPredictRequest):
    logger.info(f"Batch prediction request received | size={len(req.texts)}")

    try:
        predictions = inference.predict_batch(req.texts)
        logger.info(f"Batch prediction successful | batch_size={len(predictions)}")

        return {"predictions": predictions}

    except Exception:
        logger.exception("Batch prediction failed")

        raise HTTPException(
            status_code=503,
            detail="Batch prediction failed"
        )