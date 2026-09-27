#file thaT Contains the main FastAPI application and endpoints

#class used for app object
from unittest import result

from fastapi import FastAPI, HTTPException

#loaded model instance
from app.model import classifier

#schemas for request and response validation
from app.schemas import RequestData, ResponseData

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ResponseData)
def classify(request: RequestData):
    try:
        result = classifier.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}") from e
        
    return ResponseData(**result)