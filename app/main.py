#file thaT Contains the main FastAPI application and endpoints

#class used for app object
from unittest import result

from fastapi import FastAPI, HTTPException

from app.model import classifier

#schemas for request and response validation
from app.schemas import RequestData, ResponseData

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/classify", response_model=ResponseData)
def classify(request: RequestData):
    try:
        result = classifier.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}") from e
        
    return ResponseData(**result)