#file thaT Contains the main FastAPI application and endpoints

from fastapi import FastAPI, HTTPException

from app.model import classifier

from app.schemas import ClassifyRequest, ClassifyResponse

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")