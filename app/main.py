from fastapi import FastAPI, HTTPException

from app.model import classifier

from app.schemas import ClassifyRequest, ClassifyResponse