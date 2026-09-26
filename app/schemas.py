#file for validation of request and response data

from pydantic import BaseModel, Field

# Validation for request data
class RequestData(BaseModel):
    text: str = Field(..., min_length = 1, 
                      max_length = 2500, 
                      description = "SMS Message")

class ResponseData(BaseModel):
    label: str
    confidence: float
    raw_score: float
