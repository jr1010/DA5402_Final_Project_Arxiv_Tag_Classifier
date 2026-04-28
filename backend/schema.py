from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    text: str


class BulkPredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    labels: List[str]


class BulkPredictResponse(BaseModel):
    predictions: List[List[str]]