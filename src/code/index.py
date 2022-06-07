import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib



DB = "mongodb://aidchannel:aidchannel_password123456@51.77.134.195:27028/aidchannel?authSource=aidchannel"

DATA_PATH = '/www/data/'
app = FastAPI(debug = True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class Input(BaseModel):
    text: str

#summary model
class InputSummary(BaseModel):
    text: str
    limit_phrases: int
    limit_sentences: int

@app.get('/')
def hello():
    return 'hello world'



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0',port = 80)
