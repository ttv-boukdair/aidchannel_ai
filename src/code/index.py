import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib



l = "mongodb+srv://mongo:xrAXHRuGtHpzxIIk@programmesideals.kh0sa.mongodb.net/programmes?retryWrites=true&w=majority"


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


@app.get('/')
def hello():
    return 'hello world'

@app.post('/sim-jobs')
def sim_jobs(input : Input):
    req = jsonable_encoder(input)
    text = req['text']
    ids, dis = get_cos_sim(text, model, index)
    res = format_res(ids, dis, umtc_jobs)
    return res

def getUMTC():
    data = db["umtcs"].find({'version':10})
    umtc = []
    for c in data:
        umtc.append(c)
    return umtc

def vectorizeJobs():
    umtc_jobs = [u['metier'] for u in umtc]
    umtc_jobs_vectors = [model.encode(u['metier']) for u in umtc]
    return umtc_jobs, umtc_jobs_vectors

def get_cos_sim(text, model, index):
    q = model.encode(text)
    ids, distances = index.knnQuery(q, k=10)
    return ids.tolist(), distances.tolist()

def format_res(ids, dis, umtc_jobs):
    formated_res = []
    for i in range(len(ids)):
        formated_res.append([umtc_jobs[ids[i]], dis[i]])
    return formated_res

if __name__ == '__main__':
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    client = pymongo.MongoClient(l)
    db = client.programmes
    umtc = getUMTC()
    umtc_jobs, umtc_jobs_vectors = vectorizeJobs()
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(umtc_jobs_vectors)
    index.createIndex({'post': 2}, print_progress=True)

    uvicorn.run(app, host='0.0.0.0',port = 80)
