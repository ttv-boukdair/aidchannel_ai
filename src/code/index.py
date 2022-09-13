import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib
from pymongo import MongoClient
import time
l = "mongodb://tunisie-tn-jobs:gn!%40Qg%5EFH94MW%5E5Q7me%24@51.77.134.195:29098/tunisie-tn-jobs?authSource=test&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"

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
    ids, dis = get_cos_sim(text, model, index, 10)
    res = format_res(ids, dis, rtmc)
    return res


@app.get('/normalize-tunisie-utmc')
def normalize_tunisie_utmc():
    while True:
        # TO DO !!!!
        if len(1):
            pass
        else:
            time.sleep(300)


    return ''



def getRTMC():
    rtmc_jobs_appellations = []
    cur_rtmc_jobs = db.rtmcjobs.find({})
    for job in cur_rtmc_jobs:
        rtmc_jobs_appellations.append(job)
    cur_rtmc_appellations = db.rtmcappelations.find({})
    for appellation in cur_rtmc_appellations:
        rtmc_jobs_appellations.append(appellation)
    return rtmc_jobs_appellations

def vectorizeJobs():
    rtmc_jobs_vectors = [model.encode(r.name) for r in rtmc]
    return rtmc_jobs_vectors

def get_cos_sim(text, model, index, k):
    q = model.encode(text)
    ids, distances = index.knnQuery(q, k=k)
    return ids.tolist(), distances.tolist()

def format_res(ids, dis, rtmc):
    formated_res = []
    for i in range(len(ids)):
        formated_res.append([rtmc[ids[i]]._id, rtmc[ids[i]].name, dis[i]])
    return formated_res

if __name__ == '__main__':
    print('loading models ...')
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    print('connect to db and vectorize jobs ...')
    client = MongoClient(l)
    db = client['tunisie-tn-jobs']
    rtmc = getRTMC()
    rtmc_jobs_vectors = vectorizeJobs()
    print('indexing vectors jobs ...')
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(rtmc_jobs_vectors)
    index.createIndex({'post': 2}, print_progress=True)
    print('API Ready !!!')
    uvicorn.run(app, host='0.0.0.0',port = 80)
