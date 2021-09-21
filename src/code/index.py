import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pymongo import MongoClient
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib
import numpy as np

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


@app.post('/sim-orgs')
def sim_orgs(input : Input):
    req = jsonable_encoder(input)
    text = req['text']
    if len(text) <= 10:
        deg=0.4
    else:
        deg=0.3
    ids, dis = get_cos_sim(text)
    res = format_res(ids, dis, deg)
    return res

@app.post('/sim-desc')
def sim_desc(input : Input):
    req = jsonable_encoder(input)
    text = req['text']
    res = get_desc_sim(text)
    return res


@app.post('/sim-sect')
def sim_sect(input : Input):
    req = jsonable_encoder(input)
    text = req['text']
    res = get_sect_sim(text)
    return res

def get_sect_sim(text):
    vect = model.encode(text)
    min = []
    for i in thems_vects:
        deg = float(cos_sim(vect, i[1]))
        min.append([str(i[0]), deg])
    return sorted(min, key = lambda i: i[1], reverse=True) 

def format_res(ids, dis, deg):
    formated_res = []
    for i in range(len(ids)):
        if dis[i]>=deg:
            break
        formated_res.append([orgs[ids[i]][0], orgs[ids[i]][1], dis[i]])
    return formated_res

def getOrgs():
    cloud_client=MongoClient(DB)
    cloud_db = cloud_client.aidchannel
    cur = cloud_db.organizations.find({"head_office_id":{"$exists":False}})
    res = []
    for i in cur:
        res.append((str(i['_id']), i['name']))
    return res

def encodeOrgs(model, orgs):
    res_code = []
    for i in orgs:
        res_code.append(model.encode(i[1]))
    return res_code

def encodeDescs():
    res_code = []
    for i in descs:
        res_code.append(model.encode(i[1]))
    return res_code

def cos_sim(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def get_desc_sim(text):
    q = model.encode(text)
    res = []
    for i in range(len(descs_encoded)):
        res.append([descs[i], cos_sim(q,descs_encoded[i])])
    return res

def get_cos_sim(text):
    q = model.encode(text)
    ids, distances = index.knnQuery(q, k=10)
    return ids.tolist(), distances.tolist()

def thematiques_vectors():
    res = []
    for i in thematiques:
        res.append([i,model.encode(i)])
    return res


if __name__ == '__main__':
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    descs = ['description', 'beneficiaries', 'objectives']
    orgs = getOrgs()
    descs_encoded = encodeDescs()
    encoded_orgs = encodeOrgs(model, orgs)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(encoded_orgs)
    index.createIndex({'post': 2}, print_progress=True)
    thematiques = ['Agriculture & Rural Development','Energy','Environment & Natural Resources','Global Health','Economic Development','Infrastructure','Gouvernance, Human Rights, Democracy, Public Sector','Education','Finance','Digital, Innovation','Humanitarian Assistance','Water & Sanitation','Urban Development & Transportation']
    thems_vects = thematiques_vectors()
    uvicorn.run(app, host='0.0.0.0',port = 80)
