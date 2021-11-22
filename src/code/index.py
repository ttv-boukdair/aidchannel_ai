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
#SUMMARIZE
import spacy
import pytextrank
from icecream import ic
from math import sqrt
from operator import itemgetter

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

@app.post('/summary')
def summary(input : InputSummary):
    req = jsonable_encoder(input)
    text = req['text']
    limit_phrases = req['limit_phrases']
    limit_sentences = req['limit_sentences']
    res = summarize(text, limit_phrases, limit_sentences)
    
    return res

@app.post('/summary-desc')
def summary_desc(input : Input):
    req = jsonable_encoder(input)
    text = req['text']
    limit_phrases = 4
    limit_sentences = 2
    res = summarize(text, limit_phrases, limit_sentences)
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




#SuMMARIZE FUNC
def summarize(text, limit_phrases, limit_sentences):
  if text == None:
    text =''
  doc = nlp(text)
  res = []
  #Construct a list of the sentence boundaries with a phrase vector (initialized to empty set) for each...
  sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]

  #Iterate through the top-ranked phrases, added them to the phrase vector for each sentence...
  phrase_id = 0
  unit_vector = []

  for p in doc._.phrases:
      ic(phrase_id, p.text, p.rank)

      unit_vector.append(p.rank)

      for chunk in p.chunks:
          ic(chunk.start, chunk.end)

          for sent_start, sent_end, sent_vector in sent_bounds:
              if chunk.start >= sent_start and chunk.end <= sent_end:
                  ic(sent_start, chunk.start, chunk.end, sent_end)
                  sent_vector.add(phrase_id)
                  break

      phrase_id += 1

      if phrase_id == limit_phrases:
          break

  sum_ranks = sum(unit_vector)

  unit_vector = [ rank/sum_ranks for rank in unit_vector ]

  sent_rank = {}
  sent_id = 0

  for sent_start, sent_end, sent_vector in sent_bounds:
      ic(sent_vector)
      sum_sq = 0.0
      ic
      for phrase_id in range(len(unit_vector)):
          ic(phrase_id, unit_vector[phrase_id])

          if phrase_id not in sent_vector:
              sum_sq += unit_vector[phrase_id]**2.0

      sent_rank[sent_id] = sqrt(sum_sq)
      sent_id += 1

  sorted(sent_rank.items(), key=itemgetter(1)) 

  sent_text = {}
  sent_id = 0

  for sent in doc.sents:
      sent_text[sent_id] = sent.text
      sent_id += 1

  num_sent = 0

  for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
      res.append(sent_text[sent_id])
      num_sent += 1

      if num_sent == limit_sentences:
          break
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
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("textrank", last=True)
    uvicorn.run(app, host='0.0.0.0',port = 80)
