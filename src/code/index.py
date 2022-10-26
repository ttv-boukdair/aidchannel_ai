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
import spacy
import pandas as pd
import os
l = "mongodb://tunisie-tn-jobs:gn!%40Qg%5EFH94MW%5E5Q7me%24@51.77.134.195:29098/tunisie-tn-jobs?authSource=test&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"

DATA_PATH = '/www/data/'
aneti_site='www.emploi.nat.tn'
app = FastAPI(debug = True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class Input(BaseModel):
    text: str
class Input_k(BaseModel):
    text: str
    k: int

@app.get('/')
def hello():
    return 'RTMC AI NORM'

# @app.post('/sim-jobs')
# def sim_jobs(input : Input):
#     req = jsonable_encoder(input)
#     text = req['text']
#     ids, dis = get_cos_sim(text, model, index, 10)
#     res = format_res(ids, dis, rtmc)
#     return res

# @app.post('/sim-skills')
# def sim_skills(input : Input_k):
#     req = jsonable_encoder(input)
#     text = req['text']
#     k = int(req['k'])
#     if sentsLength(nlp(text)) <= 3:
#         res = get_skills_1(text)
#     else:
#         if k:
#             res = get_skills_2(text, df_noise, k)
#         else:
#             res = get_skills_2(text, df_noise)
#     return res

# @app.post('/sim-skills-1')
# def sim_skills_1(input : Input_k):
#     req = jsonable_encoder(input)
#     text = req['text']
#     res = get_skills_1(text)
#     return res

# @app.post('/sim-skills-2')
# def sim_skills_2(input : Input_k):
#     req = jsonable_encoder(input)
#     text = req['text']
#     k = int(req['k'])
#     res = get_skills_2(text, df_noise, k)
#     return res

@app.get('/normalize-tunisie-rtmc')
def normalize_tunisie_rtmc():
    count_offer = 0
    max_offer_to_proccess_before_sleep = 50
    sleep_if_no_offer = 15
    sleep_btewheen_max_offer_to_proccess = 1
    while True:
        if count_offer >= max_offer_to_proccess_before_sleep:
            time.sleep(60 * sleep_btewheen_max_offer_to_proccess)
            count_offer = 0
        offer = db.offers.find_one({'config_is_normalized' : {'$ne': True}})
        if offer :
            title = offer['title']
            id = offer['_id']
            # cur = db.offers.update_one({'_id': id}, {'$set':{'config_normalization_processing' : True}})
            # normalize offer return id job_designation degre
            ids, dis = get_cos_sim(title, model, index, 1)
            res = format_res_id(ids, dis, rtmc)
            if len(res):
                rtmc_job_designation_id,rtmc_metier_id,rtmc_job_designation_title,config_normalized_degre=res[0]

                # update offer(rtmc_job_designation_id, degre, is_normalized = True)
                cur = db.offers.update_one({'_id': id}, {'$set':{ 'config_is_normalized' : True, 'rtmc_appelation_id': rtmc_job_designation_id, 'rtmc_metier_id': rtmc_metier_id, 'rtmc_score': config_normalized_degre}})
            else:
                # in case there is a normalization pb update and don't set rtmc_job_id
                cur = db.offers.update_one({'_id': id}, {'$set':{ 'config_is_normalized' : True, 'rtmc_appelation_id': None, 'rtmc_metier_id': None, 'rtmc_score': -1}})
            count_offer += 1
        else:
            time.sleep(sleep_if_no_offer * 60)
    return ''

@app.get('/normalize-tunisie-skills')
def normalize_tunisie_skills():
    # processing parameters
    count_offer = 0
    max_offer_to_proccess_before_sleep = 500
    max_skills = 15
    sleep_btewheen_max_offer_to_proccess = 1
    while True:
        if count_offer >= max_offer_to_proccess_before_sleep:
            time.sleep(60 * sleep_btewheen_max_offer_to_proccess)
            count_offer = 0
        offer = db.offers.find_one({'config_is_skill_normalized' : {'$ne': True}})
        if offer :
            title = offer['title']
            id = offer['_id']
            res = []
            description = offer['description']
            text = title+' \n '+description
            if offer['website'] == aneti_site:
                res = get_skills_2(text, df_noise, max_skills, 1)
            elif sentsLength(nlp(text)) <= 3:
                try:
                    res = get_skills_1(text)
                except:
                    pass
            else:
                try:
                    res = get_skills_2(text, df_noise, max_skills)
                except:
                    pass
            if len(res):
                rtmc_skills_id=[s[0] for s in res['competences']]
                # update offer(rtmc_job_designation_id, degre, is_normalized = True)
                cur = db.offers.update_one({'_id': id}, {'$set':{ 'config_is_skill_normalized' : True, 'rtmc_skills_id': rtmc_skills_id}})
            else:
                # in case there is a normalization pb update and don't set rtmc_job_id
                cur = db.offers.update_one({'_id': id}, {'$set':{ 'config_is_skill_normalized' : True, 'rtmc_skills_id': None}})
            count_offer += 1
        else:
            time.sleep(sleep_if_no_offer * 60)
    return ''



def getRTMC():
    rtmc_jobs_appellations = []
    cur_rtmc_appellations = db.rtmcappelations.find({})
    for appellation in cur_rtmc_appellations:
        rtmc_jobs_appellations.append(appellation)
    return rtmc_jobs_appellations

def vectorizeJobs():
    rtmc_jobs_vectors = [model.encode(r['name']) for r in rtmc]
    return rtmc_jobs_vectors

def vectorizeSkills(rtmc):
    rtmc_vectors = []
    for i in range(len(rtmc)):
      rtmc_vectors.append(model.encode(rtmc[i]['name']))
      if i%100 == 0:
        print(i)
    return rtmc_vectors
def get_cos_sim(text, model, index, k):
    q = model.encode(text)
    ids, distances = index.knnQuery(q, k=k)
    return ids.tolist(), distances.tolist()

def format_res(ids, dis, rtmc):
    formated_res = []
    for i in range(len(ids)):
        formated_res.append([str(rtmc[ids[i]]['_id']), str(rtmc[ids[i]]['rtmc_job_id']), rtmc[ids[i]]['name'], dis[i]])
    return formated_res
def format_res_id(ids, dis, rtmc):
    formated_res = []
    for i in range(len(ids)):
        formated_res.append([rtmc[ids[i]]['_id'], rtmc[ids[i]]['rtmc_job_id'], rtmc[ids[i]]['name'], dis[i]])
    return formated_res

def getRTMCSkills():
    rtmc_skills = []
    cur_rtmc_skills = db.rtmcskills.find({})
    for skill in cur_rtmc_skills:
        rtmc_skills.append(skill)
    return rtmc_skills

def format_res_skills(ids, dis, competences, deg = 0.5):
    formated_res = []
    for i in range(len(ids)):
        if dis[i] >= deg:
          break
        formated_res.append([competences[ids[i]]['_id'], competences[ids[i]]['code_ogr'], competences[ids[i]]['name'], dis[i]])
    return formated_res

def get_skills_1(text):
    try:
        ids, distances = get_cos_sim(text, model, index_competences, 10)
        return {'text': text, 'competences': format_res_skills(ids, distances, competences)}
    except Exception as e:
        print(e)
        return {'text': text, 'competences': []}

def get_skills_2(text, df_noise, max_skills = 100, top_k_sents = 3):
  # skills for all description 
  sent_skills = []
  # sentence chunks
  sents = []
  # spacy camembert pipeline tags, ner, dep ... etc
  doc = nlp(text)
  for sent in doc.sents:
    noise = noise_person(sent, df_noise)
    if noise:
      print(sent)
      continue
    sents.append(sent.text)
    ids, distances = get_cos_sim(sent.text, model, index_competences, top_k_sents)
    sent_skills = sent_skills + format_res_skills(ids, distances, competences)
  sent_skills = sorted(sent_skills, key=lambda tup: tup[2])
  return {'text': text, 'sents': sents,'competences': sent_skills[:max_skills]}

def sentsLength(doc):
    i = 0
    for sent in doc.sents:
        if docLength(sent):
            i+=1
    return i

def docLength(doc):
  i = 0 
  for token in doc:
    if token.is_stop or token.tag_ in ['DET', 'ADP', 'CCONJ']:
      continue
    i+=1
  return i

def noise_person(text, df_noise):
  #if sentence is without useful words
  if not docLength(text):
    return True
  # if verb in sentence is in 1 person singular or plural return true
  for token in text:
    if(token.tag_ == 'VERB' and token.morph.get('Person') == ['1']):
      return True
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.morph.get('Person'))
  # if sentence contains org or money it is a noise for skills
  for ent in text.ents:
    if ent == 'ORG' or ent == 'MONEY':
      return True
  # if noise snippet exists in text return True
  noise_list = [n.lower() for n in df_noise['Bruit'].tolist()]
  for noise in noise_list:
    if noise in text.text.lower():
      return True
  return False

if __name__ == '__main__':
    print('loading models ...')
    df_noise = pd.read_csv('/www/code/NoiseAction.csv')
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    print('connect to db and vectorize jobs ...')
    nlp = spacy.load('fr_dep_news_trf')
    client = MongoClient(l)
    db = client['tunisie-tn-jobs']
    rtmc = getRTMC()
    rtmc_jobs_vectors = vectorizeJobs()
    print('indexing vectors jobs ...')
    competences = getRTMCSkills()
    competences_vectors = vectorizeSkills(competences)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(rtmc_jobs_vectors)
    index.createIndex({'post': 2}, print_progress=True)
    index_competences = nmslib.init(method='hnsw', space='cosinesimil')
    index_competences.addDataPointBatch(competences_vectors)
    index_competences.createIndex({'post': 2}, print_progress=True)
    print('API Ready !!!')
    uvicorn.run(app, host='0.0.0.0',port = 80)
