import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib
import spacy
import pandas as pd
import numpy as np
from numpy.linalg import norm
from Levenshtein import distance
from spellchecker import SpellChecker

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# MongoDB URI and Data Path
l = "mongodb://tunisie-tn-jobs:gn!%40Qg%5EFH94MW%5E5Q7me%24@51.77.134.195:29098/tunisie-tn-jobs?authSource=test&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
DATA_PATH = '/www/data/'
aneti_site = 'www.emploi.nat.tn'

# FastAPI App Setup
app = FastAPI(debug=True)
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

class Input_compare(BaseModel):
    text1: str
    text2: str

# MongoDB Client with Connection Pooling
client = MongoClient(l, maxPoolSize=5)
db = client['tunisie-tn-jobs']

# Thread Locks for Shared Resources
index_lock = threading.Lock()
index_competences_lock = threading.Lock()

# Thread Pool for Background Tasks
executor = ThreadPoolExecutor(max_workers=5)

@app.get("/")
def hello():
    return "RTMC AI NORM"

@app.post("/compare-sim")
def compare_sim(input: Input_compare):
    req = jsonable_encoder(input)
    text1 = req['text1']
    text2 = req['text2']
    res = compare2texts_sim(text1, text2)
    return res

@app.post("/compare-leneshtein")
def compare_leneshtein(input: Input_compare):
    req = jsonable_encoder(input)
    text1 = req['text1']
    text2 = req['text2']
    res = compare2texts_levenshtein(text1, text2)
    return res

@app.get("/correct-titles")
def correct_titles():
    executor.submit(process_correct_titles)
    return {"status": "Task started"}

def process_correct_titles():
    sleep_if_no_offer = 5
    while True:
        offer = db.offers.find_one({'is_title_corrected': {'$ne': True}})
        if offer:
            title = offer['title']
            special_char = '�'
            if special_char in title:
                db.offers.update_one(
                    {'_id': offer['_id']},
                    {'$set': {'title_corrected': title_correction(title), 'is_title_corrected': True}}
                )
            else:
                db.offers.update_one({'_id': offer['_id']}, {'$set': {'is_title_corrected': True}})
        else:
            time.sleep(sleep_if_no_offer * 60)

@app.get("/normalize-tunisie-rtmc")
def normalize_tunisie_rtmc():
    executor.submit(process_normalize_tunisie_rtmc)
    return {"status": "Task started"}

def process_normalize_tunisie_rtmc():
    max_offer_to_proccess_before_sleep = 50
    sleep_if_no_offer = 15
    count_offer = 0
    while True:
        if count_offer >= max_offer_to_proccess_before_sleep:
            time.sleep(60)
            count_offer = 0
        offer = db.offers.find_one({'config_is_normalized': {'$ne': True}})
        if offer:
            title = offer['title']
            ids, dis = get_cos_sim(title, model, index, 1)
            res = format_res_id(ids, dis, rtmc)
            if len(res):
                rtmc_job_designation_id, rtmc_metier_id, rtmc_job_designation_title, config_normalized_degre = res[0]
                db.offers.update_one(
                    {'_id': offer['_id']},
                    {'$set': {'config_is_normalized': True, 'rtmc_appelation_id': rtmc_job_designation_id,
                              'rtmc_metier_id': rtmc_metier_id, 'rtmc_score': config_normalized_degre}}
                )
            else:
                db.offers.update_one(
                    {'_id': offer['_id']},
                    {'$set': {'config_is_normalized': True, 'rtmc_appelation_id': None,
                              'rtmc_metier_id': None, 'rtmc_score': -1}}
                )
            count_offer += 1
        else:
            time.sleep(sleep_if_no_offer * 60)

@app.get("/normalize-tunisie-skills")
def normalize_tunisie_skills():
    executor.submit(process_normalize_tunisie_skills)
    return {"status": "Task started"}

def process_normalize_tunisie_skills():
    max_offer_to_proccess_before_sleep = 500
    sleep_if_no_offer = 15
    count_offer = 0
    while True:
        if count_offer >= max_offer_to_proccess_before_sleep:
            time.sleep(60)
            count_offer = 0
        offer = db.offers.find_one({'config_is_skill_normalized': {'$ne': True}})
        if offer:
            title = offer['title']
            description = offer['description']
            text = title + ' \n ' + description
            res = []
            try:
                if offer['website'] == aneti_site:
                    res = get_skills_aneti(description, df_noise, 15, 1)
                elif sentsLength(nlp(text)) <= 3:
                    res = get_skills_1(text)
                else:
                    res = get_skills_2(text, df_noise, 15, 3)
            except Exception as e:
                print(e)

            rtmc_skills_id = [s[0] for s in res.get('competences', [])]
            db.offers.update_one(
                {'_id': offer['_id']},
                {'$set': {'config_is_skill_normalized': True, 'rtmc_skills_id': rtmc_skills_id}}
            )
            count_offer += 1
        else:
            time.sleep(sleep_if_no_offer * 60)

# Model and Data Initialization
print('Loading models...')
spell_fr = SpellChecker(language='fr')
df_noise = pd.read_csv('/www/code/NoiseAction.csv')
model = SentenceTransformer('dangvantuan/sentence-camembert-large')
print('Connecting to DB and vectorizing jobs...')
nlp = spacy.load('fr_dep_news_trf')
rtmc = getRTMC()
rtmc_jobs_vectors = vectorizeJobs()
print('Indexing vectors jobs...')
competences = getRTMCSkills()
competences_vectors = vectorizeSkills(competences)
index = nmslib.init(method='hnsw', space='cosinesimil')
with index_lock:
    index.addDataPointBatch(rtmc_jobs_vectors)
    index.createIndex({'post': 2}, print_progress=True)

index_competences = nmslib.init(method='hnsw', space='cosinesimil')
with index_competences_lock:
    index_competences.addDataPointBatch(competences_vectors)
    index_competences.createIndex({'post': 2}, print_progress=True)

print('API Ready !!!')
uvicorn.run(app, host='0.0.0.0', port=80)
