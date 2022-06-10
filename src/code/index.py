import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import nmslib
import psycopg2



pg_user= 'postgres'
pg_host= '51.77.134.195'
pg_database= 'tunisie'
pg_password= 'postgresGlobal16'
pg_port= 5435


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
    res = format_res(ids, dis, umtc)
    return res

def getUMTC():
    cur = conn.cursor()
    cur.execute("""SELECT * FROM data.rtmc_job_designation;""")
    umtc = cur.fetchall()
    return umtc

def vectorizeJobs():
    umtc_jobs_vectors = [model.encode(u[1]) for u in umtc]
    return umtc_jobs_vectors

def get_cos_sim(text, model, index):
    q = model.encode(text)
    ids, distances = index.knnQuery(q, k=10)
    return ids.tolist(), distances.tolist()

def format_res(ids, dis, umtc):
    formated_res = []
    for i in range(len(ids)):
        formated_res.append([umtcs[ids[i]][0], umtcs[ids[i]][1], dis[i]])
    return formated_res

if __name__ == '__main__':
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    conn = psycopg2.connect(host=pg_host,port=pg_port, dbname=pg_database, user=pg_user, password=pg_password)
    umtc = getUMTC()
    umtc_jobs_vectors = vectorizeJobs()
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(umtc_jobs_vectors)
    index.createIndex({'post': 2}, print_progress=True)

    uvicorn.run(app, host='0.0.0.0',port = 80)
