FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y python3.6
RUN apt-get install -y git
WORKDIR /www
RUN apt install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install -U sentence-transformers
RUN python3 -m pip install fastapi
RUN python3 -m pip install uvicorn
RUN python3 -m pip install nmslib
RUN python3 -m pip install promise
RUN python3 -m pip install requests
RUN python3 -m pip install pandas
RUN python3 -m pip install python-dotenv
RUN python3 -m pip install pymongo
RUN python3 -m pip install -U spacy
RUN python3 -m spacy download en_core_web_trf
RUN python3 -m pip install -U pytextrank
COPY ./src /www/
CMD ["python3","/www/code/index.py"]