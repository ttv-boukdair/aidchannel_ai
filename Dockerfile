FROM ubuntu:20.04

RUN apt-get update -y
RUN apt update -y
RUN apt-get install -y build-essential
RUN apt-get install -y python3.11
RUN apt install -y protobuf-compiler
WORKDIR /www
RUN apt install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pip setuptools wheel
RUN python3 -m pip install transformers
RUN python3 -m pip install sentencepiece
RUN python3 -m pip install sentence-transformers
RUN python3 -m pip install fastapi
RUN python3 -m pip install uvicorn
RUN python3 -m pip install nmslib
RUN python3 -m pip install promise
RUN python3 -m pip install requests
RUN python3 -m pip install pandas
RUN python3 -m pip install python-dotenv
# RUN python3 -m pip install pymongo[srv]
RUN python3 -m pip install spacy==3.6.0
RUN python3 -m spacy download fr_dep_news_trf
RUN python3 -m pip install pytextrank
RUN python3 -m pip install protobuf==3.20.1
RUN python3 -m pip install psycopg2-binary
RUN python3 -m pip install pymongo[srv]
RUN python3 -m pip install python-Levenshtein
RUN python3 -m pip install pyspellchecker
COPY ./src /www/
CMD ["python3","/www/code/index.py"]
