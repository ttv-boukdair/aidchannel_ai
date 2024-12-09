FROM ubuntu:22.04
# Set environment variables to make tzdata install non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
# Ensure the package lists are updated
RUN apt-get update -y && apt-get upgrade -y

# Install necessary tools and Python 3.10 from deadsnakes PPA
RUN apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils build-essential protobuf-compiler

# Use get-pip.py to install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade setuptools and wheel
RUN python3.10 -m pip install --upgrade setuptools wheel
# RUN apt install -y python3-pip
# RUN python3.10 -m pip install --upgrade pip
# RUN python3.10 -m pip install pip setuptools wheel
RUN python3.10 -m pip install -U transformers
RUN python3.10 -m pip install sentencepiece
RUN python3.10 -m pip install -U sentence-transformers
RUN python3.10 -m pip install fastapi
RUN python3.10 -m pip install uvicorn
RUN python3.10 -m pip install nmslib
RUN python3.10 -m pip install promise
RUN python3.10 -m pip install requests
RUN python3.10 -m pip install pandas
RUN python3.10 -m pip install python-dotenv
# RUN python3.10 -m pip install pymongo[srv]
RUN python3.10 -m pip install spacy==3.6.0
RUN python3.10 -m spacy download fr_dep_news_trf
RUN python3.10 -m pip install pytextrank
RUN python3.10 -m pip install protobuf==3.20.1
RUN python3.10 -m pip install psycopg2-binary
RUN python3.10 -m pip install pymongo[srv]
RUN python3.10 -m pip install python-Levenshtein
RUN python3.10 -m pip install pyspellchecker
WORKDIR /www
COPY ./src /www/
CMD ["python3.10","/www/code/index.py"]
