FROM ubuntu:20.04

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

# Install required Python packages
RUN python3.10 -m pip install \
    transformers \
    sentencepiece \
    sentence-transformers \
    fastapi \
    uvicorn \
    nmslib \
    promise \
    requests \
    pandas \
    python-dotenv \
    spacy==3.6.0 \
    pytextrank \
    protobuf==3.20.1 \
    psycopg2-binary \
    "pymongo[srv]" \
    python-Levenshtein \
    pyspellchecker

# Download spaCy language model
RUN python3.10 -m spacy download fr_dep_news_trf

# Set the working directory
WORKDIR /www

# Copy application code
COPY ./src /www/

# Default command to run the application
CMD ["python3.10", "/www/code/index.py"]