FROM python:3.10-slim-buster

# Set environment variables for threading
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
    transformers \
    sentencepiece \
    sentence-transformers \
    fastapi \
    uvicorn \
    nmslib \
    pandas \
    pymongo[srv]==4.6.0 \
    python-Levenshtein \
    pyspellchecker \
    spacy \
    pytextrank \
    && python3 -m spacy download fr_dep_news_trf

# Copy application code
WORKDIR /www
COPY ./src /www/

# Set command to run application
CMD ["python3", "/www/code/index.py"]
