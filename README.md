# BreadVec

A small vector API service for generating mixedbred.ai vectors.  It's currently using the mxbai-embed-large-v1 model.

## Local Installation

```
pip install -r requirements.txt
```

## Local Running

```
uvicorn main:app --host 0.0.0.0 --port 3006
```

**Warning**: The first run will be VERY slow to load

Visit `http://localhost:3006/docs` in a browser once it's loaded

Call it in python like this:

```
# Function to call the text embedder
def embed(text):
    response = requests.get(embedder["embedding_endpoint"], params={"text":text, "instruction": "Represent this text for retrieval:" }, headers={"accept": "application/json"})
    vector_embedding = response.json()
    return vector_embedding
```
