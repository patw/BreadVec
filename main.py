import time
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Bread.ai model(s)
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device='cpu')
default_instruction = "Represent this sentence for searching relevant passages:"

app = FastAPI(title="BreadVectorizer",
        description="Text embedding using mixedbread.ai model mxbai-embed-large-v1. Instruction is optional, it will represent the text for retrieval by default.",
        version="1.0",
        contact={
            "name": "Pat Wendorf",
            "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    })

@app.get("/")
async def root():
    return {"message": "Vectorize some text! See /docs for more info."}

@app.get("/vectorize/")
async def vectorize(text: str, instruction: str = default_instruction):
    embedding = model.encode([[instruction,text]]).tolist()[0]
    return embedding

@app.get("/compare/")
async def compare(text1: str, text2: str, instruction: str = default_instruction):
    embedding1 = model.encode([[instruction,text1]]).tolist()[0]
    embedding2 = model.encode([[instruction,text2]]).tolist()[0]
    similarity = cos_sim(embedding1, embedding2)
    similarity = float(similarity[0][0])
    return {"cosine": similarity}

@app.get("/benchmark/")
async def benchmark(text: str, instruction: str = default_instruction):
    # Perform 10 calls to the model and calculate the average time
    times = []
    for _ in range(10):
        start_time = time.time()
        model.encode([[instruction, text]])
        end_time = time.time()
        times.append(end_time - start_time)
    
    average_time = sum(times) / 10
    return {"average_time": average_time}

