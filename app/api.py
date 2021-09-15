from collections import defaultdict
import os

from dotenv import load_dotenv, find_dotenv
from fastapi import Body, FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import srsly
import uvicorn
import numpy as np

from app.models import AIModel
from app.core.AI.neuronka import network


load_dotenv(find_dotenv())
app = FastAPI()


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse("/docs")


@app.post("/ai")
def get_result(data: AIModel):
    data_list = np.array([
        [data.weight, data.height],
    ])
    res = network.result(data_list)
    if res >= 0.7:
        return "Женщина"
    else:
        return "Мужчина"
    
