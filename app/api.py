from collections import defaultdict
import os

from dotenv import load_dotenv, find_dotenv
from fastapi import Body, FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import srsly
import uvicorn

from app.models import AIModel


load_dotenv(find_dotenv())
app = FastAPI()


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse("/docs")

