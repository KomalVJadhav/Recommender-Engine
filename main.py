from fastapi import FastAPI
from app.api import router
from app.logger import setup_logger

app = FastAPI(title="Contextual Topic Recommendation Engine")
setup_logger()
app.include_router(router)
