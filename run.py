# run.py - Production-grade entrypoint for hosting with uvicorn or gunicorn

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.logger import setup_logger

# Create FastAPI app
app = FastAPI(
    title="Topic Recommendation API",
    description="A contextual topic suggestion engine for annotation tools using Zero-shot and SetFit models.",
    version="1.0.0"
)

# Setup logging
setup_logger()

# CORS (Cross-Origin Resource Sharing) configuration for frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

# Entrypoint for development
if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
