import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.compare import router as compare_router
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.routes.search import router as search_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Drug Review ML Backend",
    version="1.0.0",
    description="FastAPI backend for drug-response prediction and analytics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(compare_router, prefix="/compare", tags=["Compare"])
