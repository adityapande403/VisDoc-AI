from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from visdoc.api.routes import router, set_retriever
from visdoc.retrieval.retriever import VisualDocumentRetriever
from visdoc.utils.logging import logger

def create_app(retriever: VisualDocumentRetriever | None = None) -> FastAPI:
    """Application factory for VisDoc AI REST API."""
    app = FastAPI(
        title="VisDoc AI - Multimodal Visual Document Retrieval API",
        description="Production visual document retrieval engine based on vision-language late-interaction MaxSim embeddings.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    if retriever is not None:
        set_retriever(retriever)

    return app

if __name__ == "__main__":
    import uvicorn
    # Localhost-only default binding for security
    uvicorn.run("visdoc.api.app:create_app", host="127.0.0.1", port=8000, factory=True, reload=True)
