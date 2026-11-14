import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import chat, ingestion
from app.core.config import get_settings
from app.core.exceptions import AppError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, version="1.0.0")

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(ingestion.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    return app


app = create_app()
