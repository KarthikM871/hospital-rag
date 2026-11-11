from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.hospital_rag.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


class Health(BaseModel):
    status: str
    app_name: str
    environment: str


def create_app() -> FastAPI:
    app = FastAPI(title="Hospital RAG")

    @app.get("/health")
    def health(settings: SettingsDep) -> Health:
        return Health(
            status="ok",
            app_name=settings.app_name,
            environment=settings.environment,
        )

    return app


app = create_app()