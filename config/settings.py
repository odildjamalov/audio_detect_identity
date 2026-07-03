from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_title: str = "Speaker Verification API"
    app_version: str = "0.1.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    reference_embeddings_dir: str
    model_source: str
    model_savedir: str


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()