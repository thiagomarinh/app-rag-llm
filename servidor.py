from app import chain
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
    title="Encontre meu produto",
    version="1.0",
    description="Simples servidor usando langServer, para encontrar produtos ou serviços em podcasts",
)

add_routes(app, chain, enable_feedback_endpoint=True, path="/app")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)