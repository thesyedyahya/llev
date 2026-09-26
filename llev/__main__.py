import uvicorn

from .api import create_app

if __name__ == "__main__":
    import os

    uvicorn.run(create_app(), host=os.getenv("LLEV_HOST", "0.0.0.0"), port=int(os.getenv("LLEV_PORT", "8088")))
