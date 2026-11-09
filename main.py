"""Run the camera recording FastAPI service."""

import os

import uvicorn

from app.main import app, recorder


if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
    server = uvicorn.Server(config)
    recorder.set_shutdown_callback(lambda: setattr(server, "should_exit", True))
    server.run()
