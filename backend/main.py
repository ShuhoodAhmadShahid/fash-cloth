import os
import sys

# Add the backend directory to sys.path to resolve imports like 'routes'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.recommend import router as recommend_router
from routes.generate import router as generate_router

app = FastAPI(title="Fashion AI Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(recommend_router)
app.include_router(generate_router)

@app.on_event("startup")
async def startup_event():
    print("==========================================")
    print("Pre-loading local models...")
    try:
        from services.image_processing import _get_gender_model
        _get_gender_model()
    except Exception as e:
        print(f"Error loading gender model: {e}")
        
    print("Models pre-loaded successfully.")
    print("==========================================")


@app.get("/health")
async def health():
    return {"status": "ok"}


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
