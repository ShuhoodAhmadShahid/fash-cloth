import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from services.image_processing import analyze_image

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = analyze_image(image)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
