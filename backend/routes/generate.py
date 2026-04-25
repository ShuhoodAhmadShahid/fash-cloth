import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from services.generation import generate_outfit_images

router = APIRouter()


@router.post("/generate")
async def generate(
    file: UploadFile = File(...),
    gender: str = Form(...),
    skin_tone: str = Form(...),
    garments: str = Form(...),   # JSON-encoded list
    colors: str = Form(...),     # JSON-encoded list
    style: str = Form(...),
):
    try:
        person_image  = await file.read()
        garments_list = json.loads(garments)
        colors_list   = json.loads(colors)

        images = await generate_outfit_images(
            person_image=person_image,
            gender=gender,
            skin_tone=skin_tone,
            garments=garments_list,
            colors=colors_list,
            style=style,
        )
        return {"images": images}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
