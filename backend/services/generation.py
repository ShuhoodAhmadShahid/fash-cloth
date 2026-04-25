import os
import asyncio
import base64
import httpx
from typing import List

# HF changed their endpoint. Router is the new primary; legacy as fallback.
_HF_ROUTER = "https://router.huggingface.co/hf-inference/models"
_HF_LEGACY = "https://api-inference.huggingface.co/models"

_MODEL_TXT2IMG = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
# Using SD v1.5 Inpainting which is more widely supported on HF serverless
_MODEL_IMG2IMG = "runwayml/stable-diffusion-v1-5-inpainting"

_TRY_ON_PIPELINE = None
_SD_TURBO_PIPELINE = None

def _get_sd_turbo():
    global _SD_TURBO_PIPELINE
    if _SD_TURBO_PIPELINE is None:
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] Loading local SD Turbo on {device}...")
            _SD_TURBO_PIPELINE = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo", 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                local_files_only=True
            ).to(device)
        except Exception as e:
            print(f"[WARNING] Local SD Turbo not available: {e}")
    return _SD_TURBO_PIPELINE


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        # We don't raise error here, let the caller handle missing token
        return {}
    return {"Authorization": f"Bearer {token}"}


def _get_vton_pipeline():
    global _TRY_ON_PIPELINE
    if _TRY_ON_PIPELINE is None:
        try:
            import sys
            import os
            # Add libs/fashn-vton/src to sys.path
            vton_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs", "fashn-vton", "src"))
            if vton_src_path not in sys.path:
                sys.path.insert(0, vton_src_path)
                
            from fashn_vton import TryOnPipeline
            # Path to weights in libs directory
            weights_dir = os.path.join(os.path.dirname(__file__), "..", "libs", "fashn-vton", "weights")
            if os.path.exists(weights_dir):
                print(f"[DEBUG] Initializing TryOnPipeline with weights from {weights_dir}")
                _TRY_ON_PIPELINE = TryOnPipeline(weights_dir=weights_dir)
            else:
                print(f"[WARNING] VTON weights not found at {weights_dir}. Local VTON will be disabled.")
        except Exception as e:
            print(f"[ERROR] Failed to load VTON pipeline: {e}")
    return _TRY_ON_PIPELINE


async def _call(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    attempt: int = 0,
) -> bytes:
    resp = await client.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=120.0,
    )

    if resp.status_code == 200:
        ct = resp.headers.get("content-type", "")
        if "image" in ct or len(resp.content) > 1000:
            return resp.content
        raise RuntimeError(f"Non-image response: {resp.text[:200]}")

    # Model still warming up — wait and retry
    if resp.status_code == 503 and attempt < 4:
        try:
            wait = float(resp.json().get("estimated_time", 20))
        except Exception:
            wait = 20.0
        await asyncio.sleep(min(wait, 30))
        return await _call(client, url, payload, attempt + 1)

    raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")


async def _try_urls(client: httpx.AsyncClient, model: str, payload: dict) -> bytes:
    """Try the new HF router first, then the legacy endpoint."""
    last_err = None
    for base in [_HF_ROUTER, _HF_LEGACY]:
        try:
            return await _call(client, f"{base}/{model}", payload)
        except RuntimeError as e:
            last_err = e
            # 404 = model not on this endpoint; try the other one
            if "404" not in str(e):
                raise   # non-404 errors (401, 503 timeout) bubble up immediately
    raise RuntimeError(str(last_err))


async def _img2img(
    client: httpx.AsyncClient,
    person_image: bytes,
    garment: str,
    color: str,
    style: str,
    gender: str,
) -> bytes:
    """Inpainting: replaces the clothing area while keeping the person's face identical."""
    from PIL import Image
    import io
    from services.image_processing import create_clothing_mask
    
    # 1. Prepare images
    img = Image.open(io.BytesIO(person_image)).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS) # Inpainting works best at 512
    
    # 2. Generate mask for clothes (person minus face)
    mask = create_clothing_mask(img)
    
    # 3. Encode to base64
    def to_b64(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    img_b64 = to_b64(img)
    mask_b64 = to_b64(mask)
    
    # 4. Prompt as per user requirements
    prompt = (
        f"A high-quality fashion catalog image of a {gender} wearing a {style} {color} {garment}, "
        f"full body, clean background, realistic lighting, detailed fabric. "
        f"Same person, same face, wearing the described outfit, preserve identity, realistic."
    )
    
    negative_prompt = (
        "different face, distorted, unrealistic, extra limbs, "
        "low quality, facial distortion, face change, blurry"
    )
    
    return await _try_urls(
        client,
        _MODEL_IMG2IMG,
        {
            "inputs": {
                "image": img_b64,
                "mask": mask_b64,
                "prompt": prompt,
            },
            "parameters": {
                "negative_prompt": negative_prompt,
                "num_inference_steps": 10,
                "strength": 1.0, # Complete replacement inside the mask
                "guidance_scale": 7.5,
            },
        },
    )


async def _txt2img(
    client: httpx.AsyncClient,
    gender: str,
    skin_tone: str,
    garment: str,
    color: str,
    style: str,
) -> bytes:
    """FLUX.1-schnell text-to-image (fallback when img2img unavailable)."""
    prompt = (
        f"Fashion portrait, {skin_tone} skin tone {gender} "
        f"wearing a {garment} in {color}, {style}, "
        f"professional studio photography, clean white background, "
        f"detailed fabric texture, high quality"
    )
    return await _try_urls(
        client,
        _MODEL_TXT2IMG,
        {
            "inputs": prompt,
            "parameters": {"num_inference_steps": 4},
        },
    )


async def _generate_garment_only(
    client: httpx.AsyncClient,
    garment: str,
    color: str,
    style: str,
    gender: str = "female",
) -> bytes:
    """Generates a high-quality garment image on a white background for VTON."""
    # Tailor the prompt based on gender for better relevance
    fit = "menswear" if gender == "male" else "womenswear"
    prompt = (
        f"A high-quality fashion catalog image of a {style} {color} {garment} for {gender}, {fit}, "
        f"isolated on a clean white background, flat lay or on a invisible mannequin, "
        f"professional studio lighting, detailed fabric texture, high resolution, realistic."
    )
    
    # 1. Try local SD Turbo first (super fast, no timeout)
    sd_pipeline = _get_sd_turbo()
    if sd_pipeline:
        try:
            import io
            print(f"[DEBUG] Generating garment locally with SD Turbo for {garment}...")
            # SD Turbo works best with 1-4 steps
            image = sd_pipeline(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception as e:
            print(f"[WARNING] Local garment generation failed, falling back to API: {e}")

    # 2. Fallback to HF API
    print(f"[DEBUG] Generating garment via HF API for {garment}...")
    return await _try_urls(
        client,
        _MODEL_TXT2IMG,
        {
            "inputs": prompt,
            "parameters": {"num_inference_steps": 4},
        },
    )


async def _vton_local(
    person_image: bytes,
    garment_image: bytes,
    category: str = "tops",
) -> bytes:
    """Runs virtual try-on locally using fashn-vton-1.5."""
    from PIL import Image
    import io
    
    pipeline = _get_vton_pipeline()
    if pipeline is None:
        raise RuntimeError("VTON pipeline not initialized")

    person = Image.open(io.BytesIO(person_image)).convert("RGB")
    garment = Image.open(io.BytesIO(garment_image)).convert("RGB")

    # Run try-on
    result = pipeline(
        person_image=person,
        garment_image=garment,
        category=category,
        num_timesteps=10
    )

    buf = io.BytesIO()
    result.images[0].save(buf, format="JPEG")
    return buf.getvalue()


async def generate_outfit_images(
    person_image: bytes,
    gender: str,
    skin_tone: str,
    garments: List[str],
    colors: List[str],
    style: str,
    num_images: int = 1,
) -> List[str]:
    person_b64 = base64.b64encode(person_image).decode()
    # Check for HF Token if local VTON is not available
    pipeline = _get_vton_pipeline()
    if not pipeline and not os.environ.get("HF_TOKEN"):
        print("[WARNING] Local VTON not available and HF_TOKEN not set. Using mock results.")
        # Return empty list or we can return a message.
        # Frontend handles empty images gracefully now.
        return []

    results: List[str] = []

    async with httpx.AsyncClient() as client:
        for i in range(min(num_images, 1)):
            garment = garments[i % len(garments)]
            color   = colors[i % len(colors)]

            try:
                # 1. Try Local VTON Pipeline (High Quality)
                if pipeline:
                    print(f"[DEBUG] Attempting local VTON for {garment}")
                    # A. Generate relevant garment image first
                    garment_bytes = await _generate_garment_only(client, garment, color, style, gender)
                    
                    # B. Determine category
                    category = "tops"
                    g_lower = garment.lower()
                    if any(x in g_lower for x in ["pant", "skirt", "trouser", "jean", "short"]):
                        category = "bottoms"
                    elif any(x in g_lower for x in ["dress", "gown", "suit", "jumpsuit"]):
                        category = "one-pieces"
                    
                    # C. Run local VTON
                    image_bytes = await _vton_local(person_image, garment_bytes, category)
                else:
                    # 2. Fallback to HF Inpainting
                    print(f"[DEBUG] Falling back to HF Inpainting for {garment}")
                    image_bytes = await _img2img(client, person_image, garment, color, style, gender)
                    
            except Exception as e:
                print(f"[DEBUG] Generation failed for {garment}: {e}")
                continue 

            b64 = base64.b64encode(image_bytes).decode()
            results.append(f"data:image/jpeg;base64,{b64}")

    return results
