import os
import asyncio
import base64
import httpx
from typing import List
from dotenv import load_dotenv

# Point to the .env file in the backend directory
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_backend_dir, ".env")
load_dotenv(_env_path)


# HF Inference API endpoint (legacy api-inference.huggingface.co is dead)
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

_MODEL_TXT2IMG = os.environ.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")

_TRY_ON_PIPELINE = None
_SD_TURBO_PIPELINE = None

# ── Hardware detection ──────────────────────────────────────
def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

_GPU_AVAILABLE = _has_gpu()
if not _GPU_AVAILABLE:
    print("[INFO] No GPU detected — local portrait models disabled to save RAM.")
    print("[INFO] Local VTON (Try-On) is ENABLED on CPU (Will be very slow).")
    print("[INFO] Using HF Inference API as fallback for portraits.")

# Fallback models supported on HF Inference API
_FALLBACK_MODELS = [
    "black-forest-labs/FLUX.1-dev",
]


def _get_sd_turbo():
    if not _GPU_AVAILABLE:
        return None
    global _SD_TURBO_PIPELINE
    if _SD_TURBO_PIPELINE is None:
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] Loading local SD Turbo (Txt2Img) on {device}...")
            _SD_TURBO_PIPELINE = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo", 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                local_files_only=True
            ).to(device)
        except Exception as e:
            print(f"[WARNING] Local SD Turbo not available: {e}")
    return _SD_TURBO_PIPELINE


_INPAINTING_PIPELINE = None

def _get_inpainting_pipeline():
    if not _GPU_AVAILABLE:
        return None
    global _INPAINTING_PIPELINE
    if _INPAINTING_PIPELINE is None:
        try:
            from diffusers import AutoPipelineForInpainting
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] Loading local Inpainting model (SD Turbo) on {device}...")
            # We use SD Turbo for inpainting as well for speed and local availability
            _INPAINTING_PIPELINE = AutoPipelineForInpainting.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                local_files_only=True
            ).to(device)
        except Exception as e:
            print(f"[WARNING] Local Inpainting model not available: {e}")
    return _INPAINTING_PIPELINE


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Please check your .env file in the backend folder."
        )
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
        print(f"[DEBUG] Model loading, waiting {wait:.0f}s (attempt {attempt + 1}/4)...")
        await asyncio.sleep(min(wait, 30))
        return await _call(client, url, payload, attempt + 1)

    raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")


async def _try_api(client: httpx.AsyncClient, model: str, payload: dict) -> bytes:
    """Call the HF Inference API."""
    return await _call(client, f"{_HF_API_BASE}/{model}", payload)


async def _txt2img(
    client: httpx.AsyncClient,
    gender: str,
    skin_tone: str,
    garment: str,
    color: str,
    style: str,
) -> bytes:
    """Generates a fashion portrait using local or API models."""
    prompt = (
        f"Fashion portrait, {skin_tone} skin tone {gender} "
        f"wearing a {garment} in {color}, {style}, "
        f"professional studio photography, clean white background, "
        f"detailed fabric texture, high quality"
    )
    
    # 1. Try local SD Turbo first
    sd_pipeline = _get_sd_turbo()
    if sd_pipeline:
        try:
            import io
            print(f"[DEBUG] Generating portrait locally with SD Turbo...")
            image = sd_pipeline(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception as e:
            print(f"[WARNING] Local portrait generation failed: {e}")

    # 2. Fallback to HF API — try primary model, then fallbacks
    payload = {"inputs": prompt, "parameters": {"num_inference_steps": 4}}
    models_to_try = [_MODEL_TXT2IMG] + _FALLBACK_MODELS
    last_err = None
    for model in models_to_try:
        try:
            print(f"[DEBUG] Trying HF API model: {model}...")
            return await _try_api(client, model, payload)
        except RuntimeError as e:
            print(f"[DEBUG] Model {model} failed: {e}")
            last_err = e
            continue
    raise RuntimeError(f"All HF API models failed. Last error: {last_err}")


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

    # 2. Fallback to HF API — try primary model, then fallbacks
    payload = {"inputs": prompt, "parameters": {"num_inference_steps": 4}}
    models_to_try = [_MODEL_TXT2IMG] + _FALLBACK_MODELS
    last_err = None
    for model in models_to_try:
        try:
            print(f"[DEBUG] Trying HF API model: {model} for {garment}...")
            return await _try_api(client, model, payload)
        except RuntimeError as e:
            print(f"[DEBUG] Model {model} failed for {garment}: {e}")
            last_err = e
            continue
    raise RuntimeError(f"All HF API models failed for {garment}. Last error: {last_err}")


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

    # Run try-on (4 steps for CPU speed, 10+ for GPU quality)
    steps = 4 if not _GPU_AVAILABLE else 10
    result = pipeline(
        person_image=person,
        garment_image=garment,
        category=category,
        num_timesteps=steps
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
    results: List[str] = []

    async with httpx.AsyncClient() as client:
        for i in range(min(num_images, 1)):
            garment = garments[i % len(garments)]
            color   = colors[i % len(colors)]
            image_bytes = None

            # ── Path 1: Try Local VTON Pipeline (High Quality) ──
            try:
                pipeline = _get_vton_pipeline()
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
                    print(f"[DEBUG] ✓ Local VTON succeeded for {garment}")
            except Exception as e:
                print(f"[WARNING] Local VTON failed for {garment}: {e}")
                image_bytes = None

            # ── Path 2: Try Local Inpainting ──
            if image_bytes is None:
                try:
                    inpaint_pipeline = _get_inpainting_pipeline()
                    if inpaint_pipeline:
                        from PIL import Image
                        import io
                        from services.image_processing import create_clothing_mask
                        
                        print(f"[DEBUG] Attempting local inpainting for {garment}")
                        img = Image.open(io.BytesIO(person_image)).convert("RGB")
                        img = img.resize((512, 512), Image.LANCZOS)
                        mask = create_clothing_mask(img)
                        
                        prompt = (
                            f"A high-quality fashion catalog image of a {gender} wearing a {style} {color} {garment}, "
                            f"full body, clean background, realistic lighting, detailed fabric. "
                            f"Same person, same face, wearing the described outfit, preserve identity, realistic."
                        )
                        
                        result = inpaint_pipeline(
                            prompt=prompt,
                            image=img,
                            mask_image=mask,
                            num_inference_steps=2,
                            guidance_scale=0.0,
                        ).images[0]
                        buf = io.BytesIO()
                        result.save(buf, format="JPEG")
                        image_bytes = buf.getvalue()
                        print(f"[DEBUG] ✓ Local inpainting succeeded for {garment}")
                except Exception as e:
                    print(f"[WARNING] Local inpainting failed for {garment}: {e}")
                    image_bytes = None

            # ── Path 3: FALLBACK — Generate via HF API (FLUX.1-schnell) ──
            # This is the most reliable path; always available with a valid HF_TOKEN.
            if image_bytes is None:
                try:
                    print(f"[DEBUG] Falling back to HF API text-to-image for {garment}...")
                    image_bytes = await _txt2img(
                        client, gender, skin_tone, garment, color, style
                    )
                    print(f"[DEBUG] ✓ HF API generation succeeded for {garment}")
                except Exception as e:
                    print(f"[ERROR] HF API generation also failed for {garment}: {e}")
                    continue

            b64 = base64.b64encode(image_bytes).decode()
            results.append(f"data:image/jpeg;base64,{b64}")

    return results
