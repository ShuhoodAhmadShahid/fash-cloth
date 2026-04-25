import numpy as np
from PIL import Image

_gender_model = None


def _get_gender_model():
    global _gender_model
    if _gender_model is None:
        try:
            from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
            model_id = "rizvandwiki/gender-classification"
            print(f"[DEBUG] Attempting to load gender model: {model_id}")

            # Try to load locally first, then download if allowed
            try:
                processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
            except Exception:
                print(f"[DEBUG] Local model not found, attempting download for {model_id}...")
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)

            _gender_model = pipeline(
                "image-classification", 
                model=model, 
                image_processor=processor, 
                device=-1 # Force CPU
            )
        except Exception as e:
            print(f"[ERROR] Failed to load gender model: {e}")
            return None
    return _gender_model


def _detect_face_mediapipe(image_np):
    try:
        import mediapipe as mp
        mp_fd = mp.solutions.face_detection
        h, w = image_np.shape[:2]
        # model_selection=1 is better for full-body or medium-range photos
        with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.4) as detector:
            results = detector.process(image_np)
        if not results.detections:
            # Try again with model_selection=0 as fallback
            with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.3) as detector:
                results = detector.process(image_np)

        if not results or not results.detections:
            return None

        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width) * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "ratio": bb.width / max(bb.height, 0.001)}
    except Exception:
        return None


def _detect_face_opencv(image_np):
    try:
        import cv2
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "width": w, "height": h, "ratio": w / max(h, 1)}
    except Exception:
        return None


def _detect_face(image_np):
    face = _detect_face_mediapipe(image_np)
    if face is None:
        face = _detect_face_opencv(image_np)
    return face


def _detect_gender(face_pil: Image.Image) -> str:
    model = _get_gender_model()
    if model is None:
        return "unknown"
    results = model(face_pil)
    label = results[0]["label"].lower()
    return "male" if ("male" in label and "female" not in label) or "man" in label else "female"


def _detect_skin_tone(face_np: np.ndarray) -> str:
    pixels = face_np.astype(np.float32).reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)
    brightness = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]
    if brightness > 200:
        return "light"
    elif brightness > 140:
        return "medium"
    return "dark"


def _detect_face_shape(bbox: dict) -> str:
    ratio = bbox["ratio"]
    if 0.95 <= ratio <= 1.05:
        return "round"
    elif ratio > 1.2:
        return "wide"
    elif ratio < 0.9:
        return "long"
    return "oval"


def create_clothing_mask(image: Image.Image) -> Image.Image:
    """Generate a mask for the clothing area (person minus face)."""
    try:
        import cv2
        import mediapipe as mp
        
        # Robust mediapipe solutions import
        try:
            from mediapipe.python.solutions import selfie_segmentation as mp_selfie
        except ImportError:
            try:
                import mediapipe.solutions.selfie_segmentation as mp_selfie
            except ImportError:
                # Last resort
                mp_selfie = mp.solutions.selfie_segmentation
        
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        
        # 1. Get person mask using Selfie Segmentation
        with mp_selfie.SelfieSegmentation(model_selection=0) as segmenter:
            results = segmenter.process(image_np)
            person_mask = (results.segmentation_mask > 0.4).astype(np.uint8) * 255
            
        # 2. Get face mask
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face = _detect_face(image_np)
        if face:
            pad_x = int(face["width"] * 0.15)
            pad_y = int(face["height"] * 0.25)
            x1 = max(0, face["x1"] - pad_x)
            y1 = max(0, face["y1"] - pad_y)
            x2 = min(w, face["x2"] + pad_x)
            y2 = min(h, face["y2"] + int(pad_y * 1.8))
            cv2.rectangle(face_mask, (x1, y1), (x2, y2), 255, -1)
            
        # 3. Clothing mask = Person minus Face
        mask_np = cv2.subtract(person_mask, face_mask)
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        
        return Image.fromarray(mask_np)
    except Exception as e:
        print(f"[ERROR] Mask generation failed: {e}")
        # Return a smart fallback mask (central torso area)
        h, w = (512, 512)
        mask_fallback = np.zeros((h, w), dtype=np.uint8)
        # Assuming person is centered
        mask_fallback[int(h*0.3):int(h*0.9), int(w*0.2):int(w*0.8)] = 255
        return Image.fromarray(mask_fallback)


def analyze_image(image: Image.Image) -> dict:
    # Use higher resolution for better detection
    image_resized = image.resize((512, 512), Image.LANCZOS)
    image_np = np.array(image_resized)

    bbox = _detect_face(image_np)

    # Fallback if no face detected: assume centered head/upper body
    if bbox is None:
        print("[WARNING] No face detected, using fallback region")
        h, w = image_np.shape[:2]
        bbox = {
            "x1": int(w * 0.3), "y1": int(h * 0.1),
            "x2": int(w * 0.7), "y2": int(h * 0.4),
            "width": int(w * 0.4), "height": int(h * 0.3),
            "ratio": 1.0,
            "fallback": True
        }

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # Crop face with some padding for analysis
    pad_w = int(bbox["width"] * 0.1)
    pad_h = int(bbox["height"] * 0.1)

    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(image_np.shape[1], x2 + pad_w)
    crop_y2 = min(image_np.shape[0], y2 + pad_h)

    face_np = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
    if face_np.size == 0: # Safety check
        face_np = image_np[int(image_np.shape[0]*0.1):int(image_np.shape[0]*0.4),
                           int(image_np.shape[1]*0.3):int(image_np.shape[1]*0.7)]

    face_pil = Image.fromarray(face_np)

    try:
        gender = _detect_gender(face_pil)
    except Exception:
        gender = "unknown"

    skin_tone = _detect_skin_tone(face_np)
    face_shape = _detect_face_shape(bbox)

    return {"gender": gender, "skin_tone": skin_tone, "face_shape": face_shape}
