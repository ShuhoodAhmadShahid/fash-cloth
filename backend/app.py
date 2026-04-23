from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import base64
import io
import os
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
from diffusers import StableDiffusionPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms
import faiss
import pickle

app = Flask(__name__)
CORS(app)

# Configuration
DEVICE = "cpu"  # Use CPU for systems without GPU
MAX_RAM_GB = 8
SKIN_TONE_CLASSES = 10  # Monk Skin Tone scale (1-10)

# Global models (lazy loaded)
clip_model = None
clip_processor = None
sd_pipeline = None
pose_estimator = None
body_analysis_model = None
faiss_index = None
product_catalogue = []

print(f"[INFO] Running on {DEVICE} - Optimized for {MAX_RAM_GB}GB RAM")

def load_clip_model():
    """Load CLIP model for retrieval"""
    global clip_model, clip_processor
    print("[INFO] Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("[INFO] CLIP model loaded")

def load_stable_diffusion():
    """Load Stable Diffusion with ControlNet for outfit generation"""
    global sd_pipeline
    print("[INFO] Loading Stable Diffusion pipeline (this may take a while)...")
    
    # Use smaller model for low RAM systems
    try:
        # Load base SD model
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
            safety_checker=None,  # Disable for CPU to save memory
            requires_safety_checker=False
        )
        sd_pipeline = sd_pipeline.to(DEVICE)
        sd_pipeline.scheduler = UniPCMultistepScheduler.from_config(sd_pipeline.scheduler.config)
        
        # Reduce memory usage
        if DEVICE == "cpu":
            sd_pipeline.enable_attention_slicing()
        
        print("[INFO] Stable Diffusion loaded")
    except Exception as e:
        print(f"[WARNING] SD loading failed: {e}. Using mock generation.")
        sd_pipeline = None

def load_pose_estimator():
    """Load MediaPipe pose estimator"""
    global pose_estimator
    print("[INFO] Loading MediaPipe Pose...")
    pose_estimator = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,  # Lite model for low RAM
        min_detection_confidence=0.5
    )
    print("[INFO] MediaPipe Pose loaded")

def initialize_body_analysis_model():
    """Initialize simple body analysis (mock for MVP - can be fine-tuned later)"""
    global body_analysis_model
    print("[INFO] Initializing body analysis model...")
    # For MVP, we use a simple heuristic approach
    # In production, this would be a fine-tuned ResNet-50
    body_analysis_model = {"initialized": True}
    print("[INFO] Body analysis model ready")

def initialize_faiss_index():
    """Initialize FAISS index for product retrieval"""
    global faiss_index, product_catalogue
    print("[INFO] Initializing FAISS index...")
    
    # Create sample product catalogue (in production, load from database)
    product_catalogue = [
        {"id": 1, "name": "Blue Kurta", "category": "traditional", "image_url": "kurta_blue.jpg", "description": "Traditional blue kurta with embroidery"},
        {"id": 2, "name": "White Shalwar Kameez", "category": "traditional", "image_url": "shalwar_white.jpg", "description": "Classic white shalwar kameez"},
        {"id": 3, "name": "Red Sherwani", "category": "formal", "image_url": "sherwani_red.jpg", "description": "Elegant red sherwani for weddings"},
        {"id": 4, "name": "Green Kurti", "category": "casual", "image_url": "kurti_green.jpg", "description": "Casual green kurti"},
        {"id": 5, "name": "Black Waistcoat", "category": "formal", "image_url": "waistcoat_black.jpg", "description": "Formal black waistcoat"},
    ]
    
    # Initialize FAISS index
    embedding_dim = 512  # CLIP embedding dimension
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    
    # Generate embeddings for products (mock images for MVP)
    # In production, you'd encode actual product images
    print(f"[INFO] FAISS index initialized with {len(product_catalogue)} products")

def analyze_skin_tone(image):
    """Analyze skin tone from image using simple color analysis"""
    # Convert to numpy
    img_np = np.array(image)
    
    # Simple skin detection based on color ranges
    # In production, use the trained ResNet-50 classifier
    mask = ((img_np[:,:,0] > 95) & (img_np[:,:,0] < 200) &
            (img_np[:,:,1] > 50) & (img_np[:,:,1] < 180) &
            (img_np[:,:,2] > 50) & (img_np[:,:,2] < 150))
    
    if np.sum(mask) == 0:
        return 5, "Medium"  # Default
    
    # Calculate average skin color
    skin_pixels = img_np[mask]
    avg_brightness = np.mean(skin_pixels)
    
    # Map to Monk Skin Tone scale (1-10)
    if avg_brightness > 200:
        return 1, "Very Light"
    elif avg_brightness > 180:
        return 2, "Light"
    elif avg_brightness > 160:
        return 3, "Light-Medium"
    elif avg_brightness > 140:
        return 4, "Medium-Light"
    elif avg_brightness > 120:
        return 5, "Medium"
    elif avg_brightness > 100:
        return 6, "Medium-Dark"
    elif avg_brightness > 80:
        return 7, "Dark-Medium"
    elif avg_brightness > 60:
        return 8, "Dark"
    elif avg_brightness > 40:
        return 9, "Very Dark"
    else:
        return 10, "Deepest"

def extract_body_proportions(image):
    """Extract body proportions using MediaPipe"""
    if pose_estimator is None:
        return {"shoulder_width": 0.3, "torso_length": 0.4, "aspect_ratio": 0.6}
    
    img_np = np.array(image)
    results = pose_estimator.process(img_np)
    
    if results.pose_landmarks is None:
        # Return default proportions
        return {"shoulder_width": 0.3, "torso_length": 0.4, "aspect_ratio": 0.6}
    
    landmarks = results.pose_landmarks.landmark
    
    # Extract key proportions
    h, w, _ = img_np.shape
    
    # Shoulder width (landmarks 11 and 12)
    shoulder_dist = abs(landmarks[11].x - landmarks[12].x)
    
    # Torso length (shoulders to hips)
    torso_dist = abs((landmarks[11].y + landmarks[12].y)/2 - (landmarks[23].y + landmarks[24].y)/2)
    
    # Aspect ratio
    aspect_ratio = shoulder_dist / torso_dist if torso_dist > 0 else 0.6
    
    return {
        "shoulder_width": float(shoulder_dist),
        "torso_length": float(torso_dist),
        "aspect_ratio": float(aspect_ratio)
    }

def generate_outfit_prompt(skin_tone, body_props, style="traditional"):
    """Generate prompt for outfit generation"""
    skin_descriptions = {
        1: "very light skin tone",
        2: "light skin tone",
        3: "light-medium skin tone",
        4: "medium-light skin tone",
        5: "medium skin tone",
        6: "medium-dark skin tone",
        7: "dark-medium skin tone",
        8: "dark skin tone",
        9: "very dark skin tone",
        10: "deepest skin tone"
    }
    
    body_type = "slim" if body_props["aspect_ratio"] < 0.5 else "average" if body_props["aspect_ratio"] < 0.7 else "broad"
    
    prompts = {
        "traditional": f"professional photo of person with {skin_descriptions.get(skin_tone, 'medium skin tone')} wearing elegant traditional South Asian outfit, {body_type} build, high quality, detailed, fashion photography",
        "casual": f"casual photo of person with {skin_descriptions.get(skin_tone, 'medium skin tone')} wearing comfortable modern outfit, {body_type} build, natural lighting, high quality",
        "formal": f"formal photo of person with {skin_descriptions.get(skin_tone, 'medium skin tone')} wearing elegant formal wear, {body_type} build, studio lighting, professional, high quality"
    }
    
    return prompts.get(style, prompts["traditional"])

def get_color_recommendations(skin_tone):
    """Get color recommendations based on skin tone"""
    # Color recommendations based on Monk Skin Tone scale
    warm_colors = ["gold", "orange", "yellow", "coral", "peach"]
    cool_colors = ["blue", "purple", "emerald", "pink", "silver"]
    neutral_colors = ["white", "black", "gray", "beige", "navy"]
    
    if skin_tone <= 3:
        recommended = cool_colors + neutral_colors
    elif skin_tone <= 7:
        recommended = warm_colors + neutral_colors
    else:
        recommended = warm_colors + cool_colors
    
    return recommended[:5]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": DEVICE})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for body and skin tone"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Resize for processing
        image.thumbnail((512, 512))
        
        # Analyze skin tone
        skin_tone_id, skin_tone_desc = analyze_skin_tone(image)
        
        # Extract body proportions
        body_props = extract_body_proportions(image)
        
        # Get color recommendations
        colors = get_color_recommendations(skin_tone_id)
        
        result = {
            "skin_tone": {
                "id": skin_tone_id,
                "description": skin_tone_desc
            },
            "body_proportions": body_props,
            "recommended_colors": colors,
            "message": "Analysis complete. You can now generate outfit recommendations."
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_outfit():
    """Generate outfit recommendation"""
    try:
        data = request.json
        
        skin_tone = data.get('skin_tone', 5)
        style = data.get('style', 'traditional')
        body_props = data.get('body_proportions', {"shoulder_width": 0.3, "torso_length": 0.4, "aspect_ratio": 0.6})
        
        # Generate prompt
        prompt = generate_outfit_prompt(skin_tone, body_props, style)
        
        if sd_pipeline is None:
            # Mock generation if SD not loaded
            return jsonify({
                "success": True,
                "message": "Stable Diffusion not loaded. In production, this would generate an outfit image.",
                "prompt": prompt,
                "mock_image": "https://via.placeholder.com/512x512?text=Outfit+Generation"
            })
        
        # Generate image
        generator = torch.Generator(device=DEVICE).manual_seed(42) if DEVICE == "cpu" else None
        
        output = sd_pipeline(
            prompt=prompt,
            num_inference_steps=20,  # Reduced for speed
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512
        )
        
        generated_image = output.images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        generated_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "skin_tone": skin_tone,
            "style": style
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrieve', methods=['POST'])
def retrieve_products():
    """Retrieve similar products using CLIP"""
    try:
        data = request.json
        
        if 'image' not in data and 'prompt' not in data:
            return jsonify({"error": "No image or prompt provided"}), 400
        
        # Use CLIP to find similar products
        if clip_model is None:
            # Mock retrieval
            return jsonify({
                "products": product_catalogue[:3],
                "message": "CLIP not loaded. Returning sample products."
            })
        
        # Encode query
        if 'image' in data:
            # Decode image
            img_data = base64.b64decode(data['image'].split(',')[1])
            query_image = Image.open(io.BytesIO(img_data)).convert('RGB')
            inputs = clip_processor(images=query_image, return_tensors="pt")
        else:
            # Use text prompt
            inputs = clip_processor(text=[data['prompt']], return_tensors="pt", padding=True)
        
        # Get embedding
        with torch.no_grad():
            if 'image' in data:
                query_embedding = clip_model.get_image_features(**inputs)
            else:
                query_embedding = clip_model.get_text_features(**inputs)
        
        query_embedding = query_embedding.cpu().numpy()
        
        # Search in FAISS index (mock for MVP)
        # In production, you'd have actual product embeddings
        results = []
        for i, product in enumerate(product_catalogue):
            results.append({
                **product,
                "similarity_score": float(1.0 / (1.0 + i * 0.1))  # Mock score
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            "products": results[:5],  # Top 5
            "count": len(results)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template_string(HTML_TEMPLATE)

# HTML Template for Frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion Recommendation System</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>👗 AI Fashion Stylist</h1>
            <p>Personalized outfit recommendations using AI</p>
        </header>

        <main>
            <section class="upload-section">
                <h2>Step 1: Upload Your Photo</h2>
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="imageInput" accept="image/*" hidden>
                    <div class="upload-placeholder">
                        <span>📷</span>
                        <p>Click or drag to upload</p>
                    </div>
                    <img id="preview" class="preview" style="display:none;">
                </div>
                <button id="analyzeBtn" class="btn primary" disabled>Analyze Photo</button>
            </section>

            <section class="analysis-section" id="analysisSection" style="display:none;">
                <h2>Step 2: Your Analysis</h2>
                <div class="results-grid">
                    <div class="result-card">
                        <h3>Skin Tone</h3>
                        <p id="skinToneResult">-</p>
                    </div>
                    <div class="result-card">
                        <h3>Body Type</h3>
                        <p id="bodyTypeResult">-</p>
                    </div>
                    <div class="result-card">
                        <h3>Recommended Colors</h3>
                        <div id="colorRecommendations"></div>
                    </div>
                </div>
            </section>

            <section class="generate-section" id="generateSection" style="display:none;">
                <h2>Step 3: Generate Outfit</h2>
                <div class="style-selector">
                    <label>Style:</label>
                    <select id="styleSelect">
                        <option value="traditional">Traditional (Kurta/Shalwar)</option>
                        <option value="casual">Casual</option>
                        <option value="formal">Formal</option>
                    </select>
                </div>
                <button id="generateBtn" class="btn primary">Generate Outfit</button>
                
                <div class="generated-result" id="generatedResult" style="display:none;">
                    <img id="generatedImage" alt="Generated Outfit">
                </div>
            </section>

            <section class="products-section" id="productsSection" style="display:none;">
                <h2>Step 4: Find Similar Products</h2>
                <button id="retrieveBtn" class="btn primary">Find Products</button>
                
                <div class="products-grid" id="productsGrid"></div>
            </section>

            <div class="loading" id="loading" style="display:none;">
                <div class="spinner"></div>
                <p id="loadingText">Processing...</p>
            </div>
        </main>

        <footer>
            <p>AI Fashion Recommendation System | Built with Deep Learning</p>
        </footer>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
'''

if __name__ == '__main__':
    print("[INFO] Starting Fashion Recommendation System...")
    
    # Lazy load models on startup
    load_clip_model()
    load_pose_estimator()
    initialize_body_analysis_model()
    initialize_faiss_index()
    
    # Don't load SD by default to save RAM
    # load_stable_diffusion()
    
    print("[INFO] Server starting on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
