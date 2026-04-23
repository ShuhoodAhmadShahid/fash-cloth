# Quick Start Guide

## For Your System (8GB RAM, Core i5, No GPU)

This MVP is specifically optimized for your hardware constraints!

### Option 1: Use Docker (Easiest)

```bash
# 1. Install Docker Desktop (if not already installed)
# Download from: https://www.docker.com/products/docker-desktop/

# 2. Clone/navigate to the project
cd /workspace

# 3. Build and run
cd docker
docker-compose up --build

# 4. Open browser to http://localhost:5000
```

**First run will take 10-15 minutes** to download models (~2-3GB total):
- CLIP model: ~350MB
- Stable Diffusion: ~2GB
- Other dependencies

### Option 2: Local Python Installation

```bash
# 1. Navigate to backend
cd /workspace/backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies (this will take time)
pip install flask flask-cors
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow numpy mediapipe diffusers accelerate faiss-cpu scipy requests

# 5. Run the application
python app.py

# 6. Open browser to http://localhost:5000
```

### Important Notes for Low-RAM Systems

1. **Stable Diffusion is disabled by default** - It requires ~4GB RAM just to load. The app will work with mock generation until you have more RAM available.

2. **CLIP Model loads first** - This is the core retrieval engine (~350MB).

3. **Use CPU mode** - All models are configured for CPU inference.

4. **Image size limited to 512x512** - Reduces memory usage.

5. **Batch size = 1** - Process one image at a time.

### Expected Performance on Your System

| Operation | Time (CPU) | RAM Usage |
|-----------|-----------|-----------|
| App Startup | 30-60s | ~500MB |
| Image Analysis | 2-5s | ~800MB |
| Outfit Generation* | N/A (mock) | ~200MB |
| Product Retrieval | <1s | ~600MB |

*Real generation requires loading SD model (+2GB RAM)

### Testing Without Full Models

For initial testing, the app works in "lite mode":
- ✅ Skin tone analysis (color-based)
- ✅ Body proportion extraction (MediaPipe)
- ✅ Color recommendations
- ⚠️ Outfit generation (mock placeholder)
- ✅ Product retrieval (sample catalogue)

### Deploying to Free Cloud Tier

#### AWS Free Tier (12 months)

```bash
# 1. Launch EC2 t2.micro instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu

# 4. Clone/upload your code
# 5. Run with docker-compose
docker-compose up -d

# 6. Open http://your-instance-ip:5000
```

#### Azure Free Tier

```bash
# Use Azure Container Instances (ACI)
# 1. Build and push to container registry
docker build -t your-registry/fashion-ai .
docker push your-registry/fashion-ai

# 2. Deploy to ACI
az container create \
  --resource-group MyResourceGroup \
  --name fashion-ai \
  --image your-registry/fashion-ai \
  --cpu 1 \
  --memory 2 \
  --ports 5000 \
  --ip-address Public
```

#### Google Cloud Free Tier

```bash
# Use Cloud Run (serverless)
# 1. Build container
gcloud builds submit --tag gcr.io/PROJECT-ID/fashion-ai

# 2. Deploy to Cloud Run
gcloud run deploy fashion-ai \
  --image gcr.io/PROJECT-ID/fashion-ai \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated
```

### Troubleshooting

**"Out of Memory" Error:**
```bash
# Reduce image size in app.py
image.thumbnail((256, 256))  # Instead of 512x512
```

**Slow Performance:**
```bash
# Disable unnecessary models in app.py
# Comment out: load_stable_diffusion()
```

**Model Download Fails:**
```bash
# Manually download models
huggingface-cli download openai/clip-vit-base-patch32
huggingface-cli download runwayml/stable-diffusion-v1-5
```

### Next Steps After MVP Works

1. **Add Real Product Data**: Replace mock catalogue with actual e-commerce API
2. **Fine-tune Models**: When you get GPU access, fine-tune ResNet-50 on skin tone dataset
3. **Add Authentication**: User accounts and history
4. **Mobile App**: React Native version
5. **Monetization**: Affiliate links to real products

### Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Health check: `curl http://localhost:5000/health`
3. Test endpoints with Postman/curl

Enjoy building your AI Fashion Stylist! 🎨👗
