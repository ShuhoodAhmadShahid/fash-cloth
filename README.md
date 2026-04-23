# AI Fashion Recommendation System

A deep learning-powered personalized fashion recommendation system that analyzes body proportions, skin tone, and generates outfit recommendations using pre-trained models from Hugging Face.

## Features

- **Body & Skin Analysis**: Uses MediaPipe for pose estimation and color analysis for Monk Skin Tone classification
- **Outfit Generation**: Leverages Stable Diffusion v1.5 for generating personalized outfit images
- **Product Retrieval**: CLIP-based vector search for finding similar products
- **Low Resource Optimized**: Designed to run on CPU with 8GB RAM (no GPU required)
- **Docker Ready**: Complete containerization for easy deployment

## Architecture

The system follows a three-stage pipeline:

1. **Analysis Stage**: 
   - MediaPipe Pose for body proportion extraction
   - Color-based skin tone classification (Monk Skin Tone scale 1-10)

2. **Generation Stage**:
   - Stable Diffusion v1.5 with prompt engineering
   - Style conditioning (traditional, casual, formal)

3. **Retrieval Stage**:
   - CLIP (ViT-B/32) for embedding generation
   - FAISS for similarity search

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Access the application at http://localhost:5000
```

### Option 2: Local Installation

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://localhost:5000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze uploaded image |
| `/generate` | POST | Generate outfit recommendation |
| `/retrieve` | POST | Find similar products |

## Usage

1. **Upload Photo**: Click or drag-and-drop a front-facing photo
2. **Analyze**: Get skin tone (MST scale) and body proportion analysis
3. **Generate**: Select style preference and generate outfit
4. **Retrieve**: Find similar products from catalogue

## Configuration

### Environment Variables

- `DEVICE`: "cpu" or "cuda" (default: "cpu")
- `MAX_RAM_GB`: Maximum RAM usage in GB (default: 8)

### Model Sources

All models are downloaded from Hugging Face:
- CLIP: `openai/clip-vit-base-patch32`
- Stable Diffusion: `runwayml/stable-diffusion-v1-5`

## Free Tier Deployment

### AWS Free Tier

```bash
# Use EC2 t2.micro or t3.micro instance
# 12 months free tier eligible

# Deploy with Docker
docker-compose up -d

# Or use AWS Elastic Beanstalk
eb init
eb create fashion-ai
```

### Azure Free Tier

```bash
# Use Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fashion-ai \
  --image fashion-ai:latest \
  --cpu 1 \
  --memory 2 \
  --ports 5000
```

## Project Structure

```
/workspace
├── backend/
│   ├── app.py              # Flask backend
│   ├── requirements.txt    # Python dependencies
│   └── static/
│       ├── style.css       # Compiled CSS
│       ├── style.scss      # SCSS source
│       └── app.js          # Frontend JavaScript
├── frontend/               # (Optional separate frontend)
└── docker/
    ├── Dockerfile          # Container configuration
    └── docker-compose.yml  # Multi-container setup
```

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, SCSS, Vanilla JavaScript
- **ML Models**: 
  - MediaPipe (Pose Estimation)
  - CLIP (OpenAI)
  - Stable Diffusion v1.5
- **Vector Search**: FAISS
- **Containerization**: Docker, Docker Compose

## Limitations (MVP)

- CPU-only inference (slower than GPU)
- Basic skin tone detection (can be improved with fine-tuned ResNet-50)
- Mock product catalogue (replace with real e-commerce API)
- Single image upload (batch processing not supported)

## Future Improvements

1. Fine-tune ResNet-50 on Monk Skin Tone dataset
2. Add ControlNet for better pose-conditioned generation
3. Integrate real e-commerce APIs (Daraz, local retailers)
4. Add user authentication and history
5. Mobile app with React Native

## License

MIT License - Free for academic and commercial use

## Credits

Based on research paper: "AI-Driven Personalized Fashion Recommendation System: A Deep Learning Approach Using Diffusion Models and CLIP Retrieval"

Authors: Shuhood Ahmad Shahid et al., PAFIAST Haripur
