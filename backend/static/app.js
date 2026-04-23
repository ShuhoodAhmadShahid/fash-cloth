// Fashion Recommendation System - Frontend JavaScript

const API_BASE = '';

// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const analysisSection = document.getElementById('analysisSection');
const generateSection = document.getElementById('generateSection');
const productsSection = document.getElementById('productsSection');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');

// State
let currentImage = null;
let analysisResult = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Generate button
    document.getElementById('generateBtn').addEventListener('click', generateOutfit);
    
    // Retrieve button
    document.getElementById('retrieveBtn').addEventListener('click', retrieveProducts);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = e.target.result;
        preview.src = currentImage;
        preview.style.display = 'block';
        document.querySelector('.upload-placeholder').style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!currentImage) return;
    
    showLoading('Analyzing your photo...');
    
    try {
        // Convert base64 to blob
        const blob = await fetch(currentImage).then(r => r.blob());
        
        const formData = new FormData();
        formData.append('image', blob, 'upload.jpg');
        
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        analysisResult = await response.json();
        
        displayAnalysisResults();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze image. Please try again.');
    } finally {
        hideLoading();
    }
}

function displayAnalysisResults() {
    if (!analysisResult) return;
    
    // Update skin tone
    document.getElementById('skinToneResult').textContent = 
        `${analysisResult.skin_tone.description} (MST ${analysisResult.skin_tone.id})`;
    
    // Update body type
    const props = analysisResult.body_proportions;
    const bodyType = props.aspect_ratio < 0.5 ? 'Slim' : 
                     props.aspect_ratio < 0.7 ? 'Average' : 'Broad';
    document.getElementById('bodyTypeResult').textContent = bodyType;
    
    // Update color recommendations
    const colorsDiv = document.getElementById('colorRecommendations');
    colorsDiv.innerHTML = '';
    analysisResult.recommended_colors.forEach(color => {
        const colorSpan = document.createElement('span');
        colorSpan.className = 'color-chip';
        colorSpan.textContent = color;
        colorSpan.style.backgroundColor = color;
        colorsDiv.appendChild(colorSpan);
    });
    
    // Show sections
    analysisSection.style.display = 'block';
    generateSection.style.display = 'block';
}

async function generateOutfit() {
    if (!analysisResult) return;
    
    showLoading('Generating outfit recommendation...');
    
    try {
        const style = document.getElementById('styleSelect').value;
        
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                skin_tone: analysisResult.skin_tone.id,
                style: style,
                body_proportions: analysisResult.body_proportions
            })
        });
        
        if (!response.ok) throw new Error('Generation failed');
        
        const result = await response.json();
        
        if (result.image) {
            document.getElementById('generatedImage').src = result.image;
            document.getElementById('generatedResult').style.display = 'block';
        } else if (result.mock_image) {
            document.getElementById('generatedImage').src = result.mock_image;
            document.getElementById('generatedResult').style.display = 'block';
        }
        
        // Show products section
        productsSection.style.display = 'block';
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate outfit. Please try again.');
    } finally {
        hideLoading();
    }
}

async function retrieveProducts() {
    showLoading('Finding similar products...');
    
    try {
        const style = document.getElementById('styleSelect').value;
        const prompt = `traditional South Asian ${style} outfit`;
        
        const response = await fetch(`${API_BASE}/retrieve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt
            })
        });
        
        if (!response.ok) throw new Error('Retrieval failed');
        
        const result = await response.json();
        
        displayProducts(result.products);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to retrieve products. Please try again.');
    } finally {
        hideLoading();
    }
}

function displayProducts(products) {
    const grid = document.getElementById('productsGrid');
    grid.innerHTML = '';
    
    products.forEach(product => {
        const card = document.createElement('div');
        card.className = 'product-card';
        card.innerHTML = `
            <div class="product-image">
                <img src="https://via.placeholder.com/200x200?text=${encodeURIComponent(product.name)}" alt="${product.name}">
            </div>
            <div class="product-info">
                <h4>${product.name}</h4>
                <p class="product-category">${product.category}</p>
                <p class="product-description">${product.description}</p>
                <div class="similarity-score">Match: ${(product.similarity_score * 100).toFixed(1)}%</div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function showLoading(text) {
    loadingText.textContent = text;
    loading.style.display = 'flex';
}

function hideLoading() {
    loading.style.display = 'none';
}
