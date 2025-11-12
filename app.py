import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import json
import os

# Page config
st.set_page_config(
    page_title="FoodVision AI",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.id2label = None
    st.session_state.device = None

@st.cache_resource
def load_model():
    """Load the trained ViT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Update this path to your Kaggle dataset path
    MODEL_PATH = '/kaggle/input/your-dataset-name/best_vit_food_model.pth'
    LABELS_PATH = '/kaggle/input/your-dataset-name/label_mappings.json'
    
    # For local testing, use relative paths
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'best_vit_food_model.pth'
        LABELS_PATH = 'label_mappings.json'
    
    # Load label mappings
    with open(LABELS_PATH, 'r') as f:
        label_data = json.load(f)
        id2label = {int(k): v for k, v in label_data['id2label'].items()}
        label2id = label_data['label2id']
    
    # Load model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=101,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, id2label, device

def get_image_transforms():
    """Get image preprocessing transforms"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def predict_food(image, model, id2label, device):
    """Make prediction on uploaded image"""
    transform = get_image_transforms()
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5_probs = top5_probs.cpu().numpy()[0]
        top5_indices = top5_indices.cpu().numpy()[0]
        
        predictions = [
            (id2label[idx], float(prob) * 100) 
            for idx, prob in zip(top5_indices, top5_probs)
        ]
    
    return predictions

# Navigation
st.markdown("""
<nav class="navbar">
    <div class="nav-container">
        <div class="nav-logo">🍕 FoodVision AI</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#classify">Classify</a>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<section class="hero" id="home">
    <div class="hero-content">
        <h1 class="hero-title">Identify Any Food<br/>with AI Precision</h1>
        <p class="hero-subtitle">Upload a photo and discover what's on your plate. Powered by Vision Transformer technology trained on 101 food categories.</p>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-number">101</div>
                <div class="stat-label">Food Categories</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">95%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">&lt;1s</div>
                <div class="stat-label">Response Time</div>
            </div>
        </div>
    </div>
    <div class="hero-image">
        <div class="food-grid">
            <div class="food-item">🍕</div>
            <div class="food-item">🍔</div>
            <div class="food-item">🍜</div>
            <div class="food-item">🍰</div>
            <div class="food-item">🥗</div>
            <div class="food-item">🍱</div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# Classification Section
# This hidden div acts as an anchor for our CSS to target the correct container
st.markdown('<div id="classify-section-start" style="display: none;"></div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<h2 class="section-title">Upload Your Food Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open( uploaded_file).convert('RGB')
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="section-title">Classification Results</h2>', unsafe_allow_html=True)
    
    if uploaded_file:
        with st.spinner('🔍 Analyzing your food...'):
            if st.session_state.model is None:
                st.session_state.model, st.session_state.id2label, st.session_state.device = load_model()
            
            predictions = predict_food(
                image,
                st.session_state.model,
                st.session_state.id2label,
                st.session_state.device
            )
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        for i, (food_name, confidence) in enumerate(predictions):
            if i == 0:
                st.markdown(f"""
                <div class="prediction-card primary">
                    <div class="prediction-header">
                        <span class="prediction-rank">#{i+1}</span>
                        <span class="prediction-name">{food_name}</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    <div class="prediction-confidence">{confidence:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-header">
                        <span class="prediction-rank">#{i+1}</span>
                        <span class="prediction-name">{food_name}</span>
                    </div>
                    <div class="confidence-bar secondary">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    <div class="prediction-confidence">{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">📸</div>
            <p>Upload an image to see the magic happen!</p>
        </div>
        """, unsafe_allow_html=True)

# About Section
st.markdown("""
<section class="about-section" id="about">
    <div class="about-container">
        <h2 class="section-title centered">About FoodVision AI</h2>
        <div class="about-grid">
            <div class="about-card">
                <div class="about-icon">🧠</div>
                <h3>Vision Transformer</h3>
                <p>Powered by Google's ViT architecture, fine-tuned on 101 diverse food categories for exceptional accuracy.</p>
            </div>
            <div class="about-card">
                <div class="about-icon">⚡</div>
                <h3>Lightning Fast</h3>
                <p>Get instant predictions in under a second with optimized model inference on GPU acceleration.</p>
            </div>
            <div class="about-card">
                <div class="about-icon">🎯</div>
                <h3>High Accuracy</h3>
                <p>Trained on thousands of images with extensive data augmentation for robust real-world performance.</p>
            </div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<footer class="footer">
    <p>Built with ❤️ using Streamlit and PyTorch | Vision Transformer Model</p>
</footer>
""", unsafe_allow_html=True)
