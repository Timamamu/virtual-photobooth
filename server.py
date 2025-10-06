import io
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
from pix2pix_model import Pix2PixModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("style-server")
logger.info("Starting style server application")

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Model initialization
try:
    MODEL_FOLDER = 'models/S4_MODEL'
    logger.info(f"Attempting to load model from {MODEL_FOLDER}")
    
    # Check if model folder exists
    if not os.path.exists(MODEL_FOLDER):
        logger.error(f"Model folder not found: {MODEL_FOLDER}")
        raise FileNotFoundError(f"Model folder not found: {MODEL_FOLDER}")
    
    # Load the model
    model = Pix2PixModel.createFromFolder(MODEL_FOLDER, DEVICE)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    return "Style Server is running. Make a POST request to /style to use the styling API."

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'Server is running!',
        'device': str(DEVICE),
        'model_folder': MODEL_FOLDER
    })

def preprocess_image(base64_str):
    try:
        logger.info(f"Preprocessing image data (length: {len(base64_str)})")
        # Handle both with and without data URL prefix
        if "," in base64_str:
            prefix = base64_str.split(",")[0]
            logger.info(f"Image prefix: {prefix}")
            img_data = base64.b64decode(base64_str.split(",")[1])
        else:
            img_data = base64.b64decode(base64_str)
            
        logger.info(f"Decoded base64 data (length: {len(img_data)})")
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        logger.info(f"Image opened successfully, size: {img.size}")
        
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        tensor = transform(img).unsqueeze(0)
        logger.info(f"Image transformed to tensor: {tensor.shape}")
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def postprocess_image(tensor):
    try:
        logger.info(f"Postprocessing tensor: {tensor.shape}")
        tensor = (tensor.squeeze(0).clamp(0, 1) * 255).byte()
        img = transforms.ToPILImage()(tensor)
        logger.info(f"Final image size: {img.size}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        result = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        logger.info(f"Encoded result length: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error postprocessing image: {e}")
        raise

@app.route('/style', methods=['POST'])
def style():
    logger.info("Received styling request")
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("No image data found in request")
            return jsonify({'error': 'No image data provided'}), 400
            
        # Log image data length
        logger.info(f"Received image data length: {len(data['image'])}")
        
        # Process the image
        img_tensor = preprocess_image(data['image'])
        input_np = (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        input_np = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
        logger.info(f"Converted to numpy array: {input_np.shape}")
        
        with torch.no_grad():
            logger.info("Applying model...")
            converted_bgr = model.applyToImage(input_np)
            logger.info(f"Model output shape: {converted_bgr.shape}")
            
            converted_rgb = cv2.cvtColor(converted_bgr, cv2.COLOR_BGR2RGB)
            converted_pil = Image.fromarray(converted_rgb)
            logger.info(f"Converted to PIL image: {converted_pil.size}")
            
            tensor = transforms.ToTensor()(converted_pil).unsqueeze(0)
            styled_image = postprocess_image(tensor)
            
        logger.info(f"Styled image data length: {len(styled_image)}")
        return jsonify({'image': styled_image})
        
    except Exception as e:
        logger.error(f"Error processing style request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server with HTTPS on port 5000")
        # Use your existing SSL certificates
        app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'), debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise