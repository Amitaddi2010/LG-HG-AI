from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load model
def create_efficientnet_model(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

model = create_efficientnet_model()
# Load model from local file
model_path = 'final_efficientnet_b0_cv.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("Model loaded successfully")
else:
    print("Warning: Model file not found, using random weights for demo")
model.eval()

# Exact preprocessing pipeline from transform_images224.py
def preprocess_image(pil_image):
    # Convert PIL to numpy array then to CV2
    img_array = np.array(pil_image)
    
    # Convert RGB to BGR for CV2 processing
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert BGR back to RGB (matching training pipeline)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL for transforms
    img_pil = transforms.ToPILImage()(img_rgb)
    
    # Apply exact same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img_pil)

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/docs', methods=['GET'])
def get_documentation():
    try:
        # Embedded classification report for deployment
        classification_report = '''              precision    recall  f1-score   support

LG (Class 0)       0.95      0.93      0.94       782
HG (Class 1)       0.93      0.95      0.94       782

    accuracy                           0.94      1564
   macro avg       0.94      0.94      0.94      1564
weighted avg       0.94      0.94      0.94      1564'''
        
        docs = {
            'model_info': {
                'architecture': 'EfficientNet-B0',
                'training_method': '5-Fold Cross-Validation',
                'image_size': '224x224',
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 0.0001,
                'optimizer': 'Adam',
                'loss_function': 'CrossEntropyLoss with class weights [1.0, 1.5]',
                'patience': 5
            },
            'performance': {
                'overall_accuracy': '94%',
                'lg_precision': '95%',
                'lg_recall': '93%',
                'lg_f1_score': '94%',
                'hg_precision': '93%',
                'hg_recall': '95%',
                'hg_f1_score': '94%',
                'classification_report': classification_report
            },
            'preprocessing': {
                'steps': [
                    'RGB to BGR conversion',
                    'BGR back to RGB (CV2 compatibility)',
                    'Resize to 224x224',
                    'Normalize with ImageNet statistics',
                    'Tensor conversion'
                ]
            }
        }
        
        return jsonify(docs)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/pipeline', methods=['POST'])
def pipeline():
    try:
        # Step 1: Input - Get image from request
        file = request.files['image']
        original_image = Image.open(file.stream).convert('RGB')
        
        # Step 2: Preprocessing - Use existing preprocessing pipeline
        preprocessed_tensor = preprocess_image(original_image).unsqueeze(0)
        
        # Step 3: Analysis - Model prediction
        with torch.no_grad():
            outputs = model(preprocessed_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Step 4: Output - Structured result
        result = {
            'pipeline': {
                'step1_input': {
                    'status': 'completed',
                    'image_size': f'{original_image.size[0]}x{original_image.size[1]}',
                    'format': original_image.format or 'Unknown'
                },
                'step2_preprocessing': {
                    'status': 'completed',
                    'rgb_to_bgr_to_rgb': True,
                    'resized_to': '224x224',
                    'normalized': 'ImageNet stats'
                },
                'step3_analysis': {
                    'status': 'completed',
                    'model': 'EfficientNet-B0',
                    'processed': True
                },
                'step4_output': {
                    'prediction': 'HG (High Grade)' if predicted_class == 1 else 'LG (Low Grade)',
                    'confidence': f'{confidence:.2%}',
                    'class': predicted_class
                }
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)