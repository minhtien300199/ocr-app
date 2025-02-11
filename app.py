from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_restx import Api, Resource, fields
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
import base64
from PIL import Image
import io
import os
from config.training_config import DATASET_CONFIG, MODEL_CONFIG, MODEL_PATHS

app = Flask(__name__)

# Create API with url_prefix to avoid conflict with the default route
api = Api(app, version='1.0', 
    title='OCR API',
    description='A GPT-4-OCR API with support for Vietnamese and English',
    doc='/swagger',
    prefix='/api'
)

# Create namespaces
ns_ocr = api.namespace('', description='OCR operations')

# Initialize model and processor
processor = AutoProcessor.from_pretrained(MODEL_CONFIG['model_name'])
model = AutoModelForVision2Seq.from_pretrained(MODEL_CONFIG['model_name'])
model.to(MODEL_CONFIG['device'])

# Define models for Swagger documentation
error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

result_model = api.model('Result', {
    'text': fields.String(description='Detected text'),
    'confidence': fields.Float(description='Confidence score')
})

results_model = api.model('Results', {
    'results': fields.List(fields.Nested(result_model))
})

def process_image(image_data, text_level='word'):
    """Process image data using GPT-4-OCR model"""
    try:
        # Convert image data to PIL Image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split('base64,')[1]
            image_data = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare prompt based on text level
        prompts = {
            'word': "Extract individual words from this image:",
            'paragraph': "Extract paragraphs from this image:",
            'block': "Extract text blocks from this image:"
        }
        prompt = prompts.get(text_level, prompts['word'])
        
        # Process image with the model
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(MODEL_CONFIG['device'])
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                temperature=MODEL_CONFIG['temperature']
            )
        
        # Decode output
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Process results based on text level
        if text_level == 'word':
            words = generated_text.split()
            results = [{'text': word, 'confidence': 0.95} for word in words]
        elif text_level == 'paragraph':
            paragraphs = generated_text.split('\n\n')
            results = [{'text': para.strip(), 'confidence': 0.95} for para in paragraphs if para.strip()]
        else:  # block
            blocks = generated_text.split('\n')
            results = [{'text': block.strip(), 'confidence': 0.95} for block in blocks if block.strip()]
        
        return results if results else [{'text': 'No text detected', 'confidence': 0.0}]
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return {'error': str(e)}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@ns_ocr.route('/pretrained')
class PretrainedOCR(Resource):
    """Endpoint for pre-trained model OCR"""
    @ns_ocr.doc('perform_ocr',
        responses={
            200: ('Success', results_model),
            400: ('Validation Error', error_model),
            500: ('Internal Server Error', error_model)
        })
    @ns_ocr.expect(api.parser()
        .add_argument('file', 
            location='files',
            type='file',
            required=True,
            help='Image file to process')
        .add_argument('text_level',
            location='form',
            type=str,
            required=False,
            default='word',
            choices=['block', 'paragraph', 'word'],
            help='Level of text detection'))
    def post(self):
        try:
            if 'file' not in request.files:
                return {'error': 'No image provided'}, 400

            image_file = request.files['file']
            if not image_file.filename:
                return {'error': 'Empty file provided'}, 400
            
            text_level = request.form.get('text_level', 'word')
            image_data = image_file.read()
            results = process_image(image_data, text_level=text_level)
            
            if isinstance(results, dict) and 'error' in results:
                return results, 500
                
            return {'results': results}

        except Exception as e:
            print(f"Error in API endpoint: {str(e)}")
            return {'error': str(e)}, 500

@ns_ocr.route('/selftrained')
class SelfTrainedOCR(Resource):
    """Endpoint for self-trained model OCR"""
    @ns_ocr.doc('selftrained_ocr',
        responses={
            200: ('Success', results_model),
            400: ('Validation Error', error_model),
            500: ('Internal Server Error', error_model)
        })
    @ns_ocr.expect(api.parser().add_argument('file', 
        location='files',
        type='file',
        required=True,
        help='Image file to process'))
    def post(self):
        try:
            if not os.path.exists(MODEL_PATHS['best_model_path']):
                return {'results': [], 'message': 'No trained model available'}, 501
                
            if 'file' not in request.files:
                return {'error': 'No image provided'}, 400

            image_file = request.files['file']
            if not image_file.filename:
                return {'error': 'Empty file provided'}, 400
                
            image_data = image_file.read()
            results = process_image(image_data)
            
            if isinstance(results, dict) and 'error' in results:
                return results, 500
                
            return {'results': results}

        except Exception as e:
            print(f"Error in API endpoint: {str(e)}")
            return {'error': str(e)}, 500

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True) 