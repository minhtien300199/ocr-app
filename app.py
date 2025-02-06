from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_restx import Api, Resource, fields
import easyocr
import numpy as np
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Create API with url_prefix to avoid conflict with the default route
api = Api(app, version='1.0', 
    title='OCR API',
    description='A simple OCR API with support for Vietnamese and English',
    doc='/swagger',
    prefix='/api'
)

# Create namespaces - update the namespace to remove 'api' prefix since we added it above
ns_ocr = api.namespace('', description='OCR operations')

# Initialize the EasyOCR reader (only need to do this once)
reader = easyocr.Reader(['vi','en'])

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

def process_image(image_data):
    """Process image data and return detected text"""
    try:
        # Convert image data to format required by easyocr
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 encoded image
            image_data = image_data.split('base64,')[1]
            image_data = base64.b64decode(image_data)
        
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        # Convert to numpy array
        image_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(image_array)
        
        # Format results
        formatted_results = []
        for (bbox, text, prob) in results:
            formatted_results.append({
                'text': text,
                'confidence': float(prob)
            })
        
        return formatted_results
    
    except Exception as e:
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
    @ns_ocr.expect(api.parser().add_argument('file', 
        location='files',
        type='file',
        required=True,
        help='Image file to process'))
    def post(self):
        try:
            if 'file' not in request.files and 'image' not in request.json:
                return {'error': 'No image provided'}, 400

            if 'file' in request.files:
                image_file = request.files['file']
                image_data = image_file.read()
            else:
                image_data = request.json['image']

            results = process_image(image_data)
            return {'results': results}

        except Exception as e:
            return {'error': str(e)}, 500

@ns_ocr.route('/selftrained')
class SelfTrainedOCR(Resource):
    """Endpoint for self-trained model OCR"""
    @ns_ocr.doc('selftrained_ocr',
        responses={
            501: ('Not Implemented', error_model)
        })
    def post(self):
        """Self-trained model endpoint (to be implemented)"""
        # Return empty results array instead of message
        return {'results': []}, 501

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True) 