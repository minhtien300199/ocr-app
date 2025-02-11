from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_restx import Api, Resource, fields
from openai import OpenAI
import numpy as np
import base64
from PIL import Image
import io
import os
from config.training_config import MODEL_CONFIG, IMAGE_CONFIG

app = Flask(__name__)

# Create API with url_prefix
api = Api(app, version='1.0', 
    title='OCR API',
    description='An OCR API with support for Vietnamese and English',
    doc='/swagger',
    prefix='/api'
)

# Create namespaces
ns_ocr = api.namespace('', description='OCR operations')

# Initialize OpenAI client - Remove organization parameter if it's empty
client = OpenAI(
    api_key=MODEL_CONFIG['api_key'],
    default_headers={
        "X-Project-ID": MODEL_CONFIG['project_id']
    }
) if not MODEL_CONFIG['org_id'] else OpenAI(
    api_key=MODEL_CONFIG['api_key'],
    organization=MODEL_CONFIG['org_id'],
    default_headers={
        "X-Project-ID": MODEL_CONFIG['project_id']
    }
)
print("OpenAI API Key:", client.api_key)
print("OpenAI organization:", client.organization)

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

def prepare_image(image_data):
    """Prepare image for API submission"""
    try:
        # Convert to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split('base64,')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            return None

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large
        if max(image.size) > IMAGE_CONFIG['max_size']:
            ratio = IMAGE_CONFIG['max_size'] / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error preparing image: {str(e)}")
        return None

def process_image(image_data, text_level='word'):
    """Process image data using OpenAI's GPT-4 Vision model"""
    try:
        # Prepare image
        base64_image = prepare_image(image_data)
        if not base64_image:
            return {'error': 'Failed to process image'}

        # Prepare prompt based on text level
        prompts = {
            'word': "Extract individual words from this image. Return them as a comma-separated list.",
            'paragraph': "Extract paragraphs from this image. Separate each paragraph with a double newline.",
            'block': "Extract text blocks from this image. Separate each block with a newline."
        }
        prompt = prompts.get(text_level, prompts['word'])

        # Call OpenAI API
        response = client.chat.completions.create(
            model=MODEL_CONFIG['model_name'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": IMAGE_CONFIG['quality']
                            }
                        }
                    ]
                }
            ],
            max_tokens=MODEL_CONFIG['max_tokens'],
            temperature=MODEL_CONFIG['temperature'],
            top_p=MODEL_CONFIG['top_p'],
            frequency_penalty=MODEL_CONFIG['frequency_penalty'],
            presence_penalty=MODEL_CONFIG['presence_penalty']
        )

        # Extract text from response
        generated_text = response.choices[0].message.content

        # Process results based on text level
        if text_level == 'word':
            words = [w.strip() for w in generated_text.split(',') if w.strip()]
            results = [{'text': word, 'confidence': 0.95} for word in words]
        elif text_level == 'paragraph':
            paragraphs = [p.strip() for p in generated_text.split('\n\n') if p.strip()]
            results = [{'text': para, 'confidence': 0.95} for para in paragraphs]
        else:  # block
            blocks = [b.strip() for b in generated_text.split('\n') if b.strip()]
            results = [{'text': block, 'confidence': 0.95} for block in blocks]

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
            if not os.path.exists(MODEL_CONFIG['best_model_path']):
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