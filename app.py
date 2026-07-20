from flask import Flask, request, jsonify, render_template
import base64
import subprocess
import tempfile
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name
    
    try:
        result = subprocess.run(
            ['python3', 'inference_seq.py', image_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return jsonify({'error': result.stderr})
        
        data = json.loads(result.stdout.strip())
        return jsonify({
            'class': data.get('class', 'Неизвестно'),
            'description': data.get('description', 'Описание отсутствует'),
            'original': data.get('original'),
            'mask': data.get('mask'),
            'overlay': data.get('overlay')
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.unlink(image_path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
