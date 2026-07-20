from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(image_bytes))
    
    # Пока просто текст, завтра поставим модель
    result_text = "Классификация: пока не определена (жду новую модель)"
    return jsonify({'result': result_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
