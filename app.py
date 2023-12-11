from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import subprocess

app = Flask(__name__, template_folder='templates')

training_status = {'status': 'idle'}

def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((232, 232))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def decode_predictions(predictions):
    classes = ['COVID', 'NORMAL', 'PNEUMONIA']
    decoded_preds = {}
    for i, pred in enumerate(predictions[0]):
        decoded_preds[classes[i]] = float((pred)*100)
    return decoded_preds

def train_model():
    global training_status
    training_status['status'] = 'training'
    try:
        # Get the selected model from the JSON request
        model_choice = request.json.get('model_choice', 'model')
        
        print(f'training : {model_choice}')
        # Run the corresponding training file
        subprocess.run(['python', f'{model_choice}_training.py'], check=True)
    except subprocess.CalledProcessError as e:
        print('Error during training:', e)
    training_status['status'] = 'idle'

@app.route('/')
def main_page():
    return render_template('presentation.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(request.files['image'].stream).convert("RGB")
        model_choice = request.form.get('model_choice', 'Custom_model')
        model_path = f'datasets\Models\{model_choice}.h5'  # Fix the concatenation here
        model = load_model(model_path)
        preprocessed_img = preprocess_image(img)
        predictions = model.predict(preprocessed_img)
        decoded_predictions = decode_predictions(predictions)
        return jsonify({'predictions': decoded_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status['status'] == 'idle':
        train_model()
        return jsonify({'success': True, 'message': 'Training completed.'})
    else:
        return jsonify({'success': False, 'message': 'Training in progress.'})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    global training_status
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(debug=True)
