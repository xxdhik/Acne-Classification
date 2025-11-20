from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.inception_v3 import preprocess_input 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
model = tf.keras.models.load_model('acne.h5')
INPUT_SIZE = model.input_shape[1]

CLASS_NAMES = ["Acne Fulminans", "Acne Nodules", "Fungal Acne"]

def preprocess(img):
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('home.html', title="Home")

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', title="Predict", result=None, image_file=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    x = preprocess(img)
    pred = model.predict(x)
    result = CLASS_NAMES[np.argmax(pred)]

    return render_template('predict.html', title="Predict", result=result, image_file=filename)

@app.route('/gallery')
def gallery():
    images = os.listdir(UPLOAD_FOLDER)
    return render_template('gallery.html', title="Gallery", images=images)

@app.route('/training_stats')
def training_stats():
    stats = {
        "acc": 95,
        "val_acc": 93,
        "loss": 0.12,
        "val_loss": 0.21
    }
    return render_template('training_stats.html', title="Training Stats", stats=stats)

if __name__ == '__main__':
    app.run(debug=True)
