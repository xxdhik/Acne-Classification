from flask import Flask, render_template_string, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.inception_v3 import preprocess_input 

app = Flask(__name__)

# --- KONFIGURASI APLIKASI ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- KONFIGURASI MODEL & KELAS ---
try:
    model = tf.keras.models.load_model('acne.h5')
    INPUT_SIZE = model.input_shape[1] if model.input_shape[1] is not None else 299 
except Exception as e:
    INPUT_SIZE = 299

CLASS_NAMES = [
    "Acne Fulminans",
    "Acne Nodules",
    "Fungal Acne"
]

# --- HTML Template (Revisi untuk Menampilkan Gambar) ---
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Klasifikasi Jenis Jerawat</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        .box { width: 400px; padding: 20px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h2>Klasifikasi Jerawat</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <div class="box">
        <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">Klasifikasi</button>
        </div>
    </form>
    <br>
    {% if result %}
        <h3>Hasil Klasifikasi: {{ result }}</h3>
        {% if image_file %}
        <div style="margin-top: 20px;">
            <h4>Gambar yang Diunggah:</h4>
            <img src="{{ url_for('static', filename='uploads/' + image_file) }}" style="max-width: 300px; height: auto; border: 1px solid #ddd;">
        </div>
        {% endif %}
    {% endif %}
</body>
</html>
"""

# --- Fungsi Preprocessing ---
def preprocess(img):
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    x = preprocess_input(img_array)
    return x

# --- Route Utama ---
@app.route('/')
def home():
    # Pastikan untuk merender template tanpa hasil dan file gambar saat pertama kali dibuka
    return render_template_string(html_page, result=None, image_file=None)

# --- Route Prediksi (Direvisi untuk Penyimpanan File) ---
@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = None
    try:
        file = request.files['file']
        
        # 1. Simpan File Sementara
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        uploaded_file = filename # Simpan nama file untuk dikirim ke template

        # 2. Prediksi
        img = Image.open(filepath).convert("RGB") 
        x = preprocess(img)
        pred = model.predict(x)
        
        class_index = np.argmax(pred)
        
        if class_index < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_index]
        else:
            class_name = f"Indeks tidak valid: {class_index}."

        result = f"{class_name}" 

    except Exception as e:
        result = f"ERROR dalam prediksi: {e}"
        uploaded_file = None # Jika error, jangan tampilkan gambar

    # 3. Render Template dengan Hasil dan Gambar
    return render_template_string(html_page, result=result, image_file=uploaded_file)

# --- Menjalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)