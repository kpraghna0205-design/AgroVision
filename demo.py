from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import cv2
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "agro_secret"
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create uploads folder if missing
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

model = joblib.load('crop_model.pkl')
history = []

# --- LEAF IMAGE ANALYSIS (OpenCV) ---
def analyze_leaf(image_path):
    img = cv2.imread(image_path)
    # Convert to HSV color space to find 'unhealthy' colors (Yellow/Brown)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask)
    
    if yellow_pixels > 5000:
        return "Warning: Yellow/Brown spots detected. Possible Nitrogen deficiency or Fungal Rust.", "Red"
    return "Leaf appears healthy and green.", "Green"

# --- ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'farmer123':
            session['user'] = 'admin'
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid Login")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session: return redirect(url_for('login'))
    
    crop = request.form['crop']
    temp = float(request.form['temp'])
    rain = float(request.form['rain'])
    fert = float(request.form['fert'])
    hum = float(request.form['hum'])

    # AI Yield & Revenue (Rupees ₹)
    prediction = model.predict(np.array([[temp, rain, fert, hum]]))[0]
    prices_in_rupees = {"Rice": 55, "Wheat": 45, "Maize": 35, "Cotton": 75, "Sugarcane": 10, "Potato": 25}
    revenue = prediction * prices_in_rupees.get(crop, 20)

    # Leaf Image Processing
    leaf_status = "No image uploaded."
    status_color = "black"
    if 'leaf_image' in request.files and request.files['leaf_image'].filename != '':
        file = request.files['leaf_image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        leaf_status, status_color = analyze_leaf(filepath)

    advice = "Your crop parameters are optimal."
    if hum > 85: advice = "High humidity detected. Spray organic fungicide."

    return render_template('result.html', crop=crop, yield_val=round(prediction, 2), 
                           revenue=format(int(revenue), ','), advice=advice, leaf_status=leaf_status)

if __name__ == '__main__':
    app.run(debug=True)