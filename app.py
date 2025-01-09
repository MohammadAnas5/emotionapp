

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np 


from tensorflow.keras.utils import img_to_array  
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('new.keras')

# Emotion labels dictionary
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
index_to_emotion = {v: k for k, v in emotion_labels.items()}

# Image preparation function
def prepare_image(img_pil):
    """Preprocess the PIL image to fit your model's input requirements."""
    # Convert the PIL image to RGB to ensure it has 3 channels (R, G, B)
    img = img_pil.convert("RGB")
    # Resize the image to the target size (224, 224)
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the pixel values to [0, 1]
    img_array /= 255.0
    return img_array

# Prediction function
def predict_emotion(image):
    # Preprocess the image
    processed_image = prepare_image(image)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    # Get the emotion label with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    return predicted_emotion

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Open the uploaded image
    image = Image.open(file)
    
    # Predict emotion
    emotion = predict_emotion(image)
    
    return jsonify({'emotion': emotion})

if __name__ == "__main__":
    app.run(debug=True)
