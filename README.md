## 🚀 Emotion Detection Using Computer Vision



```markdown

This project implements an emotion detection model using computer vision and deep learning techniques. The model is trained to predict emotions from facial images, and it supports the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The backend of this project is built using **Flask** for serving the model, while the frontend is developed with **HTML**, **CSS**, and **JavaScript** for a seamless user experience.

## Features

- **Emotion Detection**: Predicts emotions from facial images.
- **Web Interface**: Users can upload images through a web interface and get real-time predictions.
- **User-friendly**: A simple and intuitive UI that makes it easy to interact with the model.

## Technologies Used

- **Python**: Backend programming language.
- **TensorFlow/Keras**: For building and using the emotion detection model.
- **Flask**: A micro web framework for serving the model.
- **HTML, CSS, JavaScript**: Frontend for the web interface.
- **NumPy**: For numerical processing.
- **Pillow**: For image preprocessing.
- **Bootstrap**: For a responsive and mobile-friendly UI.

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MohammadAnas5/emotionapp.git
   cd emotion-detection
   ```

2. **Set up a virtual environment**:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # For Linux/MacOS
   myenv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the required Python packages including TensorFlow, Flask, and other necessary libraries.

4. **Place your pre-trained model (`new.keras`)** in the project directory.

5. **Run the Flask app**:

   ```bash
   python app.py
   ```

6. **Access the web interface**:

   Open a web browser and navigate to `http://127.0.0.1:5000` to see the emotion detection web app in action.

## 🎨 UI Screenshots

### Home Page
![Home Page](https://www.awesomescreenshot.com/image/52368151?key=f3449751034a76f4cc2c8625d269ce4c)

### Prediction Result
![Prediction Result](https://www.awesomescreenshot.com/image/52368118?key=6557fb6daff4dc476ba9bae237cfea3f)

## Model Details

This project uses a pre-trained convolutional neural network (CNN) model to detect emotions. The model is loaded using **TensorFlow/Keras** and can predict seven different emotions based on the input image:

- **Emotion labels**: 
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

## File Structure (Some files are not uploaded due to big size)

```
emotion-detection/
│
├── app.py                # Main Flask application
├── new.keras             # Pre-trained emotion detection model
├── requirements.txt      # Python dependencies
├── static/               # Static files (e.g., images, CSS)
│   └── style.css         # Custom CSS for styling
├── templates/            # HTML templates
│   └── index.html        # Main page of the app
└── README.md             # Project documentation
```

## How It Works

1. The user uploads an image through the frontend web interface.
2. The image is sent to the backend (Flask server), where it is preprocessed (resized, normalized) and passed into the model.
3. The model predicts the emotion, and the result is sent back to the frontend, where it is displayed to the user.

### Preprocessing

Before the image is passed to the model, it is:

1. Converted to RGB to ensure it has 3 channels.
2. Resized to 224x224 pixels, which is the input size for the model.
3. Scaled to a [0,1] range (rescaling pixel values).
4. Expanded to add a batch dimension, making it suitable for prediction.

### Model Inference

The model uses the `predict()` function to return the probabilities of each emotion. The emotion with the highest probability is selected and displayed to the user.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **TensorFlow/Keras**: For providing the deep learning framework used for model training and inference.
- **Flask**: For serving the model through a web API.
- **Pillow**: For image preprocessing.
- **Bootstrap**: For the responsive frontend UI.

## Contact

For any questions or issues, feel free to open an issue or reach out to me via [anassiddiqui634@gmal.com].

```

### Notes:
1. **Model**: If you're using a custom model file, mention its loading and usage in the README.
2. **Dependencies**: You can update the `requirements.txt` by running `pip freeze > requirements.txt` after installing all the necessary libraries.
3. **GitHub URL**: Replace the placeholder `https://github.com/yourusername/emotion-detection.git` with the actual GitHub repository URL.

This README will guide users through setting up the project and running it locally. Let me know if you need any additional sections or adjustments!
