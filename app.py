# app.py
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# --- Model and Constants Configuration ---

# Define the disease classes (22 classes as requested)
# NOTE: Replace these with your actual disease names.
DISEASE_CLASSES = [
    "Actinic Keratoses and Intraepithelial Carcinoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
    "Acne Vulgaris",
    "Eczema",
    "Psoriasis",
    "Ringworm (Tinea)",
    "Shingles (Herpes Zoster)",
    "Warts",
    "Vitiligo",
    "Alopecia Areata",
    "Lupus Erythematosus",
    "Seborrheic Dermatitis",
    "Rosacea",
    "Hyperhidrosis",
    "Lichen Planus",
    "Scabies",
    "Other/Unknown" # Placeholder for any non-specific findings
]

# Set the size the model expects
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model.h5' # The trained Keras model file

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Keras Model Loading (Placeholder) ---

# Global variable for the model
model = None

def load_model():
    """
    Loads the trained Keras model.
    In a real-world scenario, you would use:
    
    from tensorflow.keras.models import load_model
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Optionally exit or disable prediction feature
    """
    global model
    # Placeholder: A simple function that mimics the model's structure
    class DummyModel:
        def predict(self, x):
            # Create a dummy prediction array (22 classes)
            # Make the first class (index 0) the most confident one for testing
            dummy_preds = np.random.rand(1, len(DISEASE_CLASSES))
            dummy_preds[0, 0] = 0.95 + np.random.rand() * 0.05
            # Normalize to sum to 1 (softmax-like output)
            dummy_preds /= dummy_preds.sum()
            return dummy_preds

    model = DummyModel()
    print("WARNING: Using a DUMMY model. Replace with your actual Keras model load.")


# Load the model when the application starts
with app.app_context():
    load_model()


# --- Utility Functions ---

def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Loads, resizes, and normalizes the image for model prediction.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        app.logger.error(f"Image preprocessing failed: {e}")
        return None

def predict_image(image_array):
    """
    Performs prediction using the loaded model.
    """
    if model is None:
        raise Exception("AI Model not loaded.")
        
    predictions = model.predict(image_array)[0]
    
    # Get indices of top 3 predictions
    top_3_indices = np.argsort(predictions)[::-1][:3]
    
    # Format the top 3 results
    top_3_results = []
    for i in top_3_indices:
        top_3_results.append({
            'disease': DISEASE_CLASSES[i],
            'confidence': float(predictions[i])
        })
        
    return top_3_results


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main application page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, preprocessing, and prediction.
    """
    # 1. Check for 'file' in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    
    # 2. Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # 3. Validate file type and secure filename
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the file
            file.save(filepath)
            
            # 4. Preprocess the image
            processed_image = preprocess_image(filepath)
            if processed_image is None:
                os.remove(filepath) # Cleanup
                return jsonify({'error': 'Failed to preprocess image'}), 500
                
            # 5. Get prediction
            predictions = predict_image(processed_image)
            
            # 6. Prepare the final response (best prediction is first in the list)
            best_prediction = predictions[0]
            
            response = {
                'status': 'success',
                'best_disease_name': best_prediction['disease'],
                'best_confidence': round(best_prediction['confidence'] * 100, 2),
                'top_3_predictions': [
                    {'disease': p['disease'], 'confidence': round(p['confidence'] * 100, 2)}
                    for p in predictions
                ],
                'image_url': f'/uploads/{filename}' # Path to display image
            }
            
            # Cleanup the saved file (optional, but good practice for production)
            # os.remove(filepath) 
            
            return jsonify(response)
            
        except Exception as e:
            app.logger.error(f"Prediction failed: {e}")
            # Ensure file cleanup on error
            if os.path.exists(filepath):
                 os.remove(filepath)
            return jsonify({'error': f'An internal error occurred during prediction: {str(e)}'}), 500
            
    else:
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed.'}), 400

# Route to serve uploaded files securely
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Securely serves files from the upload folder."""
    # This should be handled securely. For a simple setup:
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Use a high port and enable debug mode for development
    # Use: python app.py
    # Access at: http://127.0.0.1:5000/
    print("DermAI Flask app starting...")
    app.run(debug=True)