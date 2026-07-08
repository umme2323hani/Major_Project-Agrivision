from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import subprocess
import os
import re
import joblib  # For saving and loading the trained model

app = Flask(__name__)

# Path configuration for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the pdftoppm executable
pdftoppm_path = r'C:\poppler\Library\bin\pdftoppm'

# Global variable to store the trained model
model = None

# Path for the uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store extracted data for autofill
extracted_data = {}

# Path to save the trained model
MODEL_PATH = 'trained_model.pkl'

# Load the dataset for crop prediction
def train_model():
    global model
    try:
        # Check if a model exists and load it
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded from disk.")
            return

        df = pd.read_csv('Dataset.csv')

        # Data cleaning and preprocessing
        df = df.dropna(subset=['Crop Type'])
        target_columns = ['Rainfall(mm)', 'Humidity(%)', 'Temperature(C)', 'Fertilizer(kg/ha)', 
                         'Water(L/day)', 'Sunlight(hours)', 'Pesticide Usage(kg/ha)', 
                         'Past Yield(tons/ha)', 'Seed Variety', 'Irrigation Method', 
                         'Crop Rotation(1/0)', 'Crop Type']

        X = df[['Moisture(%)', 'pH Value', 'Sandy(1/0)', 'Chalky(1/0)', 'Clay(1/0)', 
               'Nitrogen(N)', 'Phosphorus(P)', 'Potassium(K)', 'Soil Moisture Level (1-100%)']]
        y = df[target_columns]

        for col in target_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=target_columns)
        X = df[['Moisture(%)', 'pH Value', 'Sandy(1/0)', 'Chalky(1/0)', 'Clay(1/0)', 
               'Nitrogen(N)', 'Phosphorus(P)', 'Potassium(K)', 'Soil Moisture Level (1-100%)']]
        y = df[target_columns]

        for col in target_columns:
            if y[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(y[col]):
                df[col] = LabelEncoder().fit_transform(df[col])

        if df.shape[0] >= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputClassifier(RandomForestClassifier())
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)
            print("Model trained and saved to disk.")
        else:
            print("Not enough data left to split into training and testing sets.")
    except Exception as e:
        print("Error loading or training model:", e)

# Train the model when the app starts
train_model()

# Crop recommendations data
crop_recommendations = {
    'maize': ['Rice: 2,200-3,600 kg per hectare', 'Wheat: 900-1,350 kg per hectare', 'Barley: 900-1,350 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare', 'Cotton: 1,400-2,500 kg per hectare', 'Sorghum: 2,200-3,000 kg per hectare'],
    'soybean': ['Cotton: 1,400-2,500 kg per hectare', 'Maize: 3,600-5,400 kg per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Wheat: 900-1,350 kg per hectare', 'Barley: 900-1,350 kg per hectare', 'Sorghum: 2,000-3,000 kg per hectare'],
    'rice': ['Maize: 3,600-5,400 kg per hectare', 'Wheat: 900-1,350 kg per hectare', 'Sorghum: 2,200-3,000 kg per hectare', 'Cotton: 1,400-2,500 kg per hectare', 'Barley: 900-1,350 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare'],
    'wheat': ['Maize: 3,600-5,400 kg per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Barley: 900-1,350 kg per hectare', 'Cotton: 1,400-2,500 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare', 'Sorghum: 2,000-3,000 kg per hectare'],
    'cotton': ['Maize: 3,600-5,400 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare', 'Wheat: 900-1,350 kg per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Barley: 900-1,350 kg per hectare', 'Sorghum: 2,200-3,000 kg per hectare'],
    'barley': ['Maize: 3,600-5,400 kg per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Wheat: 900-1,350 kg per hectare', 'Cotton: 1,400-2,500 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare', 'Sorghum: 2,000-3,000 kg per hectare'],
    'sugarcane': ['We can’t produce under the given conditions.'],
    'groundnut': ['We can’t produce under the given conditions.'],
    'sunflower': ['We can’t produce under the given conditions.'],
    'tomato': ['Carrot: 25-35 tons per hectare', 'Potato: 30-40 tons per hectare', 'Onion: 25-35 tons per hectare', 'Sugarcane: 50-70 tons per hectare', 'Sunflower: 2,200-3,000 kg per hectare'],
    'potato': ['Tomato: 25-35 tons per hectare', 'Onion: 20-30 tons per hectare', 'Carrot: 25-35 tons per hectare', 'Wheat: 900-1,350 kg per hectare', 'Maize: 3,600-5,400 kg per hectare'],
    'carrot': ['Potato: 30-40 tons per hectare', 'Tomato: 25-35 tons per hectare', 'Onion: 20-30 tons per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Barley: 900-1,350 kg per hectare'],
    'onion': ['Carrot: 25-35 tons per hectare', 'Potato: 30-40 tons per hectare', 'Tomato: 25-35 tons per hectare', 'Maize: 3,600-5,400 kg per hectare', 'Sugarcane: 50-70 tons per hectare'],
    'lentil': ['We can’t produce under the given conditions.'],
    'sorghum': ['Maize: 3,600-5,400 kg per hectare', 'Soybean: 1,100-1,600 kg per hectare', 'Rice: 2,200-3,600 kg per hectare', 'Cotton: 1,400-2,500 kg per hectare', 'Barley: 900-1,350 kg per hectare']
}


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/textextraction')
def textextraction():
    return render_template('text_extraction.html')

@app.route('/upload', methods=["POST"])
def upload_file():
    global extracted_data
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(".pdf"):
            image_files = convert_pdf_to_images(file_path)
            extracted_text = ""
            for image_file in image_files:
                image = preprocess_image(image_file)
                extracted_text += pytesseract.image_to_string(image, config="--oem 3 --psm 6") + "\n"
        else:
            image = preprocess_image(file_path)
            extracted_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")

        # Print the extracted text in the terminal for debugging
        print("Extracted Text:", extracted_text)

        extracted_data = extract_values_from_text(extracted_text)

        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/index')
def index():
    return render_template('index.html', autofill_data=extracted_data)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        # Check if the model has been loaded
        if model is None:
            print("Model is not loaded. Training the model...")
            train_model()  # Train the model if not already trained
            if model is None:
                raise ValueError("The model has not been trained yet.")
        
        print("Model is loaded and ready for prediction.")
        form_data = request.form
        print("Form data received:", form_data)

        # Ensure all required fields are present
        required_fields = ['moisture', 'ph', 'sandy', 'chalky', 'clay', 'nitrogen', 'phosphorus', 'potassium', 'soil_moisture_level']
        for field in required_fields:
            if field not in form_data:
                print(f"Missing field: {field}")
                raise ValueError(f"Missing input for {field}")

        input_data = [[
            float(form_data[field]) if field not in ['sandy', 'chalky', 'clay'] else int(form_data[field])
            for field in required_fields
        ]]

        try:
            input_data = [[float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value for value in input_data[0]]]
            print("Input data successfully validated:", input_data)
        except ValueError as ve:
            print("Error during input data validation:", ve)
            raise ValueError(f"Invalid input format: {ve}")

        # Make the prediction
        print("Making prediction...")
        prediction = model.predict(input_data)[0]  # Assuming the output is a single array
        print("Raw prediction output:", prediction)

        # Define the descriptive fields for the prediction
        prediction_fields = [
            'Rainfall (mm)', 'Humidity (%)', 'Temperature (°C)', 'Fertilizer (kg/ha)', 
            'Water (L/day)', 'Sunlight (hours)', 'Pesticide Usage (kg/ha)', 
            'Past Yield (tons/ha)', 'Seed Variety', 'Irrigation Method', 
            'Crop Rotation (1/0)', 'Crop Type'
        ]

        # Crop mapping for last field
        crop_mapping = {
            0: 'Maize',
            1: 'Rice',
            2: 'Cotton',
            3: 'Wheat',
            4: 'Soybean',
            5: 'Sugarcane',
            6: 'Barley',
            7: 'Groundnut',
            8: 'Sunflower',
            9: 'Tomato',
            10: 'Potato',
            11: 'Carrot',
            12: 'Onion',
            13: 'Lentil',
            14: 'Sorghum'
        }


        # Build a dictionary for formatted prediction
        formatted_prediction = {
            field: value for field, value in zip(prediction_fields, prediction)
        }

        # Add crop name based on the last field (Crop Type)
        crop_type_index = int(prediction[-1])  # Adjust based on how the crop type is represented
        formatted_prediction['Crop Name'] = crop_mapping.get(crop_type_index, 'Unknown Crop')

        print("Formatted prediction output:", formatted_prediction)

        # Prepare data for display
        target_values = {
            'Moisture(%)': form_data['moisture'],
            'PH Value': form_data['ph'],
            'Sandy(1/0)': form_data['sandy'],
            'Chalky(1/0)': form_data['chalky'],
            'Clay(1/0)': form_data['clay'],
            'Nitrogen(N)': form_data['nitrogen'],
            'Phosphorus(P)': form_data['phosphorus'],
            'Potassium(K)': form_data['potassium'],
            'Soil Moisture Level (1-100%)': form_data['soil_moisture_level']
        }

        return render_template('result.html', prediction=formatted_prediction, target_values=target_values, recommendations=crop_recommendations.get(formatted_prediction['Crop Name'].lower(), []))
    except ValueError as ve:
        print("ValueError:", ve)
        return f"Invalid input. Please make sure all fields are filled correctly. Error: {ve}", 400
    except Exception as e:
        print("Exception:", e)
        return f"An error occurred: {e}", 500

@app.route('/recom')
def recom():
    return render_template('result2.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the predicted crop name from the form data
        predicted_crop = request.form['predicted_crop'].lower()
        print("Predicted crop received:", predicted_crop)  # Debugging line

        # Retrieve recommendations for the predicted crop, or default to a message
        recommendations = crop_recommendations.get(predicted_crop, ['No recommendations available'])
        
        # Crop yield data per acre for the crops
        crop_yield_per_acre = {
        'wheat': '30 bushels per acre',
        'rice': '40 quintals per acre',
        'maize': '120 quintals per acre',
        'soybean': '50 bushels per acre',
        'cotton': '150 bushels per acre',
        'sugarcane': '50 tons per acre',
        'barley': '25 bushels per acre',
        'groundnut': '20 quintals per acre',
        'sunflower': '15 quintals per acre',
        'tomato': '30 tons per acre',
        'potato': '25 tons per acre',
        'carrot': '30 tons per acre',
        'onion': '20 tons per acre',
        'lentil': '12 quintals per acre',
        'sorghum': '25 quintals per acre'
    }

        
        yield_per_acre = crop_yield_per_acre.get(predicted_crop, 'Data not available')

        # Render the result template with the data
        return render_template('result2.html', predicted_crop=predicted_crop.capitalize(), recommendations=recommendations, yield_per_acre=yield_per_acre)
    except Exception as e:
        print("An error occurred in the /recommend route:", e)
        return f"An error occurred: {e}", 500


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.filter(ImageFilter.MedianFilter())
    image = ImageOps.invert(image)
    return image

def convert_pdf_to_images(pdf_path):
    try:
        output_path = pdf_path.replace('.pdf', '_page')
        subprocess.run([pdftoppm_path, pdf_path, output_path, '-png'], check=True)
        return [output_path + f'-{i}.png' for i in range(1, 10)]
    except Exception as e:
        raise RuntimeError(f"Error converting PDF: {e}")

# Function to extract values from text
def extract_values_from_text(text):
    fields = {
        'moisture': r'Moisture\s*\(%\):?\s?(\d+)',
        'ph': r'pH\s*Value:?\s?(\d+(\.\d+)?)',
        'sandy': r'Sandy\s*\(1/0\):?\s?(\d+)',
        'chalky': r'Chalky\s*\(1/0\):?\s?(\d+)',
        'clay': r'Clay\s*\(1/0\):?\s?(\d+)',
        'nitrogen': r'Nitrogen\s*\(N\):?\s?(\d+(\.\d+)?)',
        'phosphorus': r'Phosphorus\s*\(P\):?\s?(\d+(\.\d+)?)',
        'potassium': r'Potassium\s*\(K\):?\s?(\d+(\.\d+)?)',
        'soil_moisture_level': r'Soil\s*Moisture\s*Level\s*\(1-100%\):?\s?(\d+(\.\d+)?)'
    }
    extracted = {}
    for field, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted[field] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    print('Extracted Values:', extracted)  # Debugging: Log extracted values
    return extracted

@app.route('/autofill', methods=["GET"])
def autofill():
    global extracted_data
    defaults = {
        'moisture': 20,
        'ph': 7.0,
        'sandy': 0,
        'chalky': 0,
        'clay': 0,
        'nitrogen': 30,
        'phosphorus': 35,
        'potassium': 40,
        'soil_moisture_level': 50
    }
    autofill_data = {**defaults, **extracted_data}
    return jsonify(autofill_data)

if __name__ == '__main__':
    app.run(debug=True)


