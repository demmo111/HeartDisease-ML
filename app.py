import joblib
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Load Model and Feature Artifacts ---
# These files are created during the model training phase (in your .ipynb notebook)

try:
    # Load the pre-trained Gradient Boosting model
    model = joblib.load("best_heart_risk_model.pkl")
    
    # Load the list of feature names that the model was trained on
    with open("selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)

except FileNotFoundError:
    print("ERROR: Model or feature files not found. Make sure 'best_heart_risk_model.pkl' and 'selected_features.pkl' are in the same directory.")
    model = None
    selected_features = []

# --- 2. Define Feature Lists and Labels for the Frontend ---

# Separate the binary features for populating checkboxes in the HTML form
# The age-related features are calculated on the backend, so we exclude them here.
binary_features = [
    f for f in selected_features if f not in ("Age", "Age_Category_Young", "Age_Category_Senior")
]

# Create more user-friendly labels for the form
# These are used by the Jinja2 template in `index.html`
feature_labels = {
    "Dizziness": "Dizziness",
    "Chest_Pain": "Chest Pain",
    "Gender": "Gender",
    "Pain_Arms_Jaw_Back": "Pain in Arms, Jaw, or Back",
    "Palpitations": "Palpitations (Fast/Irregular Heartbeat)",
    "Diabetes": "Diabetes",
    "Fatigue": "Fatigue",
    "Family_History": "Family History of Heart Disease",
    "Shortness_of_Breath": "Shortness of Breath",
    "Cold_Sweats_Nausea": "Cold Sweats or Nausea",
    "Obesity": "Obesity",
    "High_BP": "High Blood Pressure",
    "Smoking": "Smoking",
    "Chronic_Stress": "Chronic Stress",
    "Swelling": "Swelling (Edema)",
    "Sedentary_Lifestyle": "Sedentary Lifestyle",
    "High_Cholesterol": "High Cholesterol",
}


# --- 3. Define Application Routes ---

@app.route('/')
def home():
    """
    Renders the main user interface page (`index.html`).
    Passes the list of features and their labels to the template,
    so it can dynamically generate the form checkboxes.
    """
    # This renders the new, modern single-page HTML file
    return render_template(
        'index.html', 
        binary_features=binary_features, 
        feature_labels=feature_labels
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend's JavaScript.
    Receives patient data as JSON, preprocesses it, and returns the
    risk prediction as JSON.
    """
    if not model:
        return jsonify({"error": "Model is not loaded."}), 500

    # Get data from the POST request (sent by the JavaScript fetch call)
    data = request.get_json()

    # --- Preprocessing ---
    # 1. Handle Age: Convert to float and create age category dummy variables
    try:
        age_val = float(data.get("Age", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid Age value provided."}), 400
        
    data["Age_Category_Young"] = 1 if age_val <= 40 else 0
    data["Age_Category_Senior"] = 1 if age_val > 60 else 0

    # 2. Build DataFrame: Create a dictionary for the input row, ensuring all
    #    expected features are present with a default value of 0.
    #    This prevents errors if a checkbox isn't checked (and thus not sent).
    row = {feat: data.get(feat, 0) for feat in selected_features}
    
    # Create a single-row DataFrame with columns in the exact order the model expects
    df_input = pd.DataFrame([row], columns=selected_features)

    # --- Prediction ---
    # Use predict_proba to get the probability of the positive class (risk=1)
    # The result is an array like [[prob_no_risk, prob_risk]]
    prediction_proba = model.predict_proba(df_input)
    
    # Get the probability of risk (the second element) and format as a percentage
    risk_percentage = prediction_proba[0][1] * 100
    
    # Return the result as a JSON object
    return jsonify(risk=f"{risk_percentage:.1f}%")

# --- 4. Run the Application ---
if __name__ == '__main__':
    # Setting debug=True is useful for development but should be
    # turned off for a production deployment.
    app.run(debug=True)