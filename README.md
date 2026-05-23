# 🩺 Multiple Disease Prediction System

## 📌 Overview
The Multiple Disease Prediction System is a machine learning-based web application designed to predict multiple diseases using patient health data. The system uses algorithms such as Logistic Regression and Support Vector Machine (SVM) to provide predictions for diseases like diabetes, heart disease, kidney disease, liver disease, breast cancer, malaria, and pneumonia.

The application is developed using Flask, which connects the trained machine learning models with a user-friendly web interface. Unlike traditional systems that focus on predicting only one disease, this project provides a single integrated platform for predicting multiple diseases efficiently.

## ⚙️ Technologies Used
* **Backend / ML:** Python 3.x, Scikit-Learn (Logistic Regression, SVM), Pandas, NumPy
* **Web Framework:** Flask
* **Frontend:** HTML5, CSS3
* **Flask 
* **Model Serialization:** Pickle / Joblib

## 📂 Project Structure
```text
├── app.py                # Main Flask application and routing script
├── models/               # Directory containing trained & serialized ML models (.pkl files)
├── static/               # CSS styles, JavaScript files, and images
├── templates/            # HTML templates for the web interface forms and results
├── requirements.txt      # List of required Python packages and dependencies
└── README.md             # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/multiple-disease-prediction.git
   cd multiple-disease-prediction
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scriptsctivate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your web browser and navigate to `http://127.0.0.1:5000/`

## 💡 Usage
1. Open the web application in your browser.
2. Select the specific disease prediction module from the homepage dashboard (e.g., Heart Disease, Diabetes, etc.).
3. Input the patient's medical test results and health parameters into the form.
4. Click the **Predict** button.
5. The system will process the input using the respective trained ML model and display the predicted outcome.

## 📜 License
This project is licensed under the MIT License.
