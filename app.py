from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    bmi = float(request.form["bmi"])
    bp = float(request.form["bp"])
    s1 = float(request.form["tc"])   # total cholesterol
    s2 = float(request.form["ldl"])
    s3 = float(request.form["hdl"])
    s4 = float(request.form["tch"])
    s5 = float(request.form["ltg"])
    s6 = float(request.form["glu"])

    # Put into numpy array
    features = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

    # Predict
    prediction = model.predict(features)[0]

    # Show result
    if prediction == 1:
        result = "The person is likely diabetic."
        diet_plan = """Common Diet Plan:
        - Eat whole grains, vegetables, fruits
        - Avoid sugary drinks & junk food
        - Eat lean proteins (fish, chicken, beans)
        - Use olive oil instead of butter
        - Exercise daily for at least 30 mins"""
    else:
        result = "The person is not diabetic."
        diet_plan = "Maintain a balanced diet and active lifestyle."

    return render_template("result.html", prediction=result, diet=diet_plan)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)