from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///customer_churn.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==========================================
# DATABASE TABLE
# ==========================================

class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    gender = db.Column(db.String(50))
    senior = db.Column(db.Integer)
    tenure = db.Column(db.Float)

    monthly_charges = db.Column(db.Float)
    total_charges = db.Column(db.Float)

    prediction = db.Column(db.String(50))
    probability = db.Column(db.Float)

    risk_level = db.Column(db.String(50))


# ==========================================
# LOAD MODEL
# ==========================================

saved_file = joblib.load(
    "Customer_churn_model.pkl"
)

model = saved_file["model"]

# ==========================================
# LOAD ENCODERS
# ==========================================

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    all_predictions = Prediction.query.all()

    return render_template(
        "history.html",
        predictions=all_predictions
    )

@app.route("/dashboard")
def dashboard():

    # ==========================
    # COUNTS
    # ==========================

    churn_count = Prediction.query.filter_by(
        prediction="Customer Likely To Churn"
    ).count()

    stay_count = Prediction.query.filter_by(
        prediction="Customer Likely To Stay"
    ).count()

    low_risk = Prediction.query.filter_by(
        risk_level="Low Risk"
    ).count()

    medium_risk = Prediction.query.filter_by(
        risk_level="Medium Risk"
    ).count()

    high_risk = Prediction.query.filter_by(
        risk_level="High Risk"
    ).count()

    total_predictions = Prediction.query.count()

    # ==========================
    # AVERAGE PROBABILITY
    # ==========================

    all_predictions = Prediction.query.all()

    probabilities = [
        p.probability
        for p in all_predictions
        if p.probability is not None
    ]

    avg_probability = round(
        sum(probabilities) / len(probabilities),
        2
    ) if probabilities else 0

    # ==========================
    # STATIC FOLDER
    # ==========================

    os.makedirs(
        "static",
        exist_ok=True
    )

    # ==========================
    # CHURN VS STAY BAR CHART
    # ==========================

    plt.figure(figsize=(5, 3))

    plt.bar(
        ["Churn", "Stay"],
        [churn_count, stay_count]
    )

    plt.title(
        "Churn vs Stay",
        fontsize=14
    )

    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(
        "static/churn_chart.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # PIE CHART
    # ==========================

    plt.figure(figsize=(4, 4))

    plt.pie(
        [churn_count, stay_count],
        labels=["Churn", "Stay"],
        autopct="%1.1f%%"
    )

    plt.title(
        "Prediction Split",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        "static/pie_chart.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # RISK DISTRIBUTION
    # ==========================

    plt.figure(figsize=(5, 3))

    plt.bar(
        ["Low", "Medium", "High"],
        [
            low_risk,
            medium_risk,
            high_risk
        ]
    )

    plt.title(
        "Risk Distribution",
        fontsize=14
    )

    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(
        "static/risk_chart.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # TREND GRAPH
    # ==========================

    plt.figure(figsize=(5, 3))

    plt.plot(
        probabilities,
        marker="o"
    )

    plt.title(
        "Probability Trend",
        fontsize=14
    )

    plt.ylabel("Risk %")
    plt.xlabel("Predictions")

    plt.tight_layout()

    plt.savefig(
        "static/trend_chart.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # RETURN TEMPLATE
    # ==========================

    return render_template(
        "dashboard.html",
        churn_count=churn_count,
        stay_count=stay_count,
        low_risk=low_risk,
        medium_risk=medium_risk,
        high_risk=high_risk,
        total_predictions=total_predictions,
        avg_probability=avg_probability
    )

# ==========================================
# PREDICT ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # ------------------------
    # CATEGORICAL INPUTS
    # ------------------------

    gender_text = request.form["gender"]

    gender = encoders["gender"].transform(
        [gender_text]
    )[0]

    partner = encoders["Partner"].transform(
        [request.form["Partner"]]
    )[0]

    dependents = encoders["Dependents"].transform(
        [request.form["Dependents"]]
    )[0]

    phone = encoders["PhoneService"].transform(
        [request.form["PhoneService"]]
    )[0]

    multiline = encoders["MultipleLines"].transform(
        [request.form["MultipleLines"]]
    )[0]

    internet = encoders["InternetService"].transform(
        [request.form["InternetService"]]
    )[0]

    security = encoders["OnlineSecurity"].transform(
        [request.form["OnlineSecurity"]]
    )[0]

    backup = encoders["OnlineBackup"].transform(
        [request.form["OnlineBackup"]]
    )[0]

    protection = encoders["DeviceProtection"].transform(
        [request.form["DeviceProtection"]]
    )[0]

    support = encoders["TechSupport"].transform(
        [request.form["TechSupport"]]
    )[0]

    tv = encoders["StreamingTV"].transform(
        [request.form["StreamingTV"]]
    )[0]

    movies = encoders["StreamingMovies"].transform(
        [request.form["StreamingMovies"]]
    )[0]

    contract = encoders["Contract"].transform(
        [request.form["Contract"]]
    )[0]

    billing = encoders["PaperlessBilling"].transform(
        [request.form["PaperlessBilling"]]
    )[0]

    payment = encoders["PaymentMethod"].transform(
        [request.form["PaymentMethod"]]
    )[0]

    # ------------------------
    # NUMERIC INPUTS
    # ------------------------

    senior = int(
        request.form["SeniorCitizen"]
    )

    tenure = float(
        request.form["tenure"]
    )

    monthly = float(
        request.form["MonthlyCharges"]
    )

    total = float(
        request.form["TotalCharges"]
    )

    # ------------------------
    # MODEL INPUT ARRAY
    # ------------------------

    data = np.array([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phone,
        multiline,
        internet,
        security,
        backup,
        protection,
        support,
        tv,
        movies,
        contract,
        billing,
        payment,
        monthly,
        total
    ]])

    # ------------------------
    # MODEL PREDICTION
    # ------------------------

    prediction = model.predict(data)[0]

    probability = model.predict_proba(
        data
    )[0][1]

    probability_percent = round(
        probability * 100,
        2
    )

    result = (
        "Customer Likely To Churn"
        if prediction == 1
        else
        "Customer Likely To Stay"
    )

    # ------------------------
    # RISK LEVEL
    # ------------------------

    if probability_percent >= 70:
        risk_level = "High Risk"
        color = "high"

    elif probability_percent >= 40:
        risk_level = "Medium Risk"
        color = "medium"

    else:
        risk_level = "Low Risk"
        color = "low"

    # ------------------------
    # SAVE TO DATABASE
    # ------------------------

    new_prediction = Prediction(
        gender=gender_text,
        senior=senior,
        tenure=tenure,
        monthly_charges=monthly,
        total_charges=total,
        prediction=result,
        probability=probability_percent,
        risk_level=risk_level
    )

    db.session.add(new_prediction)
    db.session.commit()

    # ------------------------
    # RESULT PAGE
    # ------------------------

    return render_template(
        "result.html",
        prediction=result,
        probability=probability_percent,
        risk_level=risk_level,
        color=color
    )


# ==========================================
# CREATE DATABASE
# ==========================================

with app.app_context():
    db.create_all()


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)