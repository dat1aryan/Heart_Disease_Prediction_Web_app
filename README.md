# HeartPulse ❤️
**Empowering heart health awareness through AI-driven clinical insights.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://heartdiseaseprediction-webapp.streamlit.app/)

## Overview
HeartPulse is a high-performance machine learning web application designed to predict the risk of heart disease based on clinical patient data. By leveraging advanced classification algorithms, it translates complex medical metrics into clear, actionable risk assessments, aiding in early health awareness and proactive consultation.

## Key Features
- **Instant ML Predictions**: Real-time analysis using a optimized XGBoost classification pipeline.
- **Premium UI/UX**: Responsive, card-based interface inspired by modern medical software aesthetics.
- **Dynamic Interaction**: Live input validation and high-resolution visual feedback.
- **Secure & Lightweight**: Serverless deployment architecture with optimized resource footprint.

## UI Preview
![App Screenshot](assets/hero_heart.png)

## Tech Stack
- **Languages**: Python
- **Interface**: Streamlit, Custom Vanilla CSS
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Engineering**: Pandas, NumPy
- **Environment**: Joblib (Serialization)

## Installation & Setup
Follow these steps to run the project locally.

1. **Clone the repository**
   ```bash
   git clone https://github.com/dat1aryan/Heart_Disease_Prediction_Web_app.git
   cd Heart_Disease_Prediction_Web_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the sidebar to navigate between Home, About Model, and How It Works.
2. Enter patient clinical data (Age, Chest Pain Type, Cholesterol, etc.) into the form controls.
3. Observe real-time validation for input bounds.
4. Click **Predict Risk** to generate a risk score and confidence level.

## Project Structure
```bash
Heart_Disease_Prediction_Web_app/
├── app.py              # Main Streamlit application and UI logic
├── assets/             # Media files, icons, and SVG assets
├── data/               # Clinical dataset (CSV)
├── models/             # Trained ML pipeline serialization (PKL)
├── requirements.txt    # Project dependencies
└── LICENSE             # Apache 2.0 License
```

## Model Details
- **Algorithm**: XGBoost Classifier
- **Dataset**: UCI Heart Disease Dataset (303 records, 13 clinical features)
- **Features Included**: Age, sex, cp (chest pain), trestbps (resting BP), chol, fbs, restecg, thalach (max HR), exang, oldpeak, slope, ca, thal.

## Future Improvements
- **Multi-Model Comparison**: Integration of Random Forest and LightGBM for comparative analysis.
- **PDF Report Generation**: Exporting personalized risk assessments as downloadable clinical reports.
- **User Authentication**: Secure profile management for historical health tracking.
- **Direct EHR Integration**: Potential for API connectivity with Electronic Health Record systems.

## Contributing
Contributions are welcome! Please follow these steps for a clean contribution flow:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m '-'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the **Apache License 2.0**. See `LICENSE` for more information.
