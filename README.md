# Predicting the Unpredictable: Intelligent Accident Severity Prediction

**Duration:** January 25 – May 25  
**Tools & Technologies:** Python, R, XGBoost, Feature Engineering

## Overview

This project focuses on building a machine learning model to intelligently predict the severity of road accidents — categorized as **minor**, **serious**, or **fatal**. By analyzing real-world accident datasets, including factors such as weather, traffic, and road conditions, the model aims to assist emergency services in responding more quickly and efficiently.

## Objectives

- Predict accident severity levels using real accident data.
- Apply advanced feature engineering to enhance model accuracy.
- Utilize XGBoost for efficient and robust classification.
- Provide actionable insights to improve emergency service response.

## Dataset

The dataset consists of real accident records with features including (but not limited to):

- **Weather conditions** (rain, fog, clear, etc.)
- **Road types and surfaces**
- **Traffic density**
- **Time and location**
- **Accident outcomes (severity: minor, serious, fatal)**

*Note: For privacy and compliance, ensure you have the right to use the dataset. This project uses anonymized, publicly available data.*

## Methodology

1. **Data Preprocessing:**  
   - Cleaning missing or inconsistent entries
   - Encoding categorical variables
   - Feature scaling and normalization

2. **Feature Engineering:**  
   - Creating new features based on domain knowledge
   - Selecting the most relevant features for prediction

3. **Model Building:**  
   - Implemented XGBoost classifier for multi-class prediction
   - Hyperparameter tuning to optimize performance
   - Evaluation using accuracy, F1-score, and confusion matrix

4. **Deployment & Use Case:**  
   - Model can be integrated with real-time reporting systems
   - Helps emergency services prioritize and allocate resources

## Results

- The model achieved improved accuracy over baseline methods, demonstrating the effectiveness of XGBoost and feature engineering.
- Key features contributing to severity prediction included weather conditions, time of day, and road type.

## How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/accident-severity-prediction.git
   cd accident-severity-prediction
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Model**
   - Ensure your dataset is placed in the `data/` directory.
   - Run the main script:
     ```bash
     python main.py
     ```

4. **Parameters & Configuration**
   - All hyperparameters can be configured in `config.yaml`.

## Project Structure

```
accident-severity-prediction/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model.py
├── main.py
├── config.yaml
└── README.md
```

## Applications

- **Emergency Response:** Prioritize dispatch and resources based on predicted severity.
- **Urban Planning:** Identify accident hotspots and contributing factors.
- **Public Awareness:** Inform citizens about risk factors on roads.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

---

*This project demonstrates the power of machine learning for real-world, life-saving applications in road safety analytics.*
