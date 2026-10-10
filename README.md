# Iris Flower Classification 🌸

## Project Overview
This project uses Machine Learning to classify Iris flowers into three species based on their sepal and petal measurements.

This project was developed as part of my CodeAlpha Data Science Internship task.

## Objectives
- Load and explore the Iris dataset.
- Train a Machine Learning model using Random Forest Classifier.
- Predict flower species based on user-provided measurements.
- Evaluate model performance using accuracy and a confusion matrix.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
The dataset contains 150 records and four numerical features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The model classifies flowers into three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Machine Learning Model
**Algorithm:** Random Forest Classifier

The dataset is divided into:
- Training data: 120 samples (80%)
- Testing data: 30 samples (20%)

## Results
- **Test Accuracy:** 90%
- **Evaluation:** Classification report and confusion matrix
- **Prediction:** The user can enter flower measurements to predict the species.

Note: Accuracy is based on the current train-test split and may vary with different data or model settings.

## Project Structure
```text
CodeAlpha_IrisClassification/
│
├── dataset/
│   └── Iris.csv
│
├── iris_classification.py
├── confusion_matrix.png
├── Iris_Classification_Demo.mp4
└── README.md
```

## How to Run the Project

1. Install Python 3.
2. Install the required libraries:

   ```bash
   pip install pandas matplotlib scikit-learn
   ```

3. Open the terminal in the project folder.
4. Run the program:

   ```bash
   python iris_classification.py
   ```

5. Enter the four flower measurements when prompted.
6. View the predicted species and model evaluation results.

## Learning Outcomes
Through this project, I gained practical experience in:

- Data loading and preprocessing
- Machine Learning model training
- Classification and prediction
- Model evaluation using accuracy and a confusion matrix
- Working with Python data science libraries

## Internship
**CodeAlpha — Data Science Internship**