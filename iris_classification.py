
# CodeAlpha - Iris Flower Classification with User Input

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n======================================")
print("   IRIS FLOWER CLASSIFICATION")
print("======================================")

# Step 1: Load the dataset
data = pd.read_csv("dataset/Iris.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Step 2: Prepare the dataset
if "Id" in data.columns:
    data = data.drop(columns=["Id"])

target_column = next(
    col for col in data.columns
    if col.strip().lower() in ["species", "class", "variety", "target"]
)

X = data.drop(columns=[target_column])
y = data[target_column]

X = X.select_dtypes(include="number")

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 6: Take input from the user
print("\n======================================")
print("     PREDICT YOUR IRIS FLOWER")
print("======================================")
print("Enter flower measurements in centimeters.")

try:
    sepal_length = float(input("Enter Sepal Length: "))
    sepal_width = float(input("Enter Sepal Width: "))
    petal_length = float(input("Enter Petal Length: "))
    petal_width = float(input("Enter Petal Width: "))

    measurements = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]

    if any(value <= 0 for value in measurements):
        print("Error: Measurements must be greater than zero.")

    elif any(value > 10 for value in measurements):
        print("Error: Please enter realistic flower measurements.")

    else:
        # Use the same feature columns as the training data
        input_data = pd.DataFrame(
            [measurements],
            columns=X.columns
        )

        prediction = model.predict(input_data)

        print("\n======================================")
        print("       PREDICTION RESULT")
        print("======================================")
        print("Predicted Flower Species:", prediction[0])

        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100

        print(f"Model confidence: {confidence:.2f}%")

except ValueError:
    print("Error: Please enter valid numeric measurements.")

# Step 7: Save the confusion matrix graph
plt.figure(figsize=(7, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Iris Flower Classification - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.xticks(range(len(model.classes_)), model.classes_, rotation=20)
plt.yticks(range(len(model.classes_)), model.classes_)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

print("\nGraph saved as confusion_matrix.png")
print("\nProject completed successfully!")

plt.show()
