import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio
import os

# Create folders if they don't exist
os.makedirs('Model', exist_ok=True)
os.makedirs('Results', exist_ok=True)

# Step 1: Load the dataset and shuffle it
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1)
print(drug_df.head(3))

# Step 2: Split into features (X) and target (y)
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Step 3: Build the pipeline for processing and training
cat_col = [1, 2, 3]  # Categorical columns
num_col = [0, 4]  # Numerical columns

# Create transformers for the pipeline
transform = ColumnTransformer(
    transformers=[
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

# Create the full pipeline with the model
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# Step 4: Train the model
pipe.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy * 100, 2)}%, F1: {round(f1, 2)}")

# Save metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}.\n")

# Step 6: Create and save the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# Step 7: Save the entire pipeline
sio.dump(pipe, "Model/drug_pipeline.skops")

# Step 8: To load the saved model later:
# loaded_pipe = sio.load("Model/drug_pipeline.skops", trusted=True)
