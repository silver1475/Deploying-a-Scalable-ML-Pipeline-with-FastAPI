import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
model_dir = os.path.join(project_path, "model")

os.makedirs(model_dir, exist_ok=True)

data = pd.read_csv(data_path)
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = train_model(X_train, y_train)

save_model(model, os.path.join(model_dir, "model.pkl"))
save_model(encoder, os.path.join(model_dir, "encoder.pkl"))
save_model(lb, os.path.join(model_dir, "lb.pkl"))

model = load_model(os.path.join(model_dir, "model.pkl"))
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print("Overall Metrics:")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

output_file = "slice_output.txt"
if os.path.exists(output_file):
    os.remove(output_file)

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model,
        )

        with open(output_file, "a") as f:
            f.write(f"Feature: {col} | Value: {slicevalue} | Count: {count}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
            f.write("-" * 30 + "\n")

print(f"Slice performance details saved to {output_file}")
