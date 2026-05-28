from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import load_dataset


def train_baseline_model():
    X, y, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test, target_names


def main():
    model, X_test, y_test, target_names = train_baseline_model()

    sample = X_test.iloc[[0]]
    actual_label = y_test.iloc[0]

    predicted_label = model.predict(sample)[0]
    predicted_probability = model.predict_proba(sample)[0]

    print("Single Sample Prediction")
    print("------------------------")
    print(f"Actual class: {target_names[actual_label]}")
    print(f"Predicted class: {target_names[predicted_label]}")
    print(f"Probability for malignant: {predicted_probability[0]:.4f}")
    print(f"Probability for benign: {predicted_probability[1]:.4f}")


if __name__ == "__main__":
    main()