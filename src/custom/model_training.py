import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def train_model(data, *args, **kwargs):
    """
    Train a Logistic Regression model with StandardScaler and log it to MLflow.
    """
    # Unpack input data
    X_train, X_test, y_train, y_test = data

    # ✅ Set MLflow Tracking URI first
    mlflow.set_tracking_uri("http://localhost:5000")

    # ✅ Ensure experiment exists or create it
    experiment_name = "Cancer Prediction Model"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_ID = mlflow.create_experiment(experiment_name)
    else:
        experiment_ID = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"Using Experiment ID: {experiment_ID}")

    # ✅ Start an MLflow run
    with mlflow.start_run(run_name="Logistic Regression"):
        # Preprocessing: Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("MLflow Tracking URI:", mlflow.get_tracking_uri())

        # Model training
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Model evaluation
        y_pred = model.predict(X_test_scaled)

        # Generate Classification Report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Extract accuracy, precision, recall, and f1-score
        accuracy = class_report["accuracy"]
        precision = class_report["weighted avg"]["precision"]
        recall = class_report["weighted avg"]["recall"]
        f1_score = class_report["weighted avg"]["f1-score"]

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)

        # Log the trained model and scaler
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        mlflow.sklearn.log_model(scaler, "scaler")

        print(f"Model logged to MLflow with accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

    return {'model': model, 'scaler': scaler}


@test
def test_output(output, *args) -> None:
    """
    Tests the presence of the trained model and scaler in the output.
    """
    assert output is not None, 'The output is undefined'
    assert 'model' in output, 'Model is missing in output'
    assert 'scaler' in output, 'Scaler is missing in output'
