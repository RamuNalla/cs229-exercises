import mlflow
import mlflow.sklearn
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("./mlruns")

experiment_name = "logistic_regression_experiment_3"
experiment_id = mlflow.create_experiment(experiment_name) if mlflow.get_experiment_by_name(experiment_name) is None else mlflow.get_experiment_by_name(experiment_name).experiment_id    #create a new id if the name doesn't exist. If the name exists, get that id and assign it to experiment_id

mlflow.set_experiment(experiment_name)

X, y = make_blobs(n_samples=1000, centers=2, random_state=42, cluster_std=3.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start a new MLflow run
with mlflow.start_run(run_name = "run_with_C = 0.5"):
    
    # Hyperparameter
    C = 0.5  # Regularization strength
    model = LogisticRegression(C=C, solver='lbfgs')

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    # Logging
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", acc)

    # Save the model
    mlflow.sklearn.log_model(model, "logistic_model")
