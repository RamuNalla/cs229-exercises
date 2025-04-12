import wandb
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # to save the model as a file

# Initialize a new wandb run
wandb.init(
    project="logistic_regression_project",  # Name of the project on wandb.ai
    config={                                # Dictionary of hyperparameters to track
        "C": 0.1,
        "solver": "lbfgs"
    }
)

# Access hyperparameters using wandb.config
config = wandb.config

X, y = make_blobs(n_samples=1000, centers=2, cluster_std=3.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=config.C, solver=config.solver)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

wandb.log({"accuracy": accuracy})

joblib.dump(model, "logistic_model.pkl")
wandb.save("logistic_model.pkl")

# Finish the wandb run
wandb.finish()
