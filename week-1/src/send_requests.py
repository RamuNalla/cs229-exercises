import requests

url_linear = "http://127.0.0.1:8000/predict_linear"
url_logistic = "http://127.0.0.1:8000/predict_logistic"

linear_payload = {
    "x": 10
}

logistic_payload = {
    "x1": -1.5,
    "x2": -0.8
}

response_linear = requests.post(url_linear, json=linear_payload)
print("Status code:", response_linear.status_code)
print("Linear Regression Prediction: ", response_linear.json())

response_logistic = requests.post(url_logistic, json=logistic_payload)
print("Status code:", response_logistic.status_code)
print("Logistic Regression Prediction: ", response_logistic.json())