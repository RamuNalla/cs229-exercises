import requests

url = "http://127.0.0.1:8000/predict_linear"

data = {
    "x": 6.0
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response: ", response.json())