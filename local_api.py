import json
import requests

# 1. Test the GET root endpoint
url_get = "http://127.0.0.1:8000"
r_get = requests.get(url_get)

# Print the status code and the welcome message
print(f"GET Status Code: {r_get.status_code}")
print(f"GET Welcome Message: {r_get.json()['message']}")

print("-" * 20)

# 2. Test the POST inference endpoint
url_post = "http://127.0.0.1:8000/data/"
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send the POST request
# Note: requests.post(..., json=data) automatically handles JSON encoding and headers
r_post = requests.post(url_post, json=data)

# Print the status code and the inference result
print(f"POST Status Code: {r_post.status_code}")
print(r_post.text)
