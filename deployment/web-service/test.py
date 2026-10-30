import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}


uri = "http://localhost:9696/predict"

response = requests.post(uri,json = ride)

print(response.status_code)
print(response.json())

