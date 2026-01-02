import requests


url = "http://localhost:8000/predict"

household = {
                "appliance_type": "Oven",
                "season": "Winter",
                "outdoor_temperature": 5.3,
                "household_size": 4,
                "hour": 2,
                "day_of_week": 3,
                "day": 12,
                "month": 1,
                "is_weekend": 0
            }

try:
    response = requests.post(url, json=household)
    response.raise_for_status() 

    prediction = response.json()

    print(f"Energy consumption is {prediction['energy_consumption']:.2f} kWh.")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
except KeyError:
    print("Unexpected response format from server.")
except Exception as e:
    print(f"An error occurred: {e}")