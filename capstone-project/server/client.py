import requests


url = "http://localhost:8000/predict"

household = {
                "appliance_type": "Air Conditioning",
                "season": "Summer",
                "outdoor_temperature": 12,
                "household_size": 4,
                "hour": 4,
                "day_of_week": 5,
                "day": 12,
                "month": 1,
                "is_weekend": 1
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