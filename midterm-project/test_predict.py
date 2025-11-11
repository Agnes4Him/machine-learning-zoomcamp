import requests

#url = 'http://localhost:8000/predict'
url = 'https://pnr8h8v3pi.us-east-1.awsapprunner.com/predict'

teen = {
    'Gender': 'Female',
    'Location': 'Cherylburgh',
    'School_Grade': '12th',
    'Phone_Usage_Purpose': 'Other',
    'Age': 13,
    'Daily_Usage_Hours': 4.9,
    'Sleep_Hours': 8.0,
    'Academic_Performance': 74,
    'Social_Interactions': 5,
    'Exercise_Hours': 0.6,
    'Anxiety_Level': 4,
    'Depression_Level': 3,
    'Self_Esteem': 8,
    'Parental_Control': 1,
    'Screen_Time_Before_Bed': 0.6,
    'Phone_Checks_Per_Day': 84,
    'Apps_Used_Daily': 14,
    'Time_on_Social_Media': 3.1,
    'Time_on_Gaming': 0.6,
    'Time_on_Education': 0.8,
    'Family_Communication': 6,
    'Weekend_Usage_Hours': 3.5
}

try:
    response = requests.post(url, json=teen)
    response.raise_for_status() 

    prediction = response.json()

    print(f"Addiction level is {prediction['addiction_level']:.2f}.")
    print(f"Hence, addiction to smartphone is {prediction['addiction_category']}.")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
except KeyError:
    print("Unexpected response format from server.")
except Exception as e:
    print(f"An error occurred: {e}")