import requests

url = 'http://localhost:8000/predict'

teen = {'Gender': 'Female',
             'Location': 'Cherylburgh',
             'School_Grade': '12th',
             'Phone_Usage_Purpose': 'Other',
             'Age': 13,
             'Daily_Usage_Hours': 2.0,
             'Sleep_Hours': 10.0,
             'Academic_Performance': 74,
             'Social_Interactions': 8,
             'Exercise_Hours': 0.6,
             'Anxiety_Level': 4,
             'Depression_Level': 3,
             'Self_Esteem': 2,
             'Parental_Control': 0,
             'Screen_Time_Before_Bed': 0.2,
             'Phone_Checks_Per_Day': 84,
             'Apps_Used_Daily': 20,
             'Time_on_Social_Media': 3.1,
             'Time_on_Gaming': 0.6,
             'Time_on_Education': 0.8,
             'Family_Communication': 6,
             'Weekend_Usage_Hours': 2.5
        }

response = requests.post(url, json=teen)

prediction = response.json()

print(f"Addiction level is {prediction['addiction_level']}. Hence, addiction to smartphone is {prediction["addiction_category"]}")

