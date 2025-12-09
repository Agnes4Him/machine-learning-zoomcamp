import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

response = requests.post(url, json={"image_url": image_url})

print(response.json())