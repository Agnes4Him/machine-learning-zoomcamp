import requests

# url = "http://localhost:8080/2015-03-31/functions/function/invocations"    # Local test URL
url = "https://zj3yq2zhua.execute-api.us-east-1.amazonaws.com/dev/predict"           # AWS API Gateway URL

image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

response = requests.post(url, json={"image_url": image_url})

print(response.json())