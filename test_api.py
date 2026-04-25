import requests

url = "http://127.0.0.1:5000/analyze"
files = {'file': ('test.jpg', b'dummy content', 'image/jpeg')}
try:
    response = requests.post(url, files=files)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Request failed:", e)
