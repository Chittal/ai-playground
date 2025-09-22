import json
import requests

emails = []
with open("mailpit/mails.json", "r") as f:
    emails = json.load(f)

for email in emails:
    resp = requests.post(
        "http://localhost:8025/api/v1/send",
        headers={"Content-Type": "application/json"},
        data=json.dumps(email),
    )
    print("Status Code:", resp.status_code)
