import requests
import json
import os

from dotenv import load_dotenv
load_dotenv()

MAILPIT_API_URL = os.getenv("MAILPIT_API_URL", "http://localhost:8025/")
def send_email(email):
    try:
        resp = requests.post(
            MAILPIT_API_URL + "api/v1/send",
            headers={"Content-Type": "application/json"},
            data=json.dumps(email),
        )
        return resp.status_code, resp.text 
    except Exception as e:
        return 500, str(e)


def get_all_messages():
    try:
        resp = requests.get(
            MAILPIT_API_URL + "api/v1/messages"
        )
        if resp.status_code == 200:
            json_resp = resp.json()
            final_response = []
            for msg in json_resp['messages']:
                final_response.append({
                    "id": msg['ID']
                })
            return 200, final_response
        raise Exception(f"Failed to fetch messages: {resp.status_code} - {resp.text}")
    except Exception as e:
        return 500, str(e)


def get_message(message_id):
    try:
        resp = requests.get(
            MAILPIT_API_URL + "api/v1/message/" + message_id
        )
        if resp.status_code == 200:
            json_resp = resp.json()
            final_response = {
                "id": json_resp['ID'],
                "subject": json_resp['Subject'],
                "text": json_resp['Text'],
                "html": json_resp['HTML'],
                "date": json_resp['Date']
            }
            return 200, final_response
        raise Exception(f"Failed to fetch messages: {resp.status_code} - {resp.text}")
    except Exception as e:
        return 500, str(e)