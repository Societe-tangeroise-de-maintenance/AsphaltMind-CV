import requests
from core.settings import settings

url_send_incidents = settings.server_url_incidents
url_get_users = settings.server_url_users

headers = {
    "accept": "application/json",
    "Authorization": settings.server_api_token,
    "Content-Type": "application/json"
}


def send_incident(type, time, img, location, personnel):
    data = {
        "type": type,
        "time": time,
        "img": str(img),
        "location": location,
        "personnel": personnel
    }

    response = requests.post(url_send_incidents, headers=headers, json=data)

    print(response.json())


def get_all_users():

    response = requests.get(url_get_users, headers=headers)

    print(response.json())
    return response.json()


# data = {
#     "type": "test",
#     "time": "test",
#     "img": "string"
# }

# response = requests.post(url, headers=headers, json=data)

# print(response.json())