import os
from fastapi import FastAPI
from dotenv import load_dotenv
import requests
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import json
import uuid
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Return learning path for account 1, and generate new images for units if needed
@app.get("/learning-path")
def get_learning_path_endpoint():
    return get_learning_path(x_session_token=os.getenv("X_SESSION_TOKEN"))

# Return learning path for account 2, and NOT generate new images for units
@app.get("/learning-path-2")
def get_learning_path_endpoint():
    return get_learning_path(x_session_token=os.getenv("X_SESSION_TOKEN_2"))

def get_image_from_title(title):
    vector = model.encode(title)
    res = client.search(
        collection_name="titles",
        query_vector=vector,
        limit=1,
        with_payload=True,
    )
    if len(res) > 0:
        score = res[0].score
        print('Vector search score for ', title, ' is ', score)
        if score > float(os.getenv("TITLE_SEARCHING_THRESHOLD")):
            return res[0].payload['image'] or None
    return None

def generate_image(title):
    url = f'{os.getenv("TOGETHER_XYZ_URL")}/inference'
    print('Generating a new image for the title: ', title)
    payload = json.dumps({
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": title,
        "negative_prompt": "",
        "width": 1024,
        "height": 768,
        "steps": 4,
        "n": 1,
        "response_format": "b64_json"
    })
    headers = {
        'Authorization': f'Bearer {os.getenv("TOGETHER_XYZ_API_KEY")}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print('debug00', response.json())
    return 'data:image/jpeg;base64,' +response.json()['output']['choices'][0]['image_base64']

def save_title(title, image_url):
    vector = model.encode(title)
    client.upsert(
        collection_name="titles",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                payload={
                    "title": title,
                    "image": image_url,
                    "type": "title"
                },
                vector=vector,
            ),
        ],
    )


def get_learning_path(x_session_token):
    url = f'{os.getenv("USER_LEARNING_URL")}/v2.1/learning-paths'
    payload = {}
    headers = {
        'x-session-token': x_session_token
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    json_response = response.json()
    new_children = json_response['children'][:2]

    titles = [item['name'] for item in new_children]
    images = []

    for title in titles:
        image = get_image_from_title(title)
        if image:
            images.append(image)
        else:
            image = generate_image(title)
            images.append(image)
            save_title(title, image)

    # Due to free tier limit, we only generate maximum of 2 images
    for i in range(len(new_children)):
        new_children[i]['image'] = images[i] if len(images) > i else None

    json_response['children'] = new_children

    return json_response
