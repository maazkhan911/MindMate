import os
import json
import requests
import numpy as np
from PIL import Image
from keras.models import load_model
from django.shortcuts import render
from django.http import JsonResponse
import google.auth.transport.requests
from keras.preprocessing import image
from google.oauth2 import service_account
from django.core.files.storage import default_storage
from .emotion_model import detect_emotion  


def index(request):
    return render(request, 'index.html')


def chatbot(request):
    if request.method == 'GET':
        return render(request, 'chatbot.html')

    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            emotion = detect_emotion(user_message)

            creds = service_account.Credentials.from_service_account_file(
                r"C:\Users\HomePC\Desktop\MindMate project\MindMate1\MindMate\mindmate-agent.json", 
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            request_auth = google.auth.transport.requests.Request()
            creds.refresh(request_auth)

            access_token = creds.token

            project_id = "mindmate-agent-drfl"  
            session_id = "test-session"
            language_code = "en"

            url = f"https://dialogflow.googleapis.com/v2/projects/{project_id}/agent/sessions/{session_id}:detectIntent"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "queryInput": {
                    "text": {
                        "text": user_message,
                        "languageCode": language_code
                    }
                }
            }

            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                reply_from_bot = result['queryResult']['fulfillmentText']

                full_reply = f"{reply_from_bot}\n\n🧠 Emotion Detected: *{emotion}*"
                return JsonResponse({'reply': full_reply})
            else:
                return JsonResponse({'reply': f'Error from Dialogflow (status code {response.status_code})'})

        except Exception as e:
            return JsonResponse({'reply': f'Error: {str(e)}'})

    return JsonResponse({'reply': 'Invalid request method'})


model = load_model('emotion_cnn_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def image_upload(request):
    prediction = None
    image_url = None
    error_message = None

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']

        try:
            img = Image.open(img_file).convert('RGB')

            filename = img_file.name
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'   

            save_path = os.path.join('media', filename)
            img.save(save_path)
            image_url = '/' + save_path

            img = img.resize((128, 128))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  

            preds = model.predict(img_array)
            predicted_label = emotion_labels[np.argmax(preds)]
            prediction = predicted_label

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"Error processing image: {str(e)}"
            prediction = "Unable to process image"

    return render(request, 'image_upload.html', {
        'prediction': prediction,
        'image_url': image_url,
        'error_message': error_message
    })
