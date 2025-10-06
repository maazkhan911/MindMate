# from transformers import pipeline

# emotion_classifier = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base",
#     return_all_scores=True
# )

# def detect_emotion(text):
#     results = emotion_classifier(text)[0]
#     sorted_result = sorted(results, key=lambda x: x['score'], reverse=True)
#     return sorted_result[0]['label']

from transformers import pipeline

# Load once globally
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Smart detection with emoji fallback
def detect_emotion(text):
    emoji_map = {
        "😡": "anger",
        "😊": "joy",
        "😭": "sadness",
        "😱": "fear",
        "😖": "disgust",
        "😅": "neutral"
    }

    for emoji, emotion in emoji_map.items():
        if emoji in text:
            return emotion

    try:
        result = emotion_classifier(text)
        return result[0][0]['label'].lower()
    except:
        return "neutral"
