import os
import numpy as np
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST


# 1. The view to render the HTML template
def home(request):
    return render(request, 'home.html')


# Lazy load the model
_model = None


def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        # Make sure this path matches where your .keras file actually is
        model_path = "../models/InceptionResNetV2A75L1.keras"
        print(f"--- Loading model from: {model_path} ---")
        _model = load_model(model_path)
    return _model


# 2. The API endpoint to handle the image prediction
@require_POST
def predict_fracture(request):
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'success': False, 'error': 'No image provided'}, status=400)

        # Preprocessing (matches your training script)
        img = Image.open(image_file).convert('RGB')
        img = img.resize((299, 299))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        model = get_model()
        prediction = model.predict(img_array, verbose=0)

        # Get class index (0 or 1) and map it
        predicted_class_index = int(np.argmax(prediction[0]))

        # Extract probabilities (Softmax output)
        prob_not_fractured = float(prediction[0][0])
        prob_fractured = float(prediction[0][1])

        is_fracture = (predicted_class_index == 1)
        confidence = max(prob_fractured, prob_not_fractured) * 100

        return JsonResponse({
            'success': True,
            'is_fracture': is_fracture,
            'confidence': round(confidence, 1),
            'fracture_prob': round(prob_fractured * 100, 1),
            'normal_prob': round(prob_not_fractured * 100, 1),
        })

    except Exception as e:
        print(f"--- PREDICTION ERROR: {str(e)} ---")  # This will show in your terminal
        return JsonResponse({'success': False, 'error': str(e)}, status=500)