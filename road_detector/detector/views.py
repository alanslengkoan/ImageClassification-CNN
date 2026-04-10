import os
import uuid
import numpy as np

from django.conf import settings
from django.shortcuts import render

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Model di-load sekali saat pertama kali dipakai (lazy load)
_model = None

CLASS_NAMES = ['baik', 'berat', 'sedang']   # urutan alphabetical ImageDataGenerator

CLASS_INFO = {
    'baik'  : {'label': 'Baik',   'color': 'green',  'icon': '🟢',
               'desc': 'Kondisi jalan baik, tidak terdeteksi kerusakan signifikan.'},
    'sedang': {'label': 'Sedang', 'color': 'yellow', 'icon': '🟡',
               'desc': 'Terdapat kerusakan ringan hingga sedang pada permukaan jalan.'},
    'berat' : {'label': 'Berat',  'color': 'red',    'icon': '🔴',
               'desc': 'Kerusakan berat terdeteksi, perlu perbaikan segera.'},
}


def _get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load

        model_path = str(settings.MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Model tidak ditemukan: {model_path}\n'
                'Jalankan resnet50.py terlebih dahulu untuk melatih model.'
            )
        _model = keras_load(model_path)
    return _model


def _preprocess(image_file) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def _save_upload(image_file) -> str:
    ext      = os.path.splitext(image_file.name)[-1].lower() or '.jpg'
    filename = f'{uuid.uuid4().hex}{ext}'
    save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'wb') as f:
        for chunk in image_file.chunks():
            f.write(chunk)
    return f'{settings.MEDIA_URL}uploads/{filename}'


def index(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        try:
            image_url = _save_upload(image_file)
            image_file.seek(0)

            model  = _get_model()
            tensor = _preprocess(image_file)
            probs  = model.predict(tensor, verbose=0)[0]

            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            info       = CLASS_INFO[pred_class]

            probabilities = [
                {
                    'class': cls,
                    'label': CLASS_INFO[cls]['label'],
                    'color': CLASS_INFO[cls]['color'],
                    'value': round(float(probs[i]) * 100, 2),
                    'width': round(float(probs[i]) * 100, 1),
                }
                for i, cls in enumerate(CLASS_NAMES)
            ]
            probabilities.sort(key=lambda x: x['value'], reverse=True)

            context = {
                'result'        : True,
                'image_url'     : image_url,
                'pred_class'    : pred_class,
                'pred_label'    : info['label'],
                'pred_color'    : info['color'],
                'pred_icon'     : info['icon'],
                'pred_desc'     : info['desc'],
                'confidence'    : round(float(probs[pred_idx]) * 100, 2),
                'probabilities' : probabilities,
            }

        except FileNotFoundError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f'Terjadi kesalahan saat memproses gambar: {e}'

    return render(request, 'detector/index.html', context)
