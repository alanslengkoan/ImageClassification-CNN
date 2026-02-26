from google.colab import drive
drive.mount('/content/drive')

import os

# Cek isi MyDrive
for item in os.listdir('/content/drive/MyDrive'):
    print(item)