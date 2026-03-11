import os, shutil, random

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# KONFIGURASI PATH & KELAS
# ============================================================
SOURCE_DIR  = '/content/drive/MyDrive/dataset_all'   # semua gambar
TRAIN_DIR   = '/content/drive/MyDrive/dataset/train' # output train
TEST_DIR    = '/content/drive/MyDrive/dataset/test'  # output test

# ✅ Nama folder kelas yang baru
CLASSES     = ['baik', 'sedang', 'ringan', 'berat']
TRAIN_RATIO = 0.8  # 80% train, 20% test

random.seed(42)

# ============================================================
# CEK FOLDER SOURCE
# ============================================================
print('📂 Cek folder dataset_all:')
print('-' * 45)
total_source = 0
semua_ada = True
for cls in CLASSES:
    path  = os.path.join(SOURCE_DIR, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_source += count
        status = '✅' if count > 0 else '⚠️  kosong!'
        print(f'   {status} {cls:10s}: {count} gambar')
    else:
        print(f'   ❌ {cls:10s}: folder tidak ditemukan!')
        semua_ada = False

print(f'\n   {"TOTAL":10s}: {total_source} gambar')

if not semua_ada:
    print('\n❌ Ada folder yang tidak ditemukan!')
    print('   Pastikan struktur di Google Drive:')
    print('   MyDrive/dataset_all/')
    for cls in CLASSES:
        print(f'   ├── {cls}/')
    raise SystemExit('Hentikan script — perbaiki folder terlebih dahulu.')

# ============================================================
# HAPUS FOLDER TRAIN & TEST LAMA, BUAT ULANG
# ============================================================
print('\n🗑️  Menghapus folder train & test lama...')
for path in [TRAIN_DIR, TEST_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'   ✅ Dihapus : {path}')
    else:
        print(f'   ⚠️  Tidak ada: {path} (skip)')

print('\n📁 Membuat folder train & test baru...')
for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR,  cls), exist_ok=True)
print('   ✅ Semua folder berhasil dibuat')

# ============================================================
# BAGI ACAK 80% TRAIN / 20% TEST
# ============================================================
print('\n📊 Membagi dataset secara acak (80% train / 20% test)...')
print('-' * 45)

total_train = 0
total_test  = 0

for cls in CLASSES:
    src        = os.path.join(SOURCE_DIR, cls)
    all_images = [f for f in os.listdir(src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    split      = int(len(all_images) * TRAIN_RATIO)
    train_imgs = all_images[:split]
    test_imgs  = all_images[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(src, img),
                    os.path.join(TRAIN_DIR, cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src, img),
                    os.path.join(TEST_DIR, cls, img))

    total_train += len(train_imgs)
    total_test  += len(test_imgs)
    print(f'   {cls:10s}: {len(train_imgs):3d} train | {len(test_imgs):3d} test')

print('-' * 45)
print(f'   {"TOTAL":10s}: {total_train:3d} train | {total_test:3d} test')

# ============================================================
# VERIFIKASI HASIL
# ============================================================
print('\n✅ Verifikasi Hasil Pembagian:')
print('-' * 45)
for split, path in [('TRAIN', TRAIN_DIR), ('TEST', TEST_DIR)]:
    print(f'\n📁 {split}:')
    split_total = 0
    for cls in CLASSES:
        cls_path = os.path.join(path, cls)
        count    = len(os.listdir(cls_path))
        split_total += count
        print(f'   {cls:10s}: {count} gambar')
    print(f'   {"TOTAL":10s}: {split_total} gambar')

# ============================================================
# RINGKASAN
# ============================================================
print(f'\n{"=" * 45}')
print(f'🎉 Dataset berhasil dibagi!')
print(f'   Kelas  : {CLASSES}')
print(f'   Train  : {total_train} gambar ({int(TRAIN_RATIO*100)}%)')
print(f'   Test   : {total_test} gambar ({int((1-TRAIN_RATIO)*100)}%)')
print(f'{"=" * 45}')
print(f'\n📁 Struktur folder hasil:')
print(f'   MyDrive/dataset/')
print(f'   ├── train/')
for cls in CLASSES:
    print(f'   │   ├── {cls}/')
print(f'   └── test/')
for cls in CLASSES:
    print(f'       ├── {cls}/')
print(f'\n➡️  Siap untuk training! Jalankan resnet50_phase1.py')