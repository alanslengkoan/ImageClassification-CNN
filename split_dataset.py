import os, shutil, random

# ============================================================
# KONFIGURASI PATH & KELAS
# ============================================================
BASE_DIR   = '/home/echolog/Documents/Project/www/skripsi/ImageClassification-CNN'
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset_all')
TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')

# ============================================================
# MAPPING 4 KELAS → 3 KELAS
# ============================================================
# Strategi: Gabung 'ringan' + 'sedang' → 'menengah'
# Alasan: Kedua kelas overlap tinggi (F1-score rendah)
# Expected accuracy boost: 70% → 78-85%

CLASSES_SOURCE = ['baik', 'ringan', 'sedang', 'berat']  # Folder di dataset_all
CLASSES_TARGET = ['baik', 'menengah', 'berat']          # Kelas baru (3 kelas)

# Mapping dari source → target
CLASS_MAPPING = {
    'baik':   'baik',
    'ringan': 'menengah',  # Gabung ke menengah
    'sedang': 'menengah',  # Gabung ke menengah
    'berat':  'berat'
}

TRAIN_RATIO = 0.7   # 70% train
VAL_RATIO   = 0.2   # 20% val
TEST_RATIO  = 0.1   # 10% test

random.seed(42)

# ============================================================
# CEK FOLDER SOURCE
# ============================================================
print('📂 Cek folder dataset_all (4 kelas source):')
print('-' * 50)
total_source = 0
semua_ada    = True
class_counts = {}  # Simpan count per kelas source

for cls in CLASSES_SOURCE:
    path = os.path.join(SOURCE_DIR, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[cls] = count
        total_source += count
        status = '✅' if count > 0 else '⚠️  kosong!'
        target = CLASS_MAPPING[cls]
        print(f'   {status} {cls:10s}: {count:3d} gambar → {target}')
    else:
        print(f'   ❌ {cls:10s}: folder tidak ditemukan!')
        semua_ada = False

print(f'\n   {"TOTAL":10s}: {total_source} gambar')
print(f'\n🔄 Mapping ke 3 kelas target:')
for target_cls in CLASSES_TARGET:
    source_classes = [k for k, v in CLASS_MAPPING.items() if v == target_cls]
    total_target = sum([class_counts.get(sc, 0) for sc in source_classes])
    print(f'   {target_cls:10s}: {total_target:3d} gambar ({" + ".join(source_classes)})')

if not semua_ada:
    print('\n❌ Ada folder yang tidak ditemukan!')
    print('   Pastikan struktur folder:')
    print(f'   {SOURCE_DIR}/')
    for cls in CLASSES_SOURCE:
        print(f'   ├── {cls}/')
    raise SystemExit('Hentikan script — perbaiki folder terlebih dahulu.')

# ============================================================
# HAPUS FOLDER LAMA, BUAT ULANG
# ============================================================
print('\n🗑️  Menghapus folder train / val / test lama...')
for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'   ✅ Dihapus : {path}')
    else:
        print(f'   ⚠️  Tidak ada: {path} (skip)')

print('\n📁 Membuat folder train / val / test baru (3 kelas)...')
for cls in CLASSES_TARGET:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR,   cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR,  cls), exist_ok=True)
print(f'   ✅ Folder berhasil dibuat: {CLASSES_TARGET}')

# ============================================================
# BAGI ACAK 70% TRAIN / 20% VAL / 10% TEST
# ============================================================
print('\n📊 Membagi dataset secara acak (70% train / 20% val / 10% test)...')
print('   Strategi: Gabung ringan+sedang → menengah')
print('-' * 60)

total_train = 0
total_val   = 0
total_test  = 0

# Proses setiap kelas source, mapping ke target
for cls_source in CLASSES_SOURCE:
    src         = os.path.join(SOURCE_DIR, cls_source)
    cls_target  = CLASS_MAPPING[cls_source]  # baik/menengah/berat
    
    all_images = [f for f in os.listdir(src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    n          = len(all_images)
    n_train    = int(n * TRAIN_RATIO)
    n_val      = int(n * VAL_RATIO)
    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train + n_val]
    test_imgs  = all_images[n_train + n_val:]

    # Copy ke folder target (bukan source!)
    for img in train_imgs:
        # Rename file untuk avoid collision: baik_001.jpg, ringan_001.jpg → menengah_ringan_001.jpg
        new_name = f"{cls_source}_{img}" if cls_target == 'menengah' else img
        shutil.copy(os.path.join(src, img), os.path.join(TRAIN_DIR, cls_target, new_name))
    for img in val_imgs:
        new_name = f"{cls_source}_{img}" if cls_target == 'menengah' else img
        shutil.copy(os.path.join(src, img), os.path.join(VAL_DIR, cls_target, new_name))
    for img in test_imgs:
        new_name = f"{cls_source}_{img}" if cls_target == 'menengah' else img
        shutil.copy(os.path.join(src, img), os.path.join(TEST_DIR, cls_target, new_name))

    total_train += len(train_imgs)
    total_val   += len(val_imgs)
    total_test  += len(test_imgs)
    print(f'   {cls_source:10s} → {cls_target:10s}: {len(train_imgs):3d} train | {len(val_imgs):3d} val | {len(test_imgs):3d} test')

print('-' * 50)
print(f'   {"TOTAL":10s}: {total_train:3d} train | {total_val:3d} val | {total_test:3d} test')

# ============================================================
# VERIFIKASI HASIL
# ============================================================
print('\n✅ Verifikasi Hasil Pembagian (3 Kelas Target):')
print('-' * 60)
for split, path in [('TRAIN', TRAIN_DIR), ('VAL', VAL_DIR), ('TEST', TEST_DIR)]:
    print(f'\n📁 {split}:')
    split_total = 0
    for cls in CLASSES_TARGET:
        cls_path    = os.path.join(path, cls)
        count       = len(os.listdir(cls_path))
        split_total += count
        print(f'   {cls:10s}: {count} gambar')
    print(f'   {"TOTAL":10s}: {split_total} gambar')

# ============================================================
# RINGKASAN
# ============================================================
total_all = total_train + total_val + total_test
print(f'\n{"=" * 60}')
print(f'🎉 Dataset berhasil dibagi menjadi 3 KELAS!')
print(f'   Strategi : Gabung ringan+sedang → menengah')
print(f'   Kelas    : {CLASSES_TARGET}')
print(f'   Train    : {total_train} gambar ({int(TRAIN_RATIO*100)}%)')
print(f'   Val      : {total_val} gambar ({int(VAL_RATIO*100)}%)')
print(f'   Test     : {total_test} gambar ({int(TEST_RATIO*100)}%)')
print(f'   Total    : {total_all} gambar')
print(f'{"=" * 60}')
print(f'\n📁 Struktur folder hasil:')
print(f'   {BASE_DIR}/dataset/')
print(f'   ├── train/')
for cls in CLASSES_TARGET:
    print(f'   │   ├── {cls}/')
print(f'   ├── val/')
for cls in CLASSES_TARGET:
    print(f'   │   ├── {cls}/')
print(f'   └── test/')
for cls in CLASSES_TARGET:
    print(f'       ├── {cls}/')
print(f'\n✅ Expected Accuracy Boost: 70% → 78-85%')
print(f'➡️  Siap untuk training! Jalankan: python3 resnet50.py')