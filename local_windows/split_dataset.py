import os, shutil, random

# ============================================================
# KONFIGURASI PATH & KELAS
# ============================================================
BASE_DIR   = r'C:\ImageClassification-CNN'   # ✅ ganti sesuai lokasi kamu
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset_all')
TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')

# ============================================================
# MAPPING 4 KELAS → 3 KELAS
# ============================================================
# Strategi: Gabung 'ringan' + 'sedang' → 'sedang'
CLASSES_SOURCE = ['baik', 'ringan', 'sedang', 'berat']
CLASSES_TARGET = ['baik', 'sedang', 'berat']

CLASS_MAPPING = {
    'baik':   'baik',
    'ringan': 'sedang',
    'sedang': 'sedang',
    'berat':  'berat'
}

TRAIN_RATIO = 0.7   # 70% train
VAL_RATIO   = 0.2   # 20% val
TEST_RATIO  = 0.1   # 10% test

random.seed(42)

# ============================================================
# CEK FOLDER SOURCE
# ============================================================
print('📂 Cek folder dataset_all:')
print('-' * 50)

# Jika ringan tidak ada, skip ringan
if not os.path.exists(os.path.join(SOURCE_DIR, 'ringan')):
    print('   ⚠️  Folder ringan tidak ditemukan — sedang akan dipetakan langsung ke sedang')
    CLASSES_SOURCE = ['baik', 'sedang', 'berat']
    CLASS_MAPPING  = {'baik': 'baik', 'sedang': 'sedang', 'berat': 'berat'}

total_source = 0
class_counts = {}

for cls in CLASSES_SOURCE:
    path = os.path.join(SOURCE_DIR, cls)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[cls] = count
        total_source += count
        status = '✅' if count > 0 else '⚠️  kosong!'
        print(f'   {status} {cls:10s}: {count} gambar')
    else:
        print(f'   ❌ {cls:10s}: folder tidak ditemukan!')
        raise SystemExit(f'Hentikan — folder {cls} tidak ada di {SOURCE_DIR}')

print(f'\n   {"TOTAL":10s}: {total_source} gambar')

# Hitung total per kelas target
target_totals = {}
for cls_src, count in class_counts.items():
    tgt = CLASS_MAPPING[cls_src]
    target_totals[tgt] = target_totals.get(tgt, 0) + count

print(f'\n⚖️  Distribusi per kelas target (tanpa cap — class weights handle imbalance):')
for tgt, total in target_totals.items():
    print(f'   {tgt:10s}: {total:3d} gambar total')

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
print(f'   Strategi: ringan+sedang → sedang')
print('-' * 60)

total_train = 0
total_val   = 0
total_test  = 0

for cls_source in CLASSES_SOURCE:
    src        = os.path.join(SOURCE_DIR, cls_source)
    cls_target = CLASS_MAPPING[cls_source]

    all_images = [f for f in os.listdir(src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    n          = len(all_images)
    n_train    = int(n * TRAIN_RATIO)
    n_val      = int(n * VAL_RATIO)
    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train + n_val]
    test_imgs  = all_images[n_train + n_val:]

    # Prefix filename jika merge dari sumber berbeda ke sedang
    for img in train_imgs:
        new_name = f"{cls_source}_{img}" if cls_target == 'sedang' else img
        shutil.copy(os.path.join(src, img), os.path.join(TRAIN_DIR, cls_target, new_name))
    for img in val_imgs:
        new_name = f"{cls_source}_{img}" if cls_target == 'sedang' else img
        shutil.copy(os.path.join(src, img), os.path.join(VAL_DIR, cls_target, new_name))
    for img in test_imgs:
        new_name = f"{cls_source}_{img}" if cls_target == 'sedang' else img
        shutil.copy(os.path.join(src, img), os.path.join(TEST_DIR, cls_target, new_name))

    total_train += len(train_imgs)
    total_val   += len(val_imgs)
    total_test  += len(test_imgs)
    print(f'   {cls_source:10s} → {cls_target:10s}: {len(train_imgs):3d} train | {len(val_imgs):3d} val | {len(test_imgs):3d} test')

print('-' * 60)
print(f'   {"TOTAL":10s}: {total_train:3d} train | {total_val:3d} val | {total_test:3d} test')

# ============================================================
# VERIFIKASI HASIL
# ============================================================
print('\n✅ Verifikasi Hasil Pembagian:')
print('-' * 50)
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
print(f'\n{"=" * 50}')
print(f'🎉 Dataset berhasil dibagi!')
print(f'   Kelas Target : {CLASSES_TARGET}')
print(f'   Strategi     : ringan+sedang → sedang')
print(f'   Train  : {total_train} gambar ({int(TRAIN_RATIO*100)}%)')
print(f'   Val    : {total_val} gambar ({int(VAL_RATIO*100)}%)')
print(f'   Test   : {total_test} gambar ({int(TEST_RATIO*100)}%)')
print(f'   Total  : {total_all} gambar')
print(f'{"=" * 50}')
print(f'\n📁 Struktur folder hasil:')
print(f'   dataset/')
print(f'   ├── train/')
for cls in CLASSES_TARGET:
    print(f'   │   ├── {cls}/')
print(f'   ├── val/')
for cls in CLASSES_TARGET:
    print(f'   │   ├── {cls}/')
print(f'   └── test/')
for cls in CLASSES_TARGET:
    print(f'       ├── {cls}/')
print(f'\n➡️  Siap untuk training! Jalankan resnet50.py')