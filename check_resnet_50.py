import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os, warnings
warnings.filterwarnings('ignore')

print('✅ TensorFlow version :', tf.__version__)
print('✅ GPU tersedia       :', tf.config.list_physical_devices('GPU'))

# ============================================================
# PATH DATASET DARI GOOGLE DRIVE
# ============================================================
BASE_DIR   = '/content/drive/MyDrive/dataset'
TRAIN_DIR  = '/content/drive/MyDrive/dataset/train'
TEST_DIR   = '/content/drive/MyDrive/dataset/test'
OUTPUT_DIR = '/content/drive/MyDrive/dataset'  # output disimpan di sini

IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
EPOCHS           = 30
NUM_CLASSES      = 4
VALIDATION_SPLIT = 0.2

CLASS_NAMES  = ['baik', 'rusak_berat', 'rusak_ringan', 'rusak_sedang']
CLASS_LABELS = ['Baik', 'Rusak Berat', 'Rusak Ringan', 'Rusak Sedang']

# Verifikasi path
print('📂 Verifikasi Path:')
for name, path in [('BASE_DIR ', BASE_DIR), ('TRAIN_DIR', TRAIN_DIR), ('TEST_DIR ', TEST_DIR)]:
    status = '✅ ditemukan' if os.path.exists(path) else '❌ tidak ditemukan!'
    print(f'   {name} : {path} → {status}')

print('📊 Distribusi Dataset dari Google Drive:')
print('-' * 50)
total_all = 0
for split, path in [('TRAIN', TRAIN_DIR), ('TEST', TEST_DIR)]:
    if os.path.exists(path):
        print(f'\n📁 {split}:')
        split_total = 0
        for cls in sorted(os.listdir(path)):
            cls_path = os.path.join(path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                split_total += count
                print(f'   {cls:15s}: {count} gambar')
        print(f'   {"TOTAL":15s}: {split_total} gambar')
        total_all += split_total
    else:
        print(f'❌ {split} → folder tidak ditemukan!')

print(f'\n✅ TOTAL KESELURUHAN : {total_all} gambar')

if os.path.exists(TRAIN_DIR):
    total_train = sum([len(f) for r, d, f in os.walk(TRAIN_DIR)])
    print(f'\n📊 Estimasi Pembagian dari folder train:')
    print(f'   Training (80%) : ~{int(total_train * 0.8)} gambar')
    print(f'   Validasi (20%) : ~{int(total_train * 0.2)} gambar  ← otomatis')

# Train → Augmentasi + preprocessing ResNet-50
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Test → hanya preprocessing, TANPA augmentasi
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generator TRAIN → 80%
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Generator VAL → 20% otomatis
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Generator TEST
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print('\n📊 Ringkasan Pembagian Data:')
print(f'   Train (80%) : {train_generator.samples} gambar')
print(f'   Val   (20%) : {val_generator.samples} gambar  ← otomatis')
print(f'   Test        : {test_generator.samples} gambar')
print(f'\n   Kelas : {train_generator.class_indices}')

# --- 6A: Sample Gambar ---
viz_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=8,
    class_mode='categorical', shuffle=True
)
sample_imgs, sample_lbls = next(viz_gen)
cls_list = list(viz_gen.class_indices.keys())

plt.figure(figsize=(14, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(sample_imgs[i])
    label = cls_list[np.argmax(sample_lbls[i])].replace('_', ' ').title()
    plt.title(label, fontsize=10, fontweight='bold')
    plt.axis('off')
plt.suptitle('Sample Dataset — Klasifikasi Kerusakan Jalan', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_dataset_jalan.png'), dpi=150, bbox_inches='tight')
plt.show()

# --- 6B: Distribusi Kelas ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#4CAF50', '#F44336', '#FF9800', '#FF5722']
for ax, split, path in zip(axes, ['Train', 'Test'], [TRAIN_DIR, TEST_DIR]):
    counts = {}
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            counts[cls.replace('_', ' ').title()] = len(os.listdir(cls_path))
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'Distribusi Kelas — {split}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Kelas Kerusakan'); ax.set_ylabel('Jumlah Gambar')
    ax.grid(axis='y', alpha=0.3); ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribusi_kelas_jalan.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik tersimpan di Google Drive: {OUTPUT_DIR}')

def build_resnet50_road(input_shape=(224, 224, 3), num_classes=4):
    """
    ResNet-50 Transfer Learning — Klasifikasi Kerusakan Jalan
    Base  : ResNet-50 pretrained ImageNet (frozen di Phase 1)
    Head  : Custom classifier 4 kelas kerusakan jalan
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs  = keras.Input(shape=input_shape, name='input_jalan')
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name='gap')(x)
    x       = layers.Dense(512, activation='relu', name='fc_512')(x)
    x       = layers.BatchNormalization(name='bn_1')(x)
    x       = layers.Dropout(0.5, name='dropout_1')(x)
    x       = layers.Dense(256, activation='relu', name='fc_256')(x)
    x       = layers.BatchNormalization(name='bn_2')(x)
    x       = layers.Dropout(0.3, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output_kerusakan')(x)

    model = keras.Model(inputs, outputs, name='ResNet50_Klasifikasi_Jalan')
    return model, base_model

resnet_model, resnet_base = build_resnet50_road()
resnet_model.summary()

trainable = sum([tf.size(w).numpy() for w in resnet_model.trainable_weights])
frozen    = sum([tf.size(w).numpy() for w in resnet_model.non_trainable_weights])
print(f'\n📊 Total Parameter   : {resnet_model.count_params():,}')
print(f'   Trainable (aktif) : {trainable:,}')
print(f'   Frozen (dikunci)  : {frozen:,}')

resnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print('✅ Model berhasil dikompilasi')
print('   Optimizer : Adam (lr=0.001)')
print('   Loss      : Categorical Crossentropy')
print('   Metrics   : Accuracy')

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_jalan_phase1_best.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]
print('✅ Callbacks Phase 1 siap')
print(f'   Model terbaik disimpan ke: {OUTPUT_DIR}')

print('=' * 60)
print('🚀 PHASE 1: Training dengan Base ResNet-50 Frozen')
print('   Train : 80% dari dataset/train/')
print('   Val   : 20% dari dataset/train/ (otomatis)')
print('=' * 60)

history_phase1 = resnet_model.fit(
    train_generator, epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_phase1, verbose=1
)

best_p1 = max(history_phase1.history['val_accuracy'])
print(f'\n✅ Phase 1 Selesai! Best Val Accuracy: {best_p1*100:.2f}%')

# Unfreeze 30 layer terakhir ResNet-50
resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False

resnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_jalan_phase2_best.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8, verbose=1)
]

print('=' * 60)
print('🔓 PHASE 2: Fine-tuning (30 layer terakhir di-unfreeze)')
print('   Learning Rate diturunkan → 1e-5')
print('=' * 60)

history_phase2 = resnet_model.fit(
    train_generator, epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_phase2, verbose=1
)

best_p2 = max(history_phase2.history['val_accuracy'])
print(f'\n✅ Phase 2 Selesai! Best Val Accuracy: {best_p2*100:.2f}%')

def plot_history(history, title, save_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax1.plot(history.history['accuracy'],     label='Train', color='#2196F3', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val',   color='#FF5722', linewidth=2, linestyle='--')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)
    ax2.plot(history.history['loss'],     label='Train', color='#2196F3', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val',   color='#FF5722', linewidth=2, linestyle='--')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{save_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅ Tersimpan di Google Drive: {save_path}')

plot_history(history_phase1, 'ResNet-50 Phase 1 — Kerusakan Jalan (Frozen Base)', 'resnet50_jalan_history_p1')
plot_history(history_phase2, 'ResNet-50 Phase 2 — Kerusakan Jalan (Fine-tuning)', 'resnet50_jalan_history_p2')

print('=' * 60)
print('📈 Evaluasi ResNet-50 pada Test Set')
print('   Data : dataset/test/ (belum pernah dilihat model)')
print('=' * 60)
test_loss, test_acc = resnet_model.evaluate(test_generator, verbose=1)
print(f'\n🏆 Test Accuracy : {test_acc*100:.2f}%')
print(f'   Test Loss     : {test_loss:.4f}')

test_generator.reset()
y_pred_prob      = resnet_model.predict(test_generator, verbose=1)
y_pred           = np.argmax(y_pred_prob, axis=1)
y_true           = test_generator.classes
class_names_list = list(test_generator.class_indices.keys())
class_labels     = [c.replace('_', ' ').title() for c in class_names_list]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=0.5, annot_kws={'size': 12})
plt.title('Confusion Matrix — ResNet-50\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Aktual', fontsize=12); plt.xlabel('Prediksi', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_jalan_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Tersimpan di Google Drive: {save_path}')

print('=' * 60)
print('📋 Classification Report — ResNet-50')
print('=' * 60)
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

report_path = os.path.join(OUTPUT_DIR, 'resnet50_jalan_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('Classification Report\n')
    f.write('Model  : ResNet-50 (Transfer Learning)\n')
    f.write('Task   : Klasifikasi Kerusakan Jalan\n')
    f.write('Kelas  : Baik | Rusak Ringan | Rusak Sedang | Rusak Berat\n')
    f.write('Split  : Train 80% / Val 20% (otomatis) / Test terpisah\n')
    f.write('='*60 + '\n')
    f.write(report)
    f.write(f'\nTest Accuracy : {test_acc*100:.2f}%')
    f.write(f'\nTest Loss     : {test_loss:.4f}')
print(f'\n✅ Tersimpan di Google Drive: {report_path}')

report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
precision = [report_dict[c]['precision'] for c in class_labels]
recall    = [report_dict[c]['recall']    for c in class_labels]
f1        = [report_dict[c]['f1-score']  for c in class_labels]

x, width = np.arange(len(class_labels)), 0.25
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', edgecolor='black')
b2 = ax.bar(x,         recall,    width, label='Recall',    color='#4CAF50', edgecolor='black')
b3 = ax.bar(x + width, f1,        width, label='F1-Score',  color='#FF5722', edgecolor='black')
for bars in [b1, b2, b3]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Kelas Kerusakan Jalan', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall & F1-Score per Kelas\nResNet-50 — Klasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(class_labels)
ax.set_ylim(0, 1.2); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_jalan_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Tersimpan di Google Drive: {save_path}')

model_path = os.path.join(OUTPUT_DIR, 'resnet50_jalan_final.h5')
resnet_model.save(model_path)

print(f'✅ Model tersimpan di Google Drive: {model_path}')
print('\n' + '='*60)
print('🏆 RINGKASAN HASIL — ResNet-50 Klasifikasi Kerusakan Jalan')
print('='*60)
print(f'   Sumber Data            : Google Drive → MyDrive/dataset/')
print(f'   Pembagian Data         : Train 80% / Val 20% / Test terpisah')
print(f'   Best Val Accuracy P1   : {max(history_phase1.history["val_accuracy"])*100:.2f}%')
print(f'   Best Val Accuracy P2   : {max(history_phase2.history["val_accuracy"])*100:.2f}%')
print(f'   Test Accuracy (Final)  : {test_acc*100:.2f}%')
print(f'   Test Loss     (Final)  : {test_loss:.4f}')
print('='*60)
print(f'\n📁 Semua output tersimpan di: {OUTPUT_DIR}')