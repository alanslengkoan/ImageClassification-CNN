# ini frozen 

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

print('✅ TensorFlow version :', tf.__version__)
print('✅ GPU tersedia       :', tf.config.list_physical_devices('GPU'))

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# KONFIGURASI
# ============================================================
TRAIN_DIR  = '/content/drive/MyDrive/dataset/train'
TEST_DIR   = '/content/drive/MyDrive/dataset/test'
OUTPUT_DIR = '/content/drive/MyDrive/dataset'

IMG_SIZE         = (160, 160)   # ✅ diupdate 224 → 160
BATCH_SIZE       = 16           # ✅ tetap 16
EPOCHS           = 20           # ✅ diupdate 50 → 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE    = 1e-4         # ✅ tetap 1e-4
NUM_CLASSES      = 4

print('\n⚙️  Konfigurasi Phase 1:')
print(f'   Backbone      : ResNet50 pretrained ImageNet (BASE FROZEN)')
print(f'   IMG_SIZE      : {IMG_SIZE}')
print(f'   BATCH_SIZE    : {BATCH_SIZE}')
print(f'   EPOCHS        : {EPOCHS} (maks + EarlyStopping)')
print(f'   LEARNING_RATE : {LEARNING_RATE}')
print(f'   Loss          : Categorical Crossentropy')
print(f'   NUM_CLASSES   : {NUM_CLASSES}')

# ============================================================
# CEK DISTRIBUSI DATASET
# ============================================================
print('\n📊 Distribusi Dataset:')
print('-' * 45)
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
                print(f'   {cls:10s}: {count} gambar')
        print(f'   {"TOTAL":10s}: {split_total} gambar')
        total_all += split_total
    else:
        print(f'❌ {split} → folder tidak ditemukan!')
print(f'\n✅ TOTAL KESELURUHAN : {total_all} gambar')

if os.path.exists(TRAIN_DIR):
    total_train = sum([len(f) for r, d, f in os.walk(TRAIN_DIR)])
    print(f'\n📊 Estimasi Pembagian:')
    print(f'   Training (80%) : ~{int(total_train * 0.8)} gambar')
    print(f'   Validasi (20%) : ~{int(total_train * 0.2)} gambar')

# ============================================================
# AUGMENTASI REALISTIS UNTUK FOTO JALAN
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

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

print('\n📊 Pembagian Data:')
print(f'   Train (80%) : {train_generator.samples} gambar')
print(f'   Val   (20%) : {val_generator.samples} gambar')
print(f'   Test        : {test_generator.samples} gambar')
print(f'   Kelas       : {train_generator.class_indices}')

# Update label sesuai urutan folder
CLASS_NAMES  = list(train_generator.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]
print(f'   Label       : {CLASS_LABELS}')

# ============================================================
# BUILD MODEL RESNET-50 (BASE FROZEN - PHASE 1)
# ============================================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # ✅ Semua layer frozen (Phase 1)

inputs  = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_jalan')
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D(name='gap')(x)
x       = layers.Dense(512, activation='relu', name='fc_512')(x)
x       = layers.BatchNormalization(name='bn_1')(x)
x       = layers.Dropout(0.5, name='dropout_1')(x)
x       = layers.Dense(256, activation='relu', name='fc_256')(x)
x       = layers.BatchNormalization(name='bn_2')(x)
x       = layers.Dropout(0.3, name='dropout_2')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

model = keras.Model(inputs, outputs=outputs, name='ResNet50_Phase1_Jalan')
model.summary()

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
frozen    = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f'\n📊 Total Parameter   : {model.count_params():,}')
print(f'   Trainable (aktif) : {trainable:,}  ← hanya classifier head')
print(f'   Frozen (dikunci)  : {frozen:,}  ← seluruh ResNet-50 base')

# ============================================================
# COMPILE
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f'\n✅ Model dikompilasi')
print(f'   Optimizer : Adam (lr={LEARNING_RATE})')
print( '   Loss      : Categorical Crossentropy')
print( '   Metrics   : Accuracy')

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,              # ✅ diupdate 10 → 5 (sesuai max 20 epoch)
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_phase1_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAINING PHASE 1
# ============================================================
print('\n' + '=' * 60)
print('🚀 PHASE 1: Training Classifier Head (Base Frozen)')
print(f'   Train       : {train_generator.samples} gambar')
print(f'   Val         : {val_generator.samples} gambar')
print(f'   Epochs      : maks {EPOCHS} + EarlyStopping (patience=5)')
print(f'   LR          : {LEARNING_RATE}')
print( '   Target      : Val Accuracy ≥ 80%')
print('=' * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

best_val_acc = max(history.history['val_accuracy'])
total_epochs = len(history.history['accuracy'])

print(f'\n{"=" * 60}')
print(f'📊 HASIL PHASE 1:')
print(f'   Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'   Epoch berjalan    : {total_epochs}')
if best_val_acc >= 0.80:
    print(f'   Status            : ✅ TARGET TERCAPAI! Siap ke Phase 2')
elif best_val_acc >= 0.70:
    print(f'   Status            : 🟡 Mendekati target, perlu penyesuaian')
else:
    print(f'   Status            : ❌ Belum mencapai target, perlu perbaikan')
print(f'{"=" * 60}')

# ============================================================
# VISUALISASI TRAINING HISTORY
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ResNet-50 Phase 1 — Klasifikasi Kerusakan Jalan\n(Base Frozen, Input 160×160)',
             fontsize=13, fontweight='bold')

ax1.plot(history.history['accuracy'],     label='Train', color='#2196F3', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val',   color='#FF5722', linewidth=2, linestyle='--')
ax1.axhline(y=0.80, color='green', linewidth=1.5, linestyle=':', label='Target 80%')
ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)

ax2.plot(history.history['loss'],     label='Train', color='#2196F3', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val',   color='#FF5722', linewidth=2, linestyle='--')
ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_history.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik training tersimpan: {save_path}')

# ============================================================
# EVALUASI TEST SET
# ============================================================
print('\n' + '=' * 60)
print('📈 Evaluasi Phase 1 pada Test Set')
print('=' * 60)
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f'\n🏆 Test Accuracy : {test_acc*100:.2f}%')
print(f'   Test Loss     : {test_loss:.4f}')

# Prediksi
test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = test_generator.classes

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            linewidths=0.5, annot_kws={'size': 12})
plt.title('Confusion Matrix — ResNet-50 Phase 1\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Aktual', fontsize=12); plt.xlabel('Prediksi', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Confusion matrix tersimpan: {save_path}')

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print('\n' + '=' * 60)
print('📋 Classification Report — Phase 1')
print('=' * 60)
report = classification_report(y_true, y_pred, target_names=CLASS_LABELS)
print(report)

report_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('Classification Report — ResNet-50 Phase 1\n')
    f.write('Task   : Klasifikasi Kerusakan Jalan\n')
    f.write('Kelas  : Baik | Sedang | Ringan | Berat\n')
    f.write('='*60 + '\n')
    f.write(report)
    f.write(f'\nTest Accuracy : {test_acc*100:.2f}%')
    f.write(f'\nTest Loss     : {test_loss:.4f}')
print(f'✅ Report tersimpan: {report_path}')

# ============================================================
# ROC CURVE
# ============================================================
y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(9, 6))
colors = cycle(['#2196F3', '#F44336', '#4CAF50', '#FF9800'])
for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{cls} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.title('ROC Curve — ResNet-50 Phase 1\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_roc_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ ROC Curve tersimpan: {save_path}')

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
plt.figure(figsize=(9, 6))
colors = cycle(['#2196F3', '#F44336', '#4CAF50', '#FF9800'])
for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_prob[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color=color, linewidth=2,
             label=f'{cls} (AUC = {pr_auc:.2f})')
plt.title('Precision-Recall Curve — ResNet-50 Phase 1\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_pr_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Precision-Recall Curve tersimpan: {save_path}')

# ============================================================
# GRAFIK PRECISION, RECALL & F1 PER KELAS
# ============================================================
report_dict    = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
precision_list = [report_dict[c]['precision'] for c in CLASS_LABELS]
recall_list    = [report_dict[c]['recall']    for c in CLASS_LABELS]
f1_list        = [report_dict[c]['f1-score']  for c in CLASS_LABELS]

x, width = np.arange(len(CLASS_LABELS)), 0.25
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width, precision_list, width, label='Precision', color='#2196F3', edgecolor='black')
b2 = ax.bar(x,         recall_list,    width, label='Recall',    color='#4CAF50', edgecolor='black')
b3 = ax.bar(x + width, f1_list,        width, label='F1-Score',  color='#FF5722', edgecolor='black')
for bars in [b1, b2, b3]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Kelas Kerusakan Jalan', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall & F1-Score per Kelas\nResNet-50 Phase 1 — Klasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS)
ax.set_ylim(0, 1.2); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_phase1_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n' + '=' * 60)
print('📋 RINGKASAN PHASE 1')
print('=' * 60)
print(f'   Backbone          : ResNet50 pretrained ImageNet (Frozen)')
print(f'   Input size        : {IMG_SIZE}')
print(f'   Batch size        : {BATCH_SIZE}')
print(f'   Optimizer         : Adam (lr={LEARNING_RATE})')
print(f'   Loss              : Categorical Crossentropy')
print(f'   Kelas             : {CLASS_LABELS}')
print(f'   Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'   Test Accuracy     : {test_acc*100:.2f}%')
print(f'   Test Loss         : {test_loss:.4f}')
print(f'   Epoch berjalan    : {total_epochs} / {EPOCHS}')
print(f'   Model tersimpan   : {OUTPUT_DIR}/resnet50_phase1_best.h5')
print('=' * 60)
print(f'\n📁 Semua output tersimpan di: {OUTPUT_DIR}')
print('   - resnet50_phase1_best.h5')
print('   - resnet50_phase1_history.png')
print('   - resnet50_phase1_confusion_matrix.png')
print('   - resnet50_phase1_roc_curve.png')
print('   - resnet50_phase1_pr_curve.png')
print('   - resnet50_phase1_per_class_metrics.png')
print('   - resnet50_phase1_classification_report.txt')

if best_val_acc >= 0.80:
    print('\n✅ Phase 1 BERHASIL mencapai target ≥ 80%!')
    print('   → Siap lanjut ke Phase 2 (Fine-tuning)')
else:
    print(f'\n⚠️  Phase 1 belum mencapai 80% (saat ini {best_val_acc*100:.2f}%)')
    print('   → Share hasil ini untuk analisis lebih lanjut')