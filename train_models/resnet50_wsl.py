import os

# ============================================================
# [WSL] OPTIMASI CPU — Intel Core Ultra 5 225H (14 CPUs)
# Harus di-set SEBELUM import TensorFlow
# ============================================================
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '1'   # Intel oneDNN acceleration
os.environ['OMP_NUM_THREADS']         = '10'  # OpenMP threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '10'  # Operasi dalam satu op
os.environ['TF_NUM_INTEROP_THREADS'] = '4'   # Operasi antar op
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'   # Kurangi log TF

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')   # [WSL] simpan grafik tanpa display window
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
from sklearn.utils import class_weight
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

# Set CPU threading setelah import TF
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(4)

print('✅ TensorFlow version :', tf.__version__)
print('✅ GPU tersedia       :', tf.config.list_physical_devices('GPU'))
print('✅ [WSL] CPU threads  : intra=10, inter=4 (Intel Core Ultra 5 225H)')
print('✅ [WSL] oneDNN Intel : aktif')

# ============================================================
# KONFIGURASI PATH — WSL (Ubuntu on Windows 11)
# ============================================================
BASE_DIR   = '/mnt/d/ImageClassification-CNN'  # [WSL] Drive D
TRAIN_DIR  = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR    = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')

# ============================================================
# KONFIGURASI — DISESUAIKAN UNTUK CPU (WSL)
# ============================================================
IMG_SIZE         = (224, 224)   # Proven untuk ResNet50 pretrained ImageNet
BATCH_SIZE       = 16           # [WSL] ↓ 32→16: lebih efisien di CPU
EPOCHS           = 70           # [WSL] dinaikkan: batch=16 butuh lebih banyak epoch
LEARNING_RATE    = 2e-4         # Terbukti pada val 78.84%
NUM_CLASSES      = 3            # 3 kelas: baik, sedang, berat
FINE_TUNE_LAYERS = 20           # Sesuai reference: lebih banyak layer backbone adapt

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('\n⚙️  Konfigurasi ResNet50 (3 KELAS):')
print(f'   📌 Strategi      : Gabung ringan+sedang → sedang')
print(f'   Backbone         : ResNet50 pretrained ImageNet')
print(f'   Kelas            : 3 (baik, sedang, berat)')
print(f'   IMG_SIZE         : {IMG_SIZE}')
print(f'   BATCH_SIZE       : {BATCH_SIZE}')
print(f'   EPOCHS           : {EPOCHS} (maks + EarlyStopping)')
print(f'   LEARNING_RATE    : {LEARNING_RATE}')
print(f'   Fine-tune layers : {FINE_TUNE_LAYERS} layer terakhir ResNet50')
print(f'   Class weight     : ✅ Aktif')
print(f'   Head             : GAP → BN → Dense(256) → Dropout(0.5) → Softmax')
print(f'   Split data       : 70% train / 20% val / 10% test')
print(f'   🎯 Target        : Val Accuracy ≥ 80%')

# ============================================================
# CEK DISTRIBUSI DATASET
# ============================================================
print('\n📊 Distribusi Dataset:')
print('-' * 50)
total_all = 0
for split, path in [('TRAIN', TRAIN_DIR), ('VAL', VAL_DIR), ('TEST', TEST_DIR)]:
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
        print(f'❌ {split} → folder tidak ditemukan: {path}')
print(f'\n✅ TOTAL KESELURUHAN : {total_all} gambar')

# ============================================================
# AUGMENTASI
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print('\n📊 Pembagian Data:')
print(f'   Train (70%) : {train_generator.samples} gambar')
print(f'   Val   (20%) : {val_generator.samples} gambar')
print(f'   Test  (10%) : {test_generator.samples} gambar')
print(f'   Kelas       : {train_generator.class_indices}')

CLASS_NAMES  = list(train_generator.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]
print(f'   Label       : {CLASS_LABELS}')

# ============================================================
# CLASS WEIGHT
# ============================================================
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights_array))

print('\n⚖️  Class Weights:')
for cls, w in zip(CLASS_LABELS, class_weights_array):
    print(f'   {cls:10s}: {w:.4f}')

# ============================================================
# BUILD MODEL RESNET50
# ============================================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

trainable_layers = sum([1 for l in base_model.layers if l.trainable])
frozen_layers    = sum([1 for l in base_model.layers if not l.trainable])
print(f'\n📊 ResNet50 Layers:')
print(f'   Total layer      : {len(base_model.layers)}')
print(f'   Frozen layer     : {frozen_layers}')
print(f'   Trainable layer  : {trainable_layers} ({FINE_TUNE_LAYERS} terakhir) ✅')

# Classifier Head (simplified — sesuai reference)
inputs  = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_jalan')
x       = base_model(inputs, training=True)   # BN adapt ke domain road damage
x       = layers.GlobalAveragePooling2D(name='gap')(x)
x       = layers.BatchNormalization(name='bn_1')(x)
x       = layers.Dense(256, activation='relu', name='fc_256')(x)
x       = layers.Dropout(0.5, name='dropout_1')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

model = keras.Model(inputs, outputs=outputs, name='ResNet50_FineTune_Jalan')
model.summary()

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
frozen_params    = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f'\n📊 Total Parameter   : {model.count_params():,}')
print(f'   Trainable (aktif) : {trainable_params:,}')
print(f'   Frozen (dikunci)  : {frozen_params:,}')

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

# ============================================================
# CALLBACKS
# ============================================================
# ============================================================
# CALLBACKS - OPTIMIZED
# ============================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,          # [WSL] 15→20: batch=16 lebih noisy, butuh waktu lebih
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_3class_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,          # [WSL] 8→12: hindari LR turun terlalu cepat di batch=16
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAINING
# ============================================================
print('\n' + '=' * 60)
print('🚀 TRAINING ResNet50 — 3 KELAS (WSL)')
print(f'   📌 Strategi  : Gabung ringan+sedang → sedang')
print(f'   Kelas        : {NUM_CLASSES} (baik, sedang, berat)')
print(f'   Train        : {train_generator.samples} gambar (70%)')
print(f'   Val          : {val_generator.samples} gambar (20%)')
print(f'   Test         : {test_generator.samples} gambar (10%)')
print(f'   Epochs       : maks {EPOCHS} + EarlyStopping (patience=10)')
print(f'   LR           : {LEARNING_RATE}')
print(f'   Batch size   : {BATCH_SIZE}')
print(f'   Fine-tune    : {FINE_TUNE_LAYERS} layer terakhir')
print(f'   Head: GAP → BN → Dense(256) → Dropout(0.5) → Dense({NUM_CLASSES})')
print( '   🎯 Target    : Val Accuracy ≥ 80%')
print('=' * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

best_val_acc = max(history.history['val_accuracy'])
total_epochs = len(history.history['accuracy'])

print(f'\n{"=" * 60}')
print(f'📊 HASIL TRAINING:')
print(f'   Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'   Epoch berjalan    : {total_epochs}')
if best_val_acc >= 0.80:
    print(f'   Status            : ✅ TARGET TERCAPAI!')
elif best_val_acc >= 0.70:
    print(f'   Status            : 🟡 Mendekati target')
else:
    print(f'   Status            : ❌ Belum mencapai target')
print(f'{"=" * 60}')

# ============================================================
# VISUALISASI TRAINING HISTORY
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ResNet50 Fine-Tuning — Klasifikasi Kerusakan Jalan (3 KELAS)\n'
             f'(Fine-tune {FINE_TUNE_LAYERS} Layer, Input {IMG_SIZE[0]}×{IMG_SIZE[1]}, Batch {BATCH_SIZE}, LR {LEARNING_RATE})',
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
save_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_history.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik training tersimpan: {save_path}')

# ============================================================
# EVALUASI TEST SET
# ============================================================
print('\n' + '=' * 60)
print('📈 Evaluasi pada Test Set')
print('=' * 60)
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f'\n🏆 Test Accuracy : {test_acc*100:.2f}%')
print(f'   Test Loss     : {test_loss:.4f}')

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
plt.title('Confusion Matrix — ResNet50 Fine-Tuning (3 KELAS)\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Aktual', fontsize=12); plt.xlabel('Prediksi', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Confusion matrix tersimpan: {save_path}')

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print('\n' + '=' * 60)
print('📋 Classification Report')
print('=' * 60)
report = classification_report(y_true, y_pred, target_names=CLASS_LABELS)
print(report)

report_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('Classification Report — ResNet50 Fine-Tuning (3 KELAS)\n')
    f.write('Task      : Klasifikasi Kerusakan Jalan\n')
    f.write('Strategi  : Gabung ringan+sedang → sedang\n')
    f.write('Kelas     : Baik | Sedang | Berat\n')
    f.write('Platform  : WSL Ubuntu / Windows 11 — CPU Intel Core Ultra 5 225H\n')
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
plt.title('ROC Curve — ResNet50 Fine-Tuning (3 KELAS)\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_roc_curve.png')
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
plt.title('Precision-Recall Curve — ResNet50 Fine-Tuning (3 KELAS)\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_pr_curve.png')
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
ax.set_title('Precision, Recall & F1-Score per Kelas (3 KELAS)\nResNet50 Fine-Tuning — Klasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS)
ax.set_ylim(0, 1.2); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_3class_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n' + '=' * 60)
print('📋 RINGKASAN HASIL — ResNet50 Fine-Tuning 3 KELAS (WSL)')
print('=' * 60)
print(f'   Backbone          : ResNet50 pretrained ImageNet')
print(f'   📌 Strategi       : Gabung ringan+sedang → sedang')
print(f'   Kelas             : {NUM_CLASSES} kelas → {CLASS_LABELS}')
print(f'   Fine-tune layers  : {FINE_TUNE_LAYERS} layer terakhir')
print(f'   Class Weight      : ✅ Aktif')
print(f'   L2 Regularization : ❌ Dihapus (cukup BN + Dropout)')
print(f'   Input size        : {IMG_SIZE}')
print(f'   Batch size        : {BATCH_SIZE}')
print(f'   Optimizer         : Adam (lr={LEARNING_RATE})')
print(f'   Loss              : Categorical Crossentropy')
print(f'   Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'   Test Accuracy     : {test_acc*100:.2f}%')
print(f'   Test Loss         : {test_loss:.4f}')
print(f'   Epoch berjalan    : {total_epochs} / {EPOCHS}')
print(f'   Model tersimpan   : {OUTPUT_DIR}/resnet50_3class_best.h5')
print('=' * 60)

if best_val_acc >= 0.80:
    print('\n✅ TARGET TERCAPAI ≥ 80%!')
elif best_val_acc >= 0.70:
    print(f'\n🟡 Mendekati target (saat ini {best_val_acc*100:.2f}%)')
else:
    print(f'\n⚠️  Belum mencapai 80% (saat ini {best_val_acc*100:.2f}%)')
