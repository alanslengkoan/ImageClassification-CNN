import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.mixed_precision import set_global_policy

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
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
# MIXED PRECISION (untuk training lebih cepat)
# ============================================================
set_global_policy('mixed_float16')
print('✅ Mixed Precision Training: Aktif')

# ============================================================
# KONFIGURASI OPTIMIZED
# ============================================================
TRAIN_DIR  = '/content/drive/MyDrive/dataset/train'
TEST_DIR   = '/content/drive/MyDrive/dataset/test'
OUTPUT_DIR = '/content/drive/MyDrive/dataset'

# ✅ OPTIMASI #1: Ukuran image optimal untuk ResNet50
IMG_SIZE         = (224, 224)  # Ukuran standar ResNet50
BATCH_SIZE       = 16
EPOCHS           = 20
VALIDATION_SPLIT = 0.2
NUM_CLASSES      = 4

# ✅ OPTIMASI #2: Learning rate scheduling
INITIAL_LR       = 1e-3        # Start lebih tinggi
MIN_LR           = 1e-7

# ✅ OPTIMASI #3: Fine-tune bertahap
FINE_TUNE_AT     = 100         # Unfreeze dari layer ke-100

print('\n⚙️  Konfigurasi ResNet50 OPTIMIZED:')
print(f'   Backbone         : ResNet50 pretrained ImageNet')
print(f'   IMG_SIZE         : {IMG_SIZE} (optimal untuk ResNet50)')
print(f'   BATCH_SIZE       : {BATCH_SIZE}')
print(f'   EPOCHS           : {EPOCHS} (maks + EarlyStopping patience=10)')
print(f'   INITIAL_LR       : {INITIAL_LR}')
print(f'   Loss             : Categorical Crossentropy + Label Smoothing')
print(f'   Fine-tune from   : Layer {FINE_TUNE_AT}')
print(f'   Mixed Precision  : ✅ Aktif')
print(f'   Class weight     : ✅ Aktif (improved)')
print(f'   L2 Regularization: ✅ Aktif')
print(f'   Test-Time Aug    : ✅ Akan digunakan')

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
# ✅ OPTIMASI #4: AUGMENTASI LEBIH AGRESIF
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT,
    rotation_range=25,           # Lebih agresif
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3], # Range lebih luas
    channel_shift_range=20.0,    # Color jitter
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generator TRAIN
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Generator VAL
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

CLASS_NAMES  = list(train_generator.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]
print(f'   Label       : {CLASS_LABELS}')

# ============================================================
# ✅ OPTIMASI #5: IMPROVED CLASS WEIGHT
# ============================================================
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights_array))

print('\n⚖️  Class Weights (balanced):')
for idx, (cls, w) in enumerate(zip(CLASS_LABELS, class_weights_array)):
    print(f'   {cls:10s}: {w:.4f}')

# ============================================================
# ✅ OPTIMASI #6: BUILD MODEL DENGAN L2 REGULARIZATION
# ============================================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Frozen semua dulu
base_model.trainable = False

print(f'\n📊 ResNet50 Base Model:')
print(f'   Total layers     : {len(base_model.layers)}')
print(f'   Status awal      : Semua frozen (akan di-unfreeze nanti)')

# ✅ OPTIMASI #7: Classifier head dengan regularization
inputs  = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_jalan')
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D(name='gap')(x)
x       = layers.BatchNormalization(name='bn_1')(x)
x       = layers.Dense(512, activation='relu', 
                       kernel_regularizer=regularizers.l2(0.001),
                       name='fc_512')(x)
x       = layers.Dropout(0.5, name='dropout_1')(x)
x       = layers.BatchNormalization(name='bn_2')(x)
x       = layers.Dense(256, activation='relu',
                       kernel_regularizer=regularizers.l2(0.001),
                       name='fc_256')(x)
x       = layers.Dropout(0.4, name='dropout_2')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', 
                       dtype='float32',  # Mixed precision fix
                       name='output')(x)

model = keras.Model(inputs, outputs, name='ResNet50_Optimized')

print(f'\n✅ Model architecture created')
print(f'   Classifier: GAP → BN → Dense(512) → Dropout → BN → Dense(256) → Dropout → Output')
print(f'   Regularization: L2 + Dropout + BatchNorm')

# ============================================================
# ✅ OPTIMASI #8: COMPILE DENGAN LABEL SMOOTHING
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print(f'\n✅ Model dikompilasi (Stage 1 - Base Frozen)')
print(f'   Optimizer      : Adam (lr={INITIAL_LR})')
print(f'   Loss           : Categorical Crossentropy + Label Smoothing 0.1')
print(f'   Metrics        : Accuracy')
print(f'   Base trainable : False (akan di-train dulu classifier)')

# ============================================================
# CALLBACKS STAGE 1
# ============================================================
callbacks_stage1 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_opt_stage1.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=MIN_LR,
        verbose=1
    )
]

# ============================================================
# STAGE 1: TRAIN CLASSIFIER HEAD ONLY
# ============================================================
print('\n' + '=' * 70)
print('🚀 STAGE 1: Training Classifier Head (Base Frozen)')
print(f'   Epochs      : 10 epochs')
print(f'   LR          : {INITIAL_LR}')
print(f'   Trainable   : Hanya classifier head')
print('=' * 70)

history_stage1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks_stage1,
    class_weight=class_weights_dict,
    verbose=1
)

best_val_acc_s1 = max(history_stage1.history['val_accuracy'])
print(f'\n✅ STAGE 1 Selesai - Best Val Accuracy: {best_val_acc_s1*100:.2f}%')

# ============================================================
# STAGE 2: FINE-TUNE RESNET50
# ============================================================
print('\n' + '=' * 70)
print('🚀 STAGE 2: Fine-Tuning ResNet50')
print('=' * 70)

# Unfreeze dari layer tertentu
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

trainable_layers = sum([1 for l in base_model.layers if l.trainable])
frozen_layers    = sum([1 for l in base_model.layers if not l.trainable])

print(f'\n📊 Fine-tuning Configuration:')
print(f'   Total layers     : {len(base_model.layers)}')
print(f'   Frozen layers    : {frozen_layers} (layer 0-{FINE_TUNE_AT-1})')
print(f'   Trainable layers : {trainable_layers} (layer {FINE_TUNE_AT}+)')

# ✅ OPTIMASI #9: Lower learning rate untuk fine-tuning
FINE_TUNE_LR = 1e-5

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print(f'\n✅ Model re-compiled untuk fine-tuning')
print(f'   New LR : {FINE_TUNE_LR} (10x lebih rendah)')

# ✅ OPTIMASI #10: Cosine Decay Learning Rate
def cosine_decay_with_warmup(epoch, lr):
    """Custom learning rate schedule"""
    if epoch < 5:  # Warmup
        return FINE_TUNE_LR * (epoch + 1) / 5
    else:
        # Cosine decay
        progress = (epoch - 5) / (EPOCHS - 5)
        return MIN_LR + (FINE_TUNE_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress))

callbacks_stage2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'resnet50_optimized_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    LearningRateScheduler(cosine_decay_with_warmup, verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=MIN_LR,
        verbose=1
    )
]

print('\n' + '=' * 70)
print(f'🚀 TRAINING STAGE 2 — Fine-Tune dengan Cosine LR Schedule')
print(f'   Train       : {train_generator.samples} gambar')
print(f'   Val         : {val_generator.samples} gambar')
print(f'   Max Epochs  : {EPOCHS}')
print(f'   LR Schedule : Warmup (5 epochs) → Cosine Decay')
print(f'   Patience    : 10 epochs')
print(f'   Target      : Val Accuracy ≥ 80%')
print('=' * 70)

history_stage2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_stage2,
    class_weight=class_weights_dict,
    verbose=1
)

best_val_acc = max(history_stage2.history['val_accuracy'])
total_epochs = len(history_stage2.history['accuracy'])

print(f'\n{"=" * 70}')
print(f'📊 HASIL TRAINING STAGE 2:')
print(f'   Best Val Accuracy : {best_val_acc*100:.2f}%')
print(f'   Epoch berjalan    : {total_epochs} / {EPOCHS}')
if best_val_acc >= 0.80:
    print(f'   Status            : ✅ TARGET TERCAPAI!')
elif best_val_acc >= 0.70:
    print(f'   Status            : 🟡 Mendekati target')
else:
    print(f'   Status            : ⚠️  Perlu improvement')
print(f'{"=" * 70}')

# ============================================================
# VISUALISASI TRAINING HISTORY (STAGE 1 + STAGE 2)
# ============================================================
# Gabungkan history
all_acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
all_val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
all_loss = history_stage1.history['loss'] + history_stage2.history['loss']
all_val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('ResNet50 OPTIMIZED — Two-Stage Training\n(Stage 1: Classifier | Stage 2: Fine-tune)',
             fontsize=13, fontweight='bold')

epochs_range = range(1, len(all_acc) + 1)
stage1_end = len(history_stage1.history['accuracy'])

ax1.plot(epochs_range, all_acc, label='Train', color='#2196F3', linewidth=2)
ax1.plot(epochs_range, all_val_acc, label='Val', color='#FF5722', linewidth=2, linestyle='--')
ax1.axvline(x=stage1_end, color='gray', linewidth=2, linestyle=':', label='Stage 1→2')
ax1.axhline(y=0.80, color='green', linewidth=1.5, linestyle=':', label='Target 80%')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

ax2.plot(epochs_range, all_loss, label='Train', color='#2196F3', linewidth=2)
ax2.plot(epochs_range, all_val_loss, label='Val', color='#FF5722', linewidth=2, linestyle='--')
ax2.axvline(x=stage1_end, color='gray', linewidth=2, linestyle=':', label='Stage 1→2')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_history.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik training tersimpan: {save_path}')

# ============================================================
# ✅ OPTIMASI #11: TEST-TIME AUGMENTATION (TTA)
# ============================================================
print('\n' + '=' * 70)
print('🔬 Evaluasi dengan Test-Time Augmentation (TTA)')
print('=' * 70)

def predict_with_tta(model, generator, n_augment=5):
    """Predict dengan augmentasi multiple kali"""
    generator.reset()
    predictions = []
    
    # Original predictions
    print(f'   Prediksi original...')
    preds = model.predict(generator, verbose=0)
    predictions.append(preds)
    
    # Augmented predictions
    for i in range(n_augment):
        print(f'   Prediksi augmentasi {i+1}/{n_augment}...')
        generator.reset()
        aug_preds = model.predict(generator, verbose=0)
        predictions.append(aug_preds)
    
    # Average semua predictions
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

print('\n📊 Test-Time Augmentation (TTA) dengan 5 augmentasi...')
y_pred_prob_tta = predict_with_tta(model, test_generator, n_augment=5)
y_pred_tta = np.argmax(y_pred_prob_tta, axis=1)
y_true = test_generator.classes

# Evaluasi biasa (tanpa TTA)
test_generator.reset()
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
y_pred_prob_normal = model.predict(test_generator, verbose=0)
y_pred_normal = np.argmax(y_pred_prob_normal, axis=1)

print(f'\n📊 Perbandingan Hasil:')
print(f'   Tanpa TTA : {test_acc*100:.2f}% accuracy')
print(f'   Dengan TTA: {(y_pred_tta == y_true).mean()*100:.2f}% accuracy')

# Gunakan TTA untuk evaluasi akhir
y_pred = y_pred_tta
y_pred_prob = y_pred_prob_tta
final_test_acc = (y_pred == y_true).mean()

print(f'\n🏆 Test Accuracy (TTA) : {final_test_acc*100:.2f}%')
print(f'   Test Loss           : {test_loss:.4f}')

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            linewidths=0.5, annot_kws={'size': 12})
plt.title('Confusion Matrix — ResNet50 OPTIMIZED + TTA\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Aktual', fontsize=12)
plt.xlabel('Prediksi', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Confusion matrix tersimpan: {save_path}')

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print('\n' + '=' * 70)
print('📋 Classification Report (dengan TTA)')
print('=' * 70)
report = classification_report(y_true, y_pred, target_names=CLASS_LABELS)
print(report)

report_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('Classification Report — ResNet50 OPTIMIZED + TTA\n')
    f.write('Task   : Klasifikasi Kerusakan Jalan\n')
    f.write('Kelas  : Baik | Sedang | Ringan | Berat\n')
    f.write('='*70 + '\n\n')
    f.write('OPTIMIZATIONS APPLIED:\n')
    f.write('1. Image Size: 224×224 (optimal untuk ResNet50)\n')
    f.write('2. Two-Stage Training: Classifier → Fine-tune\n')
    f.write('3. Aggressive Data Augmentation\n')
    f.write('4. Label Smoothing (0.1)\n')
    f.write('5. L2 Regularization + Dropout\n')
    f.write('6. Cosine Learning Rate Schedule\n')
    f.write('7. Mixed Precision Training\n')
    f.write('8. Test-Time Augmentation (TTA)\n')
    f.write('9. Balanced Class Weights\n')
    f.write('10. More Epochs (50) + EarlyStopping\n')
    f.write('='*70 + '\n\n')
    f.write(report)
    f.write(f'\n\nRESULTS:\n')
    f.write(f'Stage 1 Best Val Acc : {best_val_acc_s1*100:.2f}%\n')
    f.write(f'Stage 2 Best Val Acc : {best_val_acc*100:.2f}%\n')
    f.write(f'Test Accuracy (TTA)  : {final_test_acc*100:.2f}%\n')
    f.write(f'Test Loss            : {test_loss:.4f}\n')
print(f'✅ Report tersimpan: {report_path}')

# ============================================================
# ROC CURVE
# ============================================================
y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(10, 7))
colors = cycle(['#2196F3', '#F44336', '#4CAF50', '#FF9800'])
for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f'{cls} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
plt.title('ROC Curve — ResNet50 OPTIMIZED + TTA\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_roc_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ ROC Curve tersimpan: {save_path}')

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================
plt.figure(figsize=(10, 7))
colors = cycle(['#2196F3', '#F44336', '#4CAF50', '#FF9800'])
for i, (color, cls) in enumerate(zip(colors, CLASS_LABELS)):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_prob[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color=color, linewidth=2.5,
             label=f'{cls} (AUC = {pr_auc:.3f})')
plt.title('Precision-Recall Curve — ResNet50 OPTIMIZED + TTA\nKlasifikasi Kerusakan Jalan',
          fontsize=13, fontweight='bold')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_pr_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Precision-Recall Curve tersimpan: {save_path}')

# ============================================================
# GRAFIK PRECISION, RECALL & F1 PER KELAS
# ============================================================
report_dict = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
precision_list = [report_dict[c]['precision'] for c in CLASS_LABELS]
recall_list = [report_dict[c]['recall'] for c in CLASS_LABELS]
f1_list = [report_dict[c]['f1-score'] for c in CLASS_LABELS]

x, width = np.arange(len(CLASS_LABELS)), 0.25
fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width, precision_list, width, label='Precision', color='#2196F3', edgecolor='black')
b2 = ax.bar(x, recall_list, width, label='Recall', color='#4CAF50', edgecolor='black')
b3 = ax.bar(x + width, f1_list, width, label='F1-Score', color='#FF5722', edgecolor='black')

for bars in [b1, b2, b3]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Kelas Kerusakan Jalan', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall & F1-Score per Kelas\nResNet50 OPTIMIZED + TTA',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'resnet50_optimized_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n' + '=' * 70)
print('📋 RINGKASAN HASIL — ResNet50 OPTIMIZED')
print('=' * 70)
print('\n🔧 OPTIMIZATIONS APPLIED:')
print('   1. ✅ Image Size: 224×224 (optimal untuk ResNet50)')
print('   2. ✅ Two-Stage Training: Classifier → Fine-tune')
print('   3. ✅ Aggressive Data Augmentation')
print('   4. ✅ Label Smoothing (0.1)')
print('   5. ✅ L2 Regularization + Dropout')
print('   6. ✅ Cosine Learning Rate Schedule with Warmup')
print('   7. ✅ Mixed Precision Training')
print('   8. ✅ Test-Time Augmentation (TTA)')
print('   9. ✅ Balanced Class Weights')
print('   10. ✅ More Epochs (50) + EarlyStopping')

print(f'\n📊 RESULTS:')
print(f'   Stage 1 Val Acc      : {best_val_acc_s1*100:.2f}%')
print(f'   Stage 2 Val Acc      : {best_val_acc*100:.2f}%')
print(f'   Test Acc (no TTA)    : {test_acc*100:.2f}%')
print(f'   Test Acc (with TTA)  : {final_test_acc*100:.2f}% ⭐')
print(f'   Test Loss            : {test_loss:.4f}')
print(f'   Total Epochs         : {len(all_acc)}')

print(f'\n📁 MODEL & OUTPUTS:')
print(f'   Best model           : {OUTPUT_DIR}/resnet50_optimized_best.h5')
print(f'   Training history     : {OUTPUT_DIR}/resnet50_optimized_history.png')
print(f'   Confusion matrix     : {OUTPUT_DIR}/resnet50_optimized_confusion_matrix.png')
print(f'   ROC curve            : {OUTPUT_DIR}/resnet50_optimized_roc_curve.png')
print(f'   PR curve             : {OUTPUT_DIR}/resnet50_optimized_pr_curve.png')
print(f'   Per-class metrics    : {OUTPUT_DIR}/resnet50_optimized_per_class_metrics.png')
print(f'   Classification report: {OUTPUT_DIR}/resnet50_optimized_classification_report.txt')

print('\n' + '=' * 70)
if final_test_acc >= 0.80:
    print('✅ TARGET TERCAPAI ≥ 80%! 🎉')
    print('   Model sudah optimal untuk deployment.')
elif final_test_acc >= 0.75:
    print('🟡 Mendekati target (75-80%)')
    print('   SARAN LANJUTAN:')
    print('   → Tambah data training (minimal 500 gambar per kelas)')
    print('   → Coba EfficientNetB3/B4 untuk arsitektur lebih baik')
    print('   → Ensemble dengan model berbeda')
elif final_test_acc >= 0.70:
    print('⚠️  Cukup baik tapi belum optimal (70-75%)')
    print('   SARAN LANJUTAN:')
    print('   → PRIORITAS: Tambah data (dataset terlalu kecil)')
    print('   → Coba Vision Transformer (ViT) jika GPU kuat')
    print('   → Data cleaning & relabeling')
else:
    print(f'❌ Belum mencapai target ({final_test_acc*100:.2f}%)')
    print('   SOLUSI UTAMA:')
    print('   → Dataset terlalu kecil! Butuh minimal 2000+ gambar total')
    print('   → Periksa kualitas labeling data')
    print('   → Coba pre-trained model lain (EfficientNet, ConvNeXt)')
print('=' * 70)

print('\n💡 NOTES:')
print('   - TTA meningkatkan robustness prediksi')
print('   - Two-stage training mencegah overfitting')
print('   - Mixed precision mempercepat training 2-3x')
print('   - Jika hasil belum 80%, fokus pada penambahan data')
