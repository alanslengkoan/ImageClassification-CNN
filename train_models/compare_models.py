import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

print('✅ TensorFlow version :', tf.__version__)

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR   = '/home/echolog/Documents/Project/www/skripsi/ImageClassification-CNN/train_models'
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')

MODEL_CNN      = os.path.join(OUTPUT_DIR, 'cnn_baseline_best.h5')
MODEL_RESNET   = os.path.join(OUTPUT_DIR, 'resnet50_3class_best.h5')
COMPARE_DIR    = os.path.join(OUTPUT_DIR, 'comparison')

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

os.makedirs(COMPARE_DIR, exist_ok=True)

# ============================================================
# CEK MODEL TERSEDIA
# ============================================================
for path, name in [(MODEL_CNN, 'CNN Baseline'), (MODEL_RESNET, 'ResNet50')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Model {name} tidak ditemukan: {path}\n'
            f'Jalankan training terlebih dahulu.'
        )
print('✅ Kedua model ditemukan')

# ============================================================
# LOAD MODEL
# ============================================================
print('\n📦 Loading models...')
model_cnn     = load_model(MODEL_CNN)
model_resnet  = load_model(MODEL_RESNET)
print('✅ CNN Baseline loaded')
print('✅ ResNet50 loaded')

# ============================================================
# DATA GENERATOR — masing-masing pakai preprocessing berbeda
# ============================================================
# CNN Baseline: rescale [0,1]
gen_cnn = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# ResNet50: preprocess_input ImageNet
gen_resnet = ImageDataGenerator(preprocessing_function=resnet_preprocess).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

CLASS_NAMES  = list(gen_cnn.class_indices.keys())
CLASS_LABELS = [c.title() for c in CLASS_NAMES]
NUM_CLASSES  = len(CLASS_NAMES)
y_true       = gen_cnn.classes

print(f'\n📊 Test set  : {gen_cnn.samples} gambar')
print(f'   Kelas     : {CLASS_LABELS}')

# ============================================================
# EVALUASI KEDUA MODEL
# ============================================================
print('\n' + '=' * 60)
print('📈 EVALUASI TEST SET')
print('=' * 60)

print('\n🔵 CNN Baseline...')
gen_cnn.reset()
loss_cnn, acc_cnn = model_cnn.evaluate(gen_cnn, verbose=0)
gen_cnn.reset()
prob_cnn  = model_cnn.predict(gen_cnn, verbose=1)
pred_cnn  = np.argmax(prob_cnn, axis=1)

print('\n🟠 ResNet50...')
gen_resnet.reset()
loss_resnet, acc_resnet = model_resnet.evaluate(gen_resnet, verbose=0)
gen_resnet.reset()
prob_resnet  = model_resnet.predict(gen_resnet, verbose=1)
pred_resnet  = np.argmax(prob_resnet, axis=1)

print(f'\n{"=" * 60}')
print(f'📊 HASIL PERBANDINGAN:')
print(f'{"=" * 60}')
print(f'   {"Model":<20} {"Test Accuracy":>15} {"Test Loss":>12}')
print(f'   {"-"*47}')
print(f'   {"CNN Baseline":<20} {acc_cnn*100:>14.2f}% {loss_cnn:>12.4f}')
print(f'   {"ResNet50":<20} {acc_resnet*100:>14.2f}% {loss_resnet:>12.4f}')
print(f'{"=" * 60}')
delta = (acc_resnet - acc_cnn) * 100
print(f'\n   ResNet50 unggul {delta:+.2f}% dibanding CNN Baseline')

# ============================================================
# 1. BAR CHART: Accuracy & Loss Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Perbandingan CNN Baseline vs ResNet50\nKlasifikasi Kerusakan Jalan (3 Kelas)',
             fontsize=14, fontweight='bold')

models  = ['CNN Baseline', 'ResNet50']
colors  = ['#FF9800', '#2196F3']
accs    = [acc_cnn * 100,  acc_resnet * 100]
losses  = [loss_cnn,       loss_resnet]

bars1 = axes[0].bar(models, accs, color=colors, edgecolor='black', width=0.5)
for b, v in zip(bars1, accs):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].axhline(y=80, color='green', linestyle=':', linewidth=1.5, label='Target 80%')
axes[0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 110)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(models, losses, color=colors, edgecolor='black', width=0.5)
for b, v in zip(bars2, losses):
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[1].set_title('Test Loss', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_accuracy_loss.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\n✅ Grafik accuracy/loss tersimpan: {save_path}')

# ============================================================
# 2. CONFUSION MATRIX SIDE-BY-SIDE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrix — CNN Baseline vs ResNet50\nKlasifikasi Kerusakan Jalan (3 Kelas)',
             fontsize=14, fontweight='bold')

for ax, pred, title, cmap in [
    (axes[0], pred_cnn,    f'CNN Baseline (Acc={acc_cnn*100:.2f}%)',    'Oranges'),
    (axes[1], pred_resnet, f'ResNet50     (Acc={acc_resnet*100:.2f}%)', 'Blues'),
]:
    cm = confusion_matrix(y_true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                linewidths=0.5, annot_kws={'size': 13}, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Aktual', fontsize=11)
    ax.set_xlabel('Prediksi', fontsize=11)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Confusion matrix tersimpan: {save_path}')

# ============================================================
# 3. CLASSIFICATION REPORT — keduanya
# ============================================================
print('\n' + '=' * 60)
print('📋 Classification Report — CNN Baseline')
print('=' * 60)
report_cnn = classification_report(y_true, pred_cnn, target_names=CLASS_LABELS)
print(report_cnn)

print('=' * 60)
print('📋 Classification Report — ResNet50')
print('=' * 60)
report_resnet = classification_report(y_true, pred_resnet, target_names=CLASS_LABELS)
print(report_resnet)

report_path = os.path.join(COMPARE_DIR, 'comparison_classification_report.txt')
with open(report_path, 'w') as f:
    f.write('=' * 60 + '\n')
    f.write('PERBANDINGAN MODEL — Klasifikasi Kerusakan Jalan (3 Kelas)\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'CNN Baseline  — Test Accuracy: {acc_cnn*100:.2f}%  | Test Loss: {loss_cnn:.4f}\n')
    f.write(f'ResNet50      — Test Accuracy: {acc_resnet*100:.2f}%  | Test Loss: {loss_resnet:.4f}\n')
    f.write(f'Selisih       — ResNet50 unggul: {delta:+.2f}%\n\n')
    f.write('=' * 60 + '\n')
    f.write('Classification Report — CNN Baseline\n')
    f.write('=' * 60 + '\n')
    f.write(report_cnn)
    f.write('\n' + '=' * 60 + '\n')
    f.write('Classification Report — ResNet50\n')
    f.write('=' * 60 + '\n')
    f.write(report_resnet)
print(f'✅ Report perbandingan tersimpan: {report_path}')

# ============================================================
# 4. ROC CURVE — keduanya dalam 1 grafik per kelas
# ============================================================
y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
colors_cls = ['#2196F3', '#F44336', '#4CAF50']

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 5))
fig.suptitle('ROC Curve per Kelas — CNN Baseline vs ResNet50\nKlasifikasi Kerusakan Jalan',
             fontsize=13, fontweight='bold')

for i, (cls, color, ax) in enumerate(zip(CLASS_LABELS, colors_cls, axes)):
    fpr_cnn,    tpr_cnn,    _ = roc_curve(y_bin[:, i], prob_cnn[:, i])
    fpr_resnet, tpr_resnet, _ = roc_curve(y_bin[:, i], prob_resnet[:, i])
    auc_cnn    = auc(fpr_cnn,    tpr_cnn)
    auc_resnet = auc(fpr_resnet, tpr_resnet)

    ax.plot(fpr_cnn,    tpr_cnn,    color='#FF9800', linewidth=2,
            label=f'CNN Baseline (AUC={auc_cnn:.2f})')
    ax.plot(fpr_resnet, tpr_resnet, color='#2196F3', linewidth=2,
            label=f'ResNet50    (AUC={auc_resnet:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title(f'Kelas: {cls}', fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_roc_curve.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ ROC Curve tersimpan: {save_path}')

# ============================================================
# 5. F1-SCORE PER KELAS — grouped bar
# ============================================================
rd_cnn    = classification_report(y_true, pred_cnn,    target_names=CLASS_LABELS, output_dict=True)
rd_resnet = classification_report(y_true, pred_resnet, target_names=CLASS_LABELS, output_dict=True)

metrics   = ['precision', 'recall', 'f1-score']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Precision / Recall / F1-Score per Kelas\nCNN Baseline vs ResNet50',
             fontsize=13, fontweight='bold')

x     = np.arange(len(CLASS_LABELS))
width = 0.35

for ax, metric in zip(axes, metrics):
    vals_cnn    = [rd_cnn[c][metric]    for c in CLASS_LABELS]
    vals_resnet = [rd_resnet[c][metric] for c in CLASS_LABELS]

    b1 = ax.bar(x - width/2, vals_cnn,    width, label='CNN Baseline', color='#FF9800', edgecolor='black')
    b2 = ax.bar(x + width/2, vals_resnet, width, label='ResNet50',     color='#2196F3', edgecolor='black')

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_title(metric.capitalize(), fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = os.path.join(COMPARE_DIR, 'comparison_per_class_metrics.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Grafik per kelas tersimpan: {save_path}')

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print('\n' + '=' * 60)
print('📋 RINGKASAN PERBANDINGAN MODEL')
print('=' * 60)
print(f'   {"":25} {"CNN Baseline":>14} {"ResNet50":>12}')
print(f'   {"-" * 51}')
print(f'   {"Pretrained":<25} {"❌ Tidak":>14} {"✅ ImageNet":>12}')
print(f'   {"Test Accuracy":<25} {acc_cnn*100:>13.2f}% {acc_resnet*100:>11.2f}%')
print(f'   {"Test Loss":<25} {loss_cnn:>14.4f} {loss_resnet:>12.4f}')
for cls in CLASS_LABELS:
    f1_c = rd_cnn[cls]['f1-score']
    f1_r = rd_resnet[cls]['f1-score']
    print(f'   {f"F1 {cls}":<25} {f1_c:>14.4f} {f1_r:>12.4f}')
print(f'   {"-" * 51}')
print(f'   {"Selisih Accuracy":<25} {f"{delta:+.2f}% (ResNet50 unggul)":>26}')
print('=' * 60)
print(f'\n📁 Semua hasil tersimpan di: {COMPARE_DIR}')
