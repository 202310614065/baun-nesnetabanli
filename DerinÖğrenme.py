import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout,
                                     LSTM, Bidirectional, SimpleRNN, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# --- Sabitler ve Rastgelelik ---
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. VERİ OKUMA VE ÖN İŞLEME ---
df = pd.read_excel("evaluation_results.xlsx")
df.rename(columns={"Predicted": "Label"}, inplace=True)

def clean_text(text):
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r"http\S+", "", text)  # URL'leri kaldır
    text = re.sub(r"@\w+", "", text)       # @mention ifadelerini kaldır
    text = re.sub(r"#\w+", "", text)        # Hashtag'leri kaldır
    text = re.sub(r"[^\w\s]", "", text)      # Noktalama işaretlerini kaldır
    return text.strip()

df["cleaned_tweet"] = df["Tweets"].apply(clean_text)

# Label encoding
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["Label"])
num_classes = len(label_encoder.classes_)

# Özellik ve hedef
X_text = df["cleaned_tweet"].values
y = df["encoded_label"].values

# --- 2. DİZİLEŞTİRME (Tokenization & Padding) ---
vocab_size = 5000
max_length = 50
embedding_dim = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)
padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# --- 3. MODEL TANIMLARI ---

# Standart CNN modeli
def build_cnn_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# LSTM modeli
def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        LSTM(64, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# BLSTM modeli
def build_bilstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))),
        Dropout(0.4),
        Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(0.001))),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Düzeltilmiş RNN modeli
def build_rnn_model_fixed():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# GAN modeli
def build_gan_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# CNN+BiLSTM (Optimum) modeli
def build_cnn_bilstm_model_optimized():
    """
    Bu modelde, önce bir CNN katmanı ile n-gram özellikleri yakalanıyor,
    ardından BiLSTM katmanı ile sekans bilgisi öğreniliyor.
    """
    input_layer = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    
    # CNN katmanı: padding ile özellik boyutu korunuyor
    x = Conv1D(filters=64, kernel_size=3, activation='relu', 
               padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    
    # BiLSTM katmanı: dropout ve recurrent_dropout eklenmiş
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Tam bağlantılı katmanlar
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- 4. ÇAPRAZ DOĞRULAMA (K-Fold Cross Validation) ---
def cross_validate_model(model_builder, X, y, folds=5, batch_size=16, epochs=30, use_class_weight=False):
    from sklearn.metrics import precision_recall_fscore_support
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n----- Fold {fold}/{folds} -----")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        cw = None
        if use_class_weight:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            cw = dict(enumerate(weights))
        
        model = model_builder()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=epochs,
                  batch_size=batch_size,
                  class_weight=cw,
                  callbacks=callbacks,
                  verbose=0)
        
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        y_val_pred = np.argmax(model.predict(X_val), axis=1)
        report = classification_report(y_val, y_val_pred, output_dict=True)
        metrics = {
            "accuracy": val_acc,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"]
        }
        print(f"Fold {fold} - Accuracy: {val_acc:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        fold_metrics.append(metrics)
    return fold_metrics

# Çapraz doğrulama: Mevcut modeller
print(">>> CNN için Çapraz Doğrulama")
cnn_cv_metrics = cross_validate_model(build_cnn_model, padded, y, folds=5, batch_size=8, epochs=30, use_class_weight=False)

print("\n>>> LSTM için Çapraz Doğrulama")
lstm_cv_metrics = cross_validate_model(build_lstm_model, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

print("\n>>> BLSTM için Çapraz Doğrulama")
bilstm_cv_metrics = cross_validate_model(build_bilstm_model, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

print("\n>>> Düzeltilmiş RNN için Çapraz Doğrulama")
rnn_cv_metrics = cross_validate_model(build_rnn_model_fixed, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

print("\n>>> GAN için Çapraz Doğrulama")
gan_cv_metrics = cross_validate_model(build_gan_model, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

print("\n>>> CNN+BiLSTM (Optimum) için Çapraz Doğrulama")
cnn_bilstm_cv_metrics = cross_validate_model(build_cnn_bilstm_model_optimized, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

def average_metrics(metrics_list):
    avg_metrics = {}
    for key in metrics_list[0]:
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    return avg_metrics

cnn_avg = average_metrics(cnn_cv_metrics)
lstm_avg = average_metrics(lstm_cv_metrics)
bilstm_avg = average_metrics(bilstm_cv_metrics)
rnn_avg = average_metrics(rnn_cv_metrics)
gan_avg = average_metrics(gan_cv_metrics)
cnn_bilstm_avg = average_metrics(cnn_bilstm_cv_metrics)

print("\n--- Ortalama Çapraz Doğrulama Metrikleri ---")
print("CNN       :", cnn_avg)
print("LSTM      :", lstm_avg)
print("BLSTM     :", bilstm_avg)
print("RNN       :", rnn_avg)
print("GAN       :", gan_avg)
print("CNN+BiLSTM (Optimum) :", cnn_bilstm_avg)

# --- 5. FINAL MODEL EĞİTİMİ VE TAHMİN ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    padded, y, test_size=0.2, random_state=42, stratify=y
)

def train_final_model(model_builder, X_train, y_train, X_test, batch_size, epochs, use_class_weight=False):
    cw = None
    if use_class_weight:
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw = dict(enumerate(weights))
    
    model = model_builder()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=epochs,
              batch_size=batch_size,
              class_weight=cw,
              callbacks=callbacks,
              verbose=1)
    y_pred_test = np.argmax(model.predict(X_test), axis=1)
    return model, y_pred_test

print("\n>>> Final Model Eğitimi")

cnn_model, y_pred_cnn_test = train_final_model(build_cnn_model, X_train_full, y_train_full, X_test, batch_size=8, epochs=30, use_class_weight=False)
lstm_model, y_pred_lstm_test = train_final_model(build_lstm_model, X_train_full, y_train_full, X_test, batch_size=16, epochs=30, use_class_weight=True)
bilstm_model, y_pred_bilstm_test = train_final_model(build_bilstm_model, X_train_full, y_train_full, X_test, batch_size=16, epochs=30, use_class_weight=True)
rnn_model, y_pred_rnn_test = train_final_model(build_rnn_model_fixed, X_train_full, y_train_full, X_test, batch_size=16, epochs=30, use_class_weight=True)
gan_model, y_pred_gan_test = train_final_model(build_gan_model, X_train_full, y_train_full, X_test, batch_size=16, epochs=30, use_class_weight=True)
cnn_bilstm_model, y_pred_cnn_bilstm_test = train_final_model(build_cnn_bilstm_model_optimized, X_train_full, y_train_full, X_test, batch_size=16, epochs=30, use_class_weight=True)

# --- 6. DETAYLI DEĞERLENDİRME ---
def get_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    }
    return metrics

cnn_metrics = get_metrics(y_test, y_pred_cnn_test)
lstm_metrics = get_metrics(y_test, y_pred_lstm_test)
bilstm_metrics = get_metrics(y_test, y_pred_bilstm_test)
rnn_metrics = get_metrics(y_test, y_pred_rnn_test)
gan_metrics = get_metrics(y_test, y_pred_gan_test)
cnn_bilstm_metrics = get_metrics(y_test, y_pred_cnn_bilstm_test)

print("\n----- Final Test Metrikleri -----")
print("CNN       :", cnn_metrics)
print("LSTM      :", lstm_metrics)
print("BLSTM     :", bilstm_metrics)
print("RNN       :", rnn_metrics)
print("GAN       :", gan_metrics)
print("CNN+BiLSTM (Optimum) :", cnn_bilstm_metrics)

def print_detailed_results(title, y_true, y_pred):
    print(f"\n----- {title} -----")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nKarışıklık Matrisi:")
    print(cm_df)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print_detailed_results("Test - CNN", y_test, y_pred_cnn_test)
print_detailed_results("Test - LSTM", y_test, y_pred_lstm_test)
print_detailed_results("Test - BLSTM", y_test, y_pred_bilstm_test)
print_detailed_results("Test - RNN (Düzeltilmiş)", y_test, y_pred_rnn_test)
print_detailed_results("Test - GAN", y_test, y_pred_gan_test)
print_detailed_results("Test - CNN+BiLSTM (Optimum)", y_test, y_pred_cnn_bilstm_test)

# --- 7. GRAFİKSEL GÖSTERİMLER ---
# (a) Grup Bar Grafiği: Her model için Test Accuracy, CV Mean Accuracy, Macro F1 ve Weighted F1
models = ["CNN", "LSTM", "BLSTM", "RNN", "GAN", "CNN+BiLSTM"]
test_acc = [
    cnn_metrics["Test Accuracy"], 
    lstm_metrics["Test Accuracy"], 
    bilstm_metrics["Test Accuracy"],
    rnn_metrics["Test Accuracy"],
    gan_metrics["Test Accuracy"],
    cnn_bilstm_metrics["Test Accuracy"]
]
cv_acc = [
    cnn_avg["accuracy"], 
    lstm_avg["accuracy"], 
    bilstm_avg["accuracy"],
    rnn_avg["accuracy"],
    gan_avg["accuracy"],
    cnn_bilstm_avg["accuracy"]
]
macro_f1 = [
    cnn_metrics["Macro F1"], 
    lstm_metrics["Macro F1"], 
    bilstm_metrics["Macro F1"],
    rnn_metrics["Macro F1"],
    gan_metrics["Macro F1"],
    cnn_bilstm_metrics["Macro F1"]
]
weighted_f1 = [
    cnn_metrics["Weighted F1"], 
    lstm_metrics["Weighted F1"], 
    bilstm_metrics["Weighted F1"],
    rnn_metrics["Weighted F1"],
    gan_metrics["Weighted F1"],
    cnn_bilstm_metrics["Weighted F1"]
]

x = np.arange(len(models))
width = 0.15

fig1, ax1 = plt.subplots(figsize=(12,7))
rects1 = ax1.bar(x - 2.5*width, test_acc, width, label='Test Accuracy')
rects2 = ax1.bar(x - 1.5*width, cv_acc, width, label='CV Mean Accuracy')
rects3 = ax1.bar(x - 0.5*width, macro_f1, width, label='Macro F1')
rects4 = ax1.bar(x + 0.5*width, weighted_f1, width, label='Weighted F1')

ax1.set_ylabel('Değer')
ax1.set_title('Modellere Göre Metrik Karşılaştırması (Optimizasyonlu)')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.show()

# (b) Confusion Matrix: 2x3 grid (6 model için)
fig2, axs = plt.subplots(2, 3, figsize=(22,12))
cm_list = [
    ("CNN", confusion_matrix(y_test, y_pred_cnn_test)),
    ("LSTM", confusion_matrix(y_test, y_pred_lstm_test)),
    ("BLSTM", confusion_matrix(y_test, y_pred_bilstm_test)),
    ("RNN", confusion_matrix(y_test, y_pred_rnn_test)),
    ("GAN", confusion_matrix(y_test, y_pred_gan_test)),
    ("CNN+BiLSTM", confusion_matrix(y_test, y_pred_cnn_bilstm_test))
]

for idx, (model_name, cm) in enumerate(cm_list):
    row, col = divmod(idx, 3)
    im = axs[row, col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[row, col].set_title(f'{model_name} Confusion Matrix')
    tick_marks = np.arange(len(label_encoder.classes_))
    axs[row, col].set_xticks(tick_marks)
    axs[row, col].set_xticklabels(label_encoder.classes_)
    axs[row, col].set_yticks(tick_marks)
    axs[row, col].set_yticklabels(label_encoder.classes_)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[row, col].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
    axs[row, col].set_ylabel('Gerçek')
    axs[row, col].set_xlabel('Tahmin')

plt.tight_layout()
plt.show()

# (c) Gerçek vs Tahmin Dağılımı grafiği
actual_counts = np.bincount(y_test, minlength=num_classes)
cnn_counts = np.bincount(y_pred_cnn_test, minlength=num_classes)
lstm_counts = np.bincount(y_pred_lstm_test, minlength=num_classes)
bilstm_counts = np.bincount(y_pred_bilstm_test, minlength=num_classes)
rnn_counts = np.bincount(y_pred_rnn_test, minlength=num_classes)
gan_counts = np.bincount(y_pred_gan_test, minlength=num_classes)
cnn_bilstm_counts = np.bincount(y_pred_cnn_bilstm_test, minlength=num_classes)

x_labels = label_encoder.classes_
x = np.arange(len(x_labels))
width = 0.1

fig3, ax3 = plt.subplots(figsize=(12,7))
rects_actual = ax3.bar(x - 2.5*width, actual_counts, width, label='Gerçek Etiketler')
rects_cnn = ax3.bar(x - 1.5*width, cnn_counts, width, label='CNN Tahmin')
rects_lstm = ax3.bar(x - 0.5*width, lstm_counts, width, label='LSTM Tahmin')
rects_bilstm = ax3.bar(x + 0.5*width, bilstm_counts, width, label='BLSTM Tahmin')
rects_rnn = ax3.bar(x + 1.5*width, rnn_counts, width, label='RNN Tahmin')
rects_gan = ax3.bar(x + 2.5*width, gan_counts, width, label='GAN Tahmin')
rects_cnn_bilstm = ax3.bar(x + 3.5*width, cnn_bilstm_counts, width, label='CNN+BiLSTM Tahmin')

ax3.set_ylabel('Örnek Sayısı')
ax3.set_title('Gerçek vs Tahmin Dağılımı (Optimizasyonlu)')
ax3.set_xticks(x)
ax3.set_xticklabels(x_labels)
ax3.legend()

def autolabel_bar(rects, axis):
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f'{height}',
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha='center', va='bottom')

autolabel_bar(rects_actual, ax3)
autolabel_bar(rects_cnn, ax3)
autolabel_bar(rects_lstm, ax3)
autolabel_bar(rects_bilstm, ax3)
autolabel_bar(rects_rnn, ax3)
autolabel_bar(rects_gan, ax3)
autolabel_bar(rects_cnn_bilstm, ax3)

plt.tight_layout()
plt.show()

# --- 8. EK: ROC, Precision-Recall ve Öğrenme Eğrileri GÖSTERİMLERİ (Tüm Modeller için) ---
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Final modellerimizi bir sözlükte toplayalım.
final_models = {
    "CNN": cnn_model,
    "LSTM": lstm_model,
    "BLSTM": bilstm_model,
    "RNN": rnn_model,
    "GAN": gan_model,
    "CNN+BiLSTM": cnn_bilstm_model
}

# ROC ve PR grafikleri için binary ve çok sınıflı durumları ayıralım.
if num_classes == 2:
    # Binary sınıflandırma durumunda:
    fig_roc, ax_roc = plt.subplots(figsize=(10,8))
    fig_pr, ax_pr = plt.subplots(figsize=(10,8))
    
    for name, model in final_models.items():
        y_score = model.predict(X_test)
        # Çıktı boyutu (n,1) ise (n,2) haline getirelim.
        if y_score.ndim == 1 or y_score.shape[1] != 2:
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)
            if y_score.shape[1] != 2:
                y_score = np.concatenate([1 - y_score, y_score], axis=1)
                
        # ROC: y_score[:,1] kullanılıyor.
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_val:.2f})')
        
        # Precision-Recall:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score[:, 1])
        average_prec = average_precision_score(y_test, y_score[:, 1])
        ax_pr.plot(recall_vals, precision_vals, lw=2, label=f'{name} (AP = {average_prec:.2f})')
    
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve - Binary Classification', fontsize=14)
    ax_roc.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curve - Binary Classification', fontsize=14)
    ax_pr.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.show()
    
else:
    # Çok sınıflı durumda:
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    # ROC için 2x3 grid
    fig_roc, axs_roc = plt.subplots(2, 3, figsize=(22,12))
    axs_roc = axs_roc.flatten()
    
    # PR için 2x3 grid
    fig_pr, axs_pr = plt.subplots(2, 3, figsize=(22,12))
    axs_pr = axs_pr.flatten()
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
    
    for ax_r, ax_p, (name, model), color in zip(axs_roc, axs_pr, final_models.items(), colors):
        y_score = model.predict(X_test)
        
        # ROC hesaplamaları
        fpr = dict()
        tpr = dict()
        roc_auc_dict = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_dict["micro"] = auc(fpr["micro"], tpr["micro"])
        
        ax_r.plot(fpr["micro"], tpr["micro"], color=color, linestyle=':', linewidth=4,
                  label=f'Micro-average ROC (AUC = {roc_auc_dict["micro"]:.2f})')
        for i in range(num_classes):
            ax_r.plot(fpr[i], tpr[i], color=color, lw=2,
                      label=f'ROC (class {label_encoder.classes_[i]}) (AUC = {roc_auc_dict[i]:.2f})')
        ax_r.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_r.set_xlim([0.0, 1.0])
        ax_r.set_ylim([0.0, 1.05])
        ax_r.set_xlabel('False Positive Rate', fontsize=12)
        ax_r.set_ylabel('True Positive Rate', fontsize=12)
        ax_r.set_title(f'ROC - {name}', fontsize=14)
        ax_r.legend(loc="lower right", fontsize=8)
        
        # Precision-Recall hesaplamaları
        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()
        for i in range(num_classes):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            average_precision_dict[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
        precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
        average_precision_dict["micro"] = average_precision_score(y_test_bin, y_score, average="micro")
        
        ax_p.plot(recall_dict["micro"], precision_dict["micro"],
                  label=f'Micro-average PR (AP = {average_precision_dict["micro"]:.2f})',
                  color=color, linestyle=':', linewidth=4)
        for i in range(num_classes):
            ax_p.plot(recall_dict[i], precision_dict[i], color=color, lw=2,
                      label=f'PR (class {label_encoder.classes_[i]}) (AP = {average_precision_dict[i]:.2f})')
        ax_p.set_xlabel('Recall', fontsize=12)
        ax_p.set_ylabel('Precision', fontsize=12)
        ax_p.set_title(f'Precision-Recall - {name}', fontsize=14)
        ax_p.legend(loc="lower left", fontsize=8)
    
    plt.tight_layout()
    plt.show()

# --- EKSTRA: Öğrenme Eğrileri (Training/Validation Loss ve Accuracy) ---
# Final model eğitimi sırasında callback'ler sayesinde en iyi ağırlıklar yüklendiğinden
# örnek bir modelin geçmişini (history) inceleyelim.
# Aşağıdaki örnekte CNN modelini tekrar eğitimle history elde edip görselleştiriyoruz.

cnn_model_for_learning, history = None, None
{
    "model": cnn_model_for_learning,
    "history": history
}
# Yeniden eğitim örneği:
cnn_model_for_learning = build_cnn_model()
history = cnn_model_for_learning.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=30,
    batch_size=8,
    verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)

# Öğrenme eğrilerini çizdirme:
fig_learn, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14,6))
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_title('Loss Eğrisi')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()

ax_acc.plot(history.history['accuracy'], label='Training Accuracy')
ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax_acc.set_title('Accuracy Eğrisi')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.legend()

plt.tight_layout()
plt.show()
