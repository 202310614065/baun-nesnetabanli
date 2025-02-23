import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- Sabitler ve Rastgelelik ---
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. VERİ OKUMA VE ÖN İŞLEME ---
df = pd.read_excel("evaluation_results.xlsx")
df.rename(columns={"Predicted": "Label"}, inplace=True)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # URL'leri kaldır
    text = re.sub(r"@\w+", "", text)      # @mention ifadelerini kaldır
    text = re.sub(r"#\w+", "", text)       # Hashtag'leri kaldır
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
def build_cnn_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        LSTM(64, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))),
        Dropout(0.4),
        Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(0.001))),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
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
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
        
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

print(">>> CNN için Çapraz Doğrulama")
cnn_cv_metrics = cross_validate_model(build_cnn_model, padded, y, folds=5, batch_size=8, epochs=30, use_class_weight=False)

print("\n>>> LSTM için Çapraz Doğrulama")
lstm_cv_metrics = cross_validate_model(build_lstm_model, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

print("\n>>> BLSTM için Çapraz Doğrulama")
bilstm_cv_metrics = cross_validate_model(build_bilstm_model, padded, y, folds=5, batch_size=16, epochs=30, use_class_weight=True)

def average_metrics(metrics_list):
    avg_metrics = {}
    for key in metrics_list[0]:
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    return avg_metrics

cnn_avg = average_metrics(cnn_cv_metrics)
lstm_avg = average_metrics(lstm_cv_metrics)
bilstm_avg = average_metrics(bilstm_cv_metrics)

print("\n--- Ortalama Çapraz Doğrulama Metrikleri ---")
print("CNN  :", cnn_avg)
print("LSTM :", lstm_avg)
print("BLSTM:", bilstm_avg)

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
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
    
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

print("\n----- Final Test Metrikleri -----")
print("CNN  :", cnn_metrics)
print("LSTM :", lstm_metrics)
print("BLSTM:", bilstm_metrics)

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

# --- 7. GRAFİKSEL GÖSTERİMLER ---
# Yeni görselleştirmeler için matplotlib kullanıyoruz.

# (a) Grup Bar Grafiği: Her model için Test Accuracy, CV Mean Accuracy, Macro F1 ve Weighted F1
models = ["CNN", "LSTM", "BLSTM"]
test_acc = [cnn_metrics["Test Accuracy"], lstm_metrics["Test Accuracy"], bilstm_metrics["Test Accuracy"]]
cv_acc = [cnn_avg["accuracy"], lstm_avg["accuracy"], bilstm_avg["accuracy"]]
macro_f1 = [cnn_metrics["Macro F1"], lstm_metrics["Macro F1"], bilstm_metrics["Macro F1"]]
weighted_f1 = [cnn_metrics["Weighted F1"], lstm_metrics["Weighted F1"], bilstm_metrics["Weighted F1"]]

x = np.arange(len(models))
width = 0.2

fig1, ax1 = plt.subplots(figsize=(10,6))
rects1 = ax1.bar(x - 1.5*width, test_acc, width, label='Test Accuracy')
rects2 = ax1.bar(x - 0.5*width, cv_acc, width, label='CV Mean Accuracy')
rects3 = ax1.bar(x + 0.5*width, macro_f1, width, label='Macro F1')
rects4 = ax1.bar(x + 1.5*width, weighted_f1, width, label='Weighted F1')

ax1.set_ylabel('Değer')
ax1.set_title('Modellere Göre Metrik Karşılaştırması')
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

# (b) Confusion Matrix: 2x2 grid (üç model için, dördüncü hücre boş)
cnn_cm = confusion_matrix(y_test, y_pred_cnn_test)
lstm_cm = confusion_matrix(y_test, y_pred_lstm_test)
bilstm_cm = confusion_matrix(y_test, y_pred_bilstm_test)

fig2, axs = plt.subplots(2, 2, figsize=(12,10))
cm_list = [("CNN", cnn_cm), ("LSTM", lstm_cm), ("BLSTM", bilstm_cm)]

for idx, (model_name, cm) in enumerate(cm_list):
    row, col = divmod(idx, 2)
    im = axs[row, col].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[row, col].set_title(f'{model_name} Confusion Matrix')
    tick_marks = np.arange(len(label_encoder.classes_))
    axs[row, col].set_xticks(tick_marks)
    axs[row, col].set_xticklabels(label_encoder.classes_)
    axs[row, col].set_yticks(tick_marks)
    axs[row, col].set_yticklabels(label_encoder.classes_)
    # Hücrelere değer ekleme
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[row, col].text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
    axs[row, col].set_ylabel('Gerçek')
    axs[row, col].set_xlabel('Tahmin')

# Dördüncü subplot boş bırakılıyor
axs[1, 1].axis('off')
plt.tight_layout()
plt.show()

# (c) Gerçek vs Tahmin Dağılımı: Test verisi için gerçek etiket sayıları ile her modelin tahmin ettiği etiket sayıları
actual_counts = np.bincount(y_test, minlength=num_classes)
cnn_counts = np.bincount(y_pred_cnn_test, minlength=num_classes)
lstm_counts = np.bincount(y_pred_lstm_test, minlength=num_classes)
bilstm_counts = np.bincount(y_pred_bilstm_test, minlength=num_classes)

x_labels = label_encoder.classes_
x = np.arange(len(x_labels))
width = 0.2

fig3, ax3 = plt.subplots(figsize=(10,6))
rects_actual = ax3.bar(x - 1.5*width, actual_counts, width, label='Gerçek Etiketler')
rects_cnn = ax3.bar(x - 0.5*width, cnn_counts, width, label='CNN Tahmin')
rects_lstm = ax3.bar(x + 0.5*width, lstm_counts, width, label='LSTM Tahmin')
rects_bilstm = ax3.bar(x + 1.5*width, bilstm_counts, width, label='BLSTM Tahmin')

ax3.set_ylabel('Örnek Sayısı')
ax3.set_title('Gerçek vs Tahmin Dağılımı')
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

plt.tight_layout()
plt.show()
