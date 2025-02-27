import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Pipeline ile model oluşturma
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# 1. VERİNİN OKUNMASI ve TEMİZLENMESİ
# ------------------------------------------------------------------
df = pd.read_excel("evaluation_results.xlsx")
df.rename(columns={"Predicted": "Label"}, inplace=True)

def clean_text(text):
    """
    Tweet metinlerinden URL, @mentions, hashtag ve gereksiz boşlukları temizler.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text.strip()

df["cleaned_tweet"] = df["Tweets"].apply(clean_text)

# ------------------------------------------------------------------
# 2. ETİKET DÖNÜŞÜMÜ (Label Encoding)
# ------------------------------------------------------------------
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["Label"])

# Özellik (X) ve hedef (y) değişkenler
X = df["cleaned_tweet"].values
y = df["encoded_label"].values

# Toplam sınıf sayısı
num_classes = len(label_encoder.classes_)

# ------------------------------------------------------------------
# 3. EĞİTİM/TEST AYRIMI (Stratified Split)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------------
# 4. MODELLERİN PIPELINE İLE TANIMLANMASI
#    (TF-IDF dönüştürme ve sınıflandırma işlemi birlikte yapılır)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "SVM": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', SVC(C=1.0, probability=True, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42))
    ])
}

# ------------------------------------------------------------------
# 5. MODEL EĞİTİMİ ve TEST SETİ DEĞERLENDİRME (Cross-validation ile)
# ------------------------------------------------------------------
results_test = {}  # Test sonuçlarını saklayacağız

for name, pipeline in models.items():
    print(f"Training {name}...")
    # Eğitim
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Test metrikleri
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cr_text = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    # Çapraz doğrulama (örnek: 5-fold)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    results_test[name] = {
        "model": pipeline,
        "accuracy": acc,
        "confusion_matrix": cm,
        "report_dict": cr_dict,
        "report_text": cr_text,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "y_pred": y_pred  # Tahminleri saklıyoruz
    }

# ------------------------------------------------------------------
# 6. SONUÇLARIN EXCEL'E KAYDEDİLMESİ (Test sonuçları üzerinden)
# ------------------------------------------------------------------
with pd.ExcelWriter("machine_learn_sonuc.xlsx") as writer:
    # Genel metrikler için özet tablo
    rows_test = []
    for name, res in results_test.items():
        rd = res["report_dict"]
        macro_f1 = rd["macro avg"]["f1-score"]
        weighted_f1 = rd["weighted avg"]["f1-score"]
        rows_test.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "CV_Mean_Accuracy": res["cv_mean"],
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        })
    df_test_overall = pd.DataFrame(rows_test)
    df_test_overall.to_excel(writer, sheet_name="Test_Overall", index=False)
    
    # Detaylı sayfalar (her model için Confusion Matrix ve Classification Report)
    for name, res in results_test.items():
        cm_df = pd.DataFrame(
            res["confusion_matrix"],
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )
        cr_df = pd.DataFrame(res["report_dict"]).transpose()
    
        cm_df.to_excel(writer, sheet_name=f"Test_{name}", startrow=0, startcol=0)
        cr_df.to_excel(writer, sheet_name=f"Test_{name}", startrow=len(cm_df)+2, startcol=0)

print("\nTüm metrikler ve değerlendirmeler 'machine_learn_sonuc.xlsx' dosyasına kaydedildi.")

# ------------------------------------------------------------------
# 7. GRAFİKSEL GÖSTERİMLER
# ------------------------------------------------------------------

# (a) Grup Bar Grafiği: Test Verisi Model Metrikleri
models_list = df_test_overall['Model']
x = np.arange(len(models_list))  # Model indexleri
bar_width = 0.2  # Her bar'ın genişliği

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - 1.5*bar_width, df_test_overall['Accuracy'], bar_width, label='Accuracy')
bars2 = ax.bar(x - 0.5*bar_width, df_test_overall['CV_Mean_Accuracy'], bar_width, label='CV Mean Accuracy')
bars3 = ax.bar(x + 0.5*bar_width, df_test_overall['Macro_F1'], bar_width, label='Macro F1')
bars4 = ax.bar(x + 1.5*bar_width, df_test_overall['Weighted_F1'], bar_width, label='Weighted F1')

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)

ax.set_ylabel('Metrik Değeri')
ax.set_title('Test Verisi: Model Karşılaştırma Metrikleri')
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.legend()

plt.tight_layout()
plt.show()

# (b) Confusion Matrix Karşılaştırması (2x2 Grid)
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
axes_cm = axes_cm.flatten()

for ax, name in zip(axes_cm, results_test.keys()):
    cm = results_test[name]["confusion_matrix"]
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    ax.set_title(name)
    tick_marks = np.arange(len(label_encoder.classes_))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(label_encoder.classes_)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(label_encoder.classes_)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
fig_cm.suptitle('Test Verisi: Model Confusion Matrix Karşılaştırması', fontsize=16)
fig_cm.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# (c) Gerçek Etiketler ile Model Tahminlerinin Karşılaştırılması
actual_counts = np.bincount(y_test, minlength=len(label_encoder.classes_))
classes = label_encoder.classes_
x = np.arange(len(classes))
bar_width = 0.15

fig_compare, ax = plt.subplots(figsize=(10, 6))
bars_actual = ax.bar(x - bar_width, actual_counts, bar_width, label="Gerçek Etiketler")

bars_models = []
for idx, (name, res) in enumerate(results_test.items()):
    pred_counts = np.bincount(res["y_pred"], minlength=len(label_encoder.classes_))
    bar = ax.bar(x + idx*bar_width, pred_counts, bar_width, label=f"{name} Tahmin")
    bars_models.append(bar)

def autolabel_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel_bars(bars_actual)
for bar in bars_models:
    autolabel_bars(bar)

ax.set_ylabel('Örnek Sayısı')
ax.set_title('Test Verisi: Gerçek Etiketler ile Model Tahminlerinin Karşılaştırılması')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 8. EK: ROC ve Precision-Recall Grafikleri (Her model için)
# ------------------------------------------------------------------
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# y_test'i çok sınıflı hesaplamalar için binarize ediyoruz
y_test_bin = label_binarize(y_test, classes=range(num_classes))
# Binary durumunda, eğer y_test_bin tek sütunluyse, eksik sütunu ekleyelim:
if num_classes == 2 and (y_test_bin.ndim == 1 or y_test_bin.shape[1] == 1):
    y_test_bin = y_test_bin.reshape(-1, 1)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.concatenate([1 - y_test_bin, y_test_bin], axis=1)

# ROC eğrilerini 2x2 grid şeklinde çizelim
fig_roc, axes_roc = plt.subplots(2, 2, figsize=(16, 12))
axes_roc = axes_roc.flatten()

for i, (name, res) in enumerate(results_test.items()):
    model = res["model"]
    try:
        y_score = model.predict_proba(X_test)
    except Exception as e:
        y_score = model.decision_function(X_test)
    y_score = np.array(y_score)
    # Eğer y_score 1D veya (n,1) ise, onu (n,2) yapalım:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        y_score = y_score.reshape(-1, 1)
        y_score = np.concatenate([1 - y_score, y_score], axis=1)
    print(f"{name}: y_score shape after conversion: {y_score.shape}")
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(num_classes):
        fpr[j], tpr[j], _ = roc_curve(y_test_bin[:, j], y_score[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    ax = axes_roc[i]
    ax.plot(fpr["micro"], tpr["micro"],
            label='Micro-average ROC (AUC = {0:0.2f})'.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
    for j in range(num_classes):
        ax.plot(fpr[j], tpr[j], color=colors[j % len(colors)], lw=2,
                label='ROC (class {0}) (AUC = {1:0.2f})'.format(label_encoder.classes_[j], roc_auc[j]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {name}', fontsize=14)
    ax.legend(loc="lower right", fontsize=10, ncol=2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.3)
plt.show()

# Precision-Recall eğrilerini 2x2 grid şeklinde çizelim
fig_pr, axes_pr = plt.subplots(2, 2, figsize=(16, 12))
axes_pr = axes_pr.flatten()

for i, (name, res) in enumerate(results_test.items()):
    model = res["model"]
    try:
        y_score = model.predict_proba(X_test)
    except Exception as e:
        y_score = model.decision_function(X_test)
    y_score = np.array(y_score)
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        y_score = y_score.reshape(-1, 1)
        y_score = np.concatenate([1 - y_score, y_score], axis=1)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for j in range(num_classes):
        precision[j], recall[j], _ = precision_recall_curve(y_test_bin[:, j], y_score[:, j])
        average_precision[j] = average_precision_score(y_test_bin[:, j], y_score[:, j])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_bin, y_score, average="micro")
    
    ax = axes_pr[i]
    ax.plot(recall["micro"], precision["micro"],
            label='Micro-average PR (AP = {0:0.2f})'.format(average_precision["micro"]),
            color='gold', linestyle=':', linewidth=4)
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
    for j in range(num_classes):
        ax.plot(recall[j], precision[j], color=colors[j % len(colors)], lw=2,
                label='PR (class {0}) (AP = {1:0.2f})'.format(label_encoder.classes_[j], average_precision[j]))
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {name}', fontsize=14)
    ax.legend(loc="lower left", fontsize=10, ncol=2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.3)
plt.show()
