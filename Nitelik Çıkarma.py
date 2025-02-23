import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Gerekli nltk verilerini indir (ilk defa çalıştırıyorsanız):
# nltk.download('punkt')
# nltk.download('stopwords')

# 1) Excel dosyasını oku
df = pd.read_excel("evaluation_results.xlsx")

# 2) "Olumsuz" etiketli satırları al
df_neg = df[df["Predicted"] == "Olumsuz"].copy()

# 3) Temizleme fonksiyonu: Küçük harfe çevirme, URL, mention ve noktalama işaretlerini temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    # Türkçe karakterleri koruyarak alfanümerik olmayan karakterleri kaldırıyoruz
    text = re.sub(r"[^a-zçğıöşü0-9\s]", "", text)
    return text

df_neg["TemizTweet"] = df_neg["Tweets"].astype(str).apply(clean_text)

# 4) Stopword temizliği: Türkçe stopword'leri çıkar ve kısa kelimeleri filtrele
turkish_stops = set(stopwords.words("turkish"))
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text, language="turkish")
    tokens = [t for t in tokens if t not in turkish_stops and len(t) > 2]
    return " ".join(tokens)

df_neg["TemizTweet"] = df_neg["TemizTweet"].apply(remove_stopwords)

# (Opsiyonel) 5) Ek ön işleme: Eğer mevcutsa kök bulma/lemmatizasyon eklenebilir.
# Türkçe için özel kütüphaneler kullanılabilir (ör. Zemberek veya diğerleri).

# 6) TF-IDF vektörleştirme (unigram ve bigram); parametrelerde ince ayar yapılabilir
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X = vectorizer.fit_transform(df_neg["TemizTweet"])

# 7) Küme sayısını belirlemek için Elbow yöntemi ve Silhouette skoru hesaplama
range_n_clusters = range(2, 10)
inertias = []
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)
    labels = kmeans_temp.labels_
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"Küme sayısı: {n_clusters} - Silhouette Skoru: {score:.3f}")

# Elbow grafiği
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertias, marker='o', linestyle='--', color='b')
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi")

# Silhouette skoru grafiği
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--', color='g')
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette Skoru")
plt.title("Silhouette Skoru")
plt.tight_layout()
plt.show()

# Kullanıcı veya gözlemlemede optimum küme sayısını belirleyin; örneğin 6 seçildi
NUM_CLUSTERS = 6
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X)
df_neg["ClusterID"] = kmeans.labels_

# 8) (Opsiyonel) Manuel küme etiketleme:
cluster_label_map = {
    0: "Siyasi Eleştiri",
    1: "Eğitim Kalitesi Eleştirisi",
    2: "Yurt/Barınma-Ulaşım Sorunları",
    3: "Ücret/Ekonomik Eleştiri",
    4: "Yönetim/İdari Eleştiri",
    5: "Diğer"
}
df_neg["Sikayet_Konusu"] = df_neg["ClusterID"].map(cluster_label_map)

# 9) Her kümedeki tweet sayısını hesaplayalım
cluster_counts = df_neg["Sikayet_Konusu"].value_counts().sort_index()
print("\nKüme Dağılımları:")
print(cluster_counts)

# 10) Sonuçları bar grafikte görselleştirme
plt.figure(figsize=(10,6))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color="steelblue")
plt.title("Olumsuz Tweetlerin Şikayet Konusu Dağılımı")
plt.xlabel("Şikayet Konusu")
plt.ylabel("Tweet Sayısı")
plt.xticks(rotation=45, ha="right")

# Her barın üzerine sayı etiketlerini ekleyelim
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()
