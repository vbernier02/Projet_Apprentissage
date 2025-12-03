import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score


def run_kmeans(X_processed_df, y_encoded, K):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_processed_df)

    silhouette = silhouette_score(X_processed_df, clusters)
    ari = adjusted_rand_score(y_encoded, clusters)

    contingency = pd.crosstab(y_encoded, clusters)

    return clusters, silhouette, ari, contingency


def run_cah(X_processed_df, y_encoded, K):
    cah = AgglomerativeClustering(n_clusters=K)
    clusters = cah.fit_predict(X_processed_df)

    silhouette = silhouette_score(X_processed_df, clusters)
    ari = adjusted_rand_score(y_encoded, clusters)

    contingency = pd.crosstab(y_encoded, clusters)

    return clusters, silhouette, ari, contingency
