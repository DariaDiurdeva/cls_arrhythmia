import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MAX_VALUE_MISSING_ALLOWED = 0
PCA_NUMBER_COMPONENTS = 100


def delete_columns_with_missing_values(feature_array, max_missing_values=0):
    number_missing_values = sum(np.isnan(feature_array), 0)
    index_columns = np.where(number_missing_values > max_missing_values)
    return np.delete(feature_array, index_columns, axis=1)


# Scaling dataset
def scale_data(feature_array):
    scaler = StandardScaler()
    return scaler.fit_transform(feature_array)


# PCA
def apply_pca_analysis(feature_array, n_components=100):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(feature_array)


def prepare_data():
    X, Y = fetch_openml(name='arrhythmia', return_X_y=True)
    Z = delete_columns_with_missing_values(X, MAX_VALUE_MISSING_ALLOWED)
    Z = scale_data(Z)
    Z = apply_pca_analysis(Z, PCA_NUMBER_COMPONENTS)
    return Z, Y
