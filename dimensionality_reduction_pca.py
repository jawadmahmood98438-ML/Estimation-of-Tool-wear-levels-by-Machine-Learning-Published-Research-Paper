from sklearn.decomposition import PCA

def reduce_dimensions(X, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca