from numpy import ndarray
from sklearn.manifold import TSNE


class T_SNE:
    @staticmethod
    def fit(x:ndarray):
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(x)
