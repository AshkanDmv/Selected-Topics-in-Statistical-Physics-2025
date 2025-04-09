import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Principal Component Analysis (PCA)
def pca_plot():
    data_generate = load_iris()
    dataset = data_generate.data
    pca_method = PCA(n_components=2)
    reduced_dataset = pca_method.fit_transform(dataset)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(reduced_dataset[:, 0], reduced_dataset[:, 1], c=data_generate.target)
    plt.xlabel('Key Variable(1)')
    plt.ylabel('Key Varaible(2)')
    plt.title('PCA method for Iris Dataset')
    plt.colorbar()
    plt.show()

    return reduced_dataset


# Multidimensional Scaling (MDS)
def mds_plot():
    data_generate = load_iris()
    dataset = data_generate.data
    labels = data_generate.target
    mds_method = MDS(n_components=2, random_state=50)
    reduced_dataset = mds_method.fit_transform(dataset)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(reduced_dataset[:, 0], reduced_dataset[:, 1], c=labels, cmap='viridis')
    plt.title('MDS method for Iris Dataset')
    plt.xlabel('Key Variable(1)')
    plt.ylabel('Key Variable(2)')
    plt.colorbar(label='Species')
    plt.grid()
    plt.show()

    return reduced_dataset


# T-Distributed Stochastic Neighbor Embedding (t-SNE)
def tsne_plot():
    iris_dataset = load_iris()
    dataset = iris_dataset.data
    labels = iris_dataset.target
    tsne = TSNE(n_components=2, random_state=50, perplexity=25)
    reduced_dataset = tsne.fit_transform(dataset)

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 2, 1)
    plt.scatter(reduced_dataset[:, 0], reduced_dataset[:, 1], c=labels, cmap='viridis', edgecolor='limegreen')
    plt.title('t-SNE method for Iris Dataset')
    plt.xlabel('Key Variable(1)')
    plt.ylabel('Key Variable(2)')
    plt.colorbar(label='Species')
    plt.grid()
    plt.show()

    return reduced_dataset


# Linear Discriminant Analysis (LDA)
def lda_plot():
    iris_dataset = load_iris()
    dataset = iris_dataset.data
    labels = iris_dataset.target
    lda = LDA(n_components=2)
    reduced_dataset = lda.fit_transform(dataset, labels)

    plt.figure(figsize=(10, 8))
    plt.subplot(4, 2, 1)
    plt.scatter(reduced_dataset[:, 0], reduced_dataset[:, 1], c=labels, cmap='viridis', edgecolor='skyblue')
    plt.title('LDA method for Iris Dataset')
    plt.xlabel('Key Variable(1)')
    plt.ylabel('Key Variable(2)')
    plt.colorbar(label='Species')
    plt.grid()
    plt.show()

    return reduced_dataset


pca_plot()
mds_plot()
tsne_plot()
lda_plot()
