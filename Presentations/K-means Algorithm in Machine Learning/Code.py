import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cell_differentiation_data(num_samples=100, num_genes=10):
    np.random.seed(42)
    cell_types = ['Type A', 'Type B', 'Type C', 'Type D']
    num_cell_types = len(cell_types)
    data = []

    for i in range(num_cell_types):
        means = (np.random.rand(num_genes) * 10) + (list(cell_types).index(cell_types[i]) * 5)
        for j in range(num_samples // num_cell_types):
            data.append(np.random.normal(means, 1))

    gene_expression_df = pd.DataFrame(data)
    column_names = []
    for k in range(num_genes):
        column_names.append(f'Gene {k + 1}')

    gene_expression_df.columns = column_names
    samples_per_type = num_samples // num_cell_types
    cell_type_list = []

    for l in range(num_cell_types):
        for m in range(samples_per_type):
            cell_type_list.append(cell_types[l])

    cell_type_array = np.array(cell_type_list)

    return gene_expression_df, cell_type_array


def plot_data(data, title='Gene Expression Data for Cell Differentiation'):
    plt.figure(figsize=(8, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Gene 1 Expression')
    plt.ylabel('Gene 2 Expression')
    plt.grid()
    plt.show()


def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return (data - mean) / std


def initialize_centroids(data, k):
    m, n = data.shape
    indices = np.random.choice(m, k, replace=False)

    return data.iloc[indices].values


def assign_clusters(data, centroids):
    distances = np.zeros((data.shape[0], centroids.shape[0]))

    for i in range(centroids.shape[0]):
        distances[:, i] = np.linalg.norm(data.values - centroids[i], axis=1)

    return np.argmin(distances, axis=1)


def update_centroids(data, clusters, k):
    centroids = np.zeros((k, data.shape[1]))

    for i in range(k):
        centroids[i] = data.values[clusters == i].mean(axis=0)

    return centroids


def kmeans_algorithm(data, k, max_iterations):
    centroids = initialize_centroids(data, k)

    for i in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):  # Check for convergence
            break
        centroids = new_centroids

    return clusters, centroids


def plot_clusters(data, clusters, centroids):
    plt.figure(figsize=(10, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, s=50, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
    plt.title('K-means Clustering on Gene Expression Data')
    plt.xlabel('Gene 1 Expression')
    plt.ylabel('Gene 2 Expression')
    plt.legend()
    plt.grid()
    plt.show()


num_samples = 1000
num_clusters = 4
max_iterations = 100
data, cell_types = cell_differentiation_data(num_samples=num_samples)
plot_data(data)
standardized_data = standardize_data(data)
clusters, centroids = kmeans_algorithm(standardized_data, num_clusters, max_iterations)
plot_clusters(standardized_data, clusters, centroids)
