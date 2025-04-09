import numpy as np
import matplotlib.pyplot as plt


def cavi_algorithm(data_points, num_data_point, num_cluster, prior_variance, max_iterations):
    cavi_distribution = np.zeros((num_data_point, num_cluster))
    cluster_means = np.random.randn(num_cluster)
    cluster_variances = np.ones(num_cluster) * prior_variance

    for iteration in range(max_iterations):
        for i in range(num_data_point):
            for j in range(num_cluster):
                gaussian_distribution_log = -0.5 * np.log(2 * np.pi * cluster_variances[j]) - (data_points[i] - cluster_means[j]) ** 2 / (2 * cluster_variances[j])
                cavi_distribution[i, j] = np.exp(gaussian_distribution_log)
            cavi_distribution[i] /= np.sum(cavi_distribution[i])
        for k in range(num_cluster):
            total_weight = np.sum(cavi_distribution[:, k])
            cluster_means[k] = np.sum(cavi_distribution[:, k] * data_points) / (total_weight + (1 / prior_variance))
            cluster_variances[k] = 1 / (total_weight + (1 / prior_variance))

        elbo = 0
        for l in range(num_data_point):
            for m in range(num_cluster):
                compute_likelihood = -0.5 * np.log(2 * np.pi * cluster_variances[m]) - (data_points[l] - cluster_means[m]) ** 2 / (2 * cluster_variances[m])
                elbo += cavi_distribution[l, m] * compute_likelihood
        kl_divergence = 0
        for n in range(num_data_point):
            kl_divergence -= np.sum(
                cavi_distribution[n] * np.log(cavi_distribution[n] + (10 ** (-10))))
        elbo += kl_divergence

        print(f"For {iteration + 1} iterations: ELBO = {elbo:.3f}")

    print(f"Final cluster means = {cluster_means}")
    print(f"Final cluster variances = {cluster_variances}")

    return cavi_distribution, cluster_means, cluster_variances, num_cluster


def plot_cavi(data_points, num_cluster, cluster_means, cluster_variances):
    plt.figure(figsize=(10, 10))
    plt.scatter(data_points, np.zeros_like(data_points), color='black', marker='o', label='Data points', alpha=0.5)
    data_range = np.linspace(np.min(data_points) - 3, np.max(data_points) + 3, 1000)

    for a in range(num_cluster):
        standard_deviation = np.sqrt(cluster_variances[a])
        gaussian_distribution_plot = (1 / (standard_deviation * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_range - cluster_means[a]) / standard_deviation) ** 2)
        plt.plot(data_range, gaussian_distribution_plot, label=f"Mixture component {a + 1} with Mean = {cluster_means[a]:.3f} and Variance = {cluster_variances[a]:.3f}")

    plt.title('Bayesian Mixture of Gaussians with CAVI algorithm')
    plt.xlabel('Data Points')
    plt.ylabel('Probability Density Function')
    plt.legend()
    plt.grid()
    plt.show()


data_points = np.random.randn(100)
num_data_point = len(data_points)
num_cluster = 5
prior_variance = 1.0
max_iterations = 100

cavi_distribution, cluster_means, cluster_variances, num_cluster = cavi_algorithm(data_points, num_data_point, num_cluster, prior_variance, max_iterations)
plot_cavi(data_points, num_cluster, cluster_means, cluster_variances)
