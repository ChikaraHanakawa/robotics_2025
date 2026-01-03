import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Kmeans:
    def __init__(self, X_dim, Y_dim, seed, k, max_iter):
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.seed = seed
        self.k = k
        self.max_iter = max_iter
        self.history = []

    def random_point(self):
        np.random.seed(self.seed)
        self.point = np.random.randn(self.X_dim, self.Y_dim)
        indices = np.random.choice(self.X_dim, self.k, replace=False)
        self.centroids = self.point[indices].copy()

    def predict(self):
        for iteration in range(self.max_iter):
            distance = np.zeros((self.X_dim, self.k))
            for i in range(self.k):
                # 距離を計算
                distance[:, i] = np.linalg.norm(self.point - self.centroids[i], axis=1)

            labels = np.argmin(distance, axis=1)

            self.history.append({
                'centroids': self.centroids.copy(),
                'labels': labels.copy(),
                'iteration': iteration
            })

            new_centroids = self.centroids.copy()

            for j in range(self.k):
                # 距離から重心を計算
                cluster_points = self.point[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

    def plot(self, figsize=(10, 8), interval=800, point_size=50,
             centroids_size=200, alpha=0.6, show_legend=True,
             grid=True, cmap='rainbow'):
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, self.k))

        def update(frame):
            ax.clear()
            state = self.history[frame]

            for i in range(self.k):
                mask = state['labels'] == i
                ax.scatter(self.point[mask, 0], self.point[mask, 1],
                           c=[colors[i]], alpha=alpha, s=point_size,
                           label=f'Cluster {i}')
                ax.scatter(self.centroids[i, 0], self.centroids[i, 1],
                c=[colors[i]], marker='x', s=centroids_size, label=f'Centroid {i}')
            
            ax.set_title(f"K-means Clustering - Iteration {state['iteration']}", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            
            if show_legend:
                ax.legend(loc='best')
            
            if grid:
                ax.grid(True, alpha=0.3, linestyle='--')
        
        anim = FuncAnimation(fig, update, frames=len(self.history), 
                           interval=interval, repeat=True)
        plt.tight_layout()
        plt.show()
        return anim

def main():
    kmeans = Kmeans(X_dim=300, Y_dim=2, seed=42, k=3, max_iter=100)
    kmeans.random_point()
    kmeans.predict()
    kmeans.plot()

if __name__ == "__main__":
    main()