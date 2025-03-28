"""
This module implements the Information Bottleneck algorithm,
which is a method for finding a compressed representation of data
while preserving relevant information about the target variable.
"""
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Real-World Context: The Information Bottleneck algorithm is used in machine learning for feature selection and dimensionality reduction.
# It preserves relevant information while compressing data.
# Example: Applied in neural networks to improve generalization by reducing overfitting.

# Real-World Context: KL divergence is used in various fields like finance for risk assessment and in machine learning for model evaluation.
# Example: Used to compare probability distributions in real-world datasets, such as in anomaly detection.

# Real-World Context: Synthetic data is often used in privacy-preserving data analysis and testing machine learning models.
# Example: Important when real data is scarce or sensitive, and the information bottleneck helps in these scenarios.

# Real-World Context: Visualizations help in understanding complex data relationships and are crucial in data-driven decision-making.
# Example: Interpretations of plots can be used to make decisions in fields like marketing or healthcare.

def kl_divergence(p, q):
    """
    Calculate KL divergence between two distributions
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sum(p * np.log2(p / q + 1e-10))

def calculate_mutual_information(joint_dist):
    """
    Calculate mutual information from joint distribution
    """
    marginal_x = np.sum(joint_dist, axis=1)
    marginal_y = np.sum(joint_dist, axis=0)
    
    mi = 0.0
    for i in range(joint_dist.shape[0]):
        for j in range(joint_dist.shape[1]):
            if joint_dist[i,j] > 0:
                mi += joint_dist[i,j] * np.log2(joint_dist[i,j] / (marginal_x[i] * marginal_y[j] + 1e-10))
    return mi

class InformationBottleneck:
    """
    Information Bottleneck algorithm implementation
    
    X - Input variable (features)
    Y - Output variable (target)
    T - Compressed representation of X
    
    The method finds a mapping p(t|x) that maximizes I(T;Y) while minimizing I(X;T),
    effectively compressing X while preserving information about Y.
    """
    def __init__(self, n_clusters=10, beta=1.0, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters  # Number of compressed representations
        self.beta = beta  # Trade-off parameter
        self.max_iter = max_iter
        self.tol = tol
        self.p_t_given_x = None
        self.p_t = None
        self.p_y_given_t = None
        self.convergence = []
        self.n_iterations = 0
        self.i_x_t = 0
        self.i_t_y = 0
        
    def fit(self, p_x_y):
        """
        Fit the information bottleneck model
        
        Args:
            p_x_y: Joint probability distribution p(x,y)
        """
        n_x, n_y = p_x_y.shape
        
        # Initialize p(t) uniformly
        p_t = np.ones(self.n_clusters) / self.n_clusters
        
        # Initialize p(t|x) randomly and normalize
        p_t_given_x = np.random.rand(self.n_clusters, n_x)
        p_t_given_x = p_t_given_x / np.sum(p_t_given_x, axis=0)
        
        # Calculate p(x)
        p_x = np.sum(p_x_y, axis=1)
        
        # Calculate p(y)
        p_y = np.sum(p_x_y, axis=0)
        
        # IB iterations
        self.convergence = []
        for iteration in range(self.max_iter):
            # Calculate p(t,y)
            p_t_y = np.zeros((self.n_clusters, n_y))
            for t in range(self.n_clusters):
                for x in range(n_x):
                    p_t_y[t, :] += p_t_given_x[t, x] * p_x_y[x, :]
            
            # Calculate p(y|t)
            p_y_given_t = p_t_y / (p_t[:, np.newaxis] + 1e-10)
            
            # Update p(t|x) using the IB self-consistent equations
            old_p_t_given_x = p_t_given_x.copy()
            z_x = np.zeros(n_x)
            
            for x in range(n_x):
                for t in range(self.n_clusters):
                    d_kl = 0
                    for y in range(n_y):
                        if p_x_y[x, y] > 0 and p_y_given_t[t, y] > 0:
                            term = p_x_y[x, y] / p_x[x] * np.log2(p_y_given_t[t, y] / (p_y[y] + 1e-10))
                            d_kl += term
                    
                    p_t_given_x[t, x] = p_t[t] * np.exp(self.beta * d_kl)
                
                z_x[x] = np.sum(p_t_given_x[:, x])
                p_t_given_x[:, x] /= (z_x[x] + 1e-10)
            
            # Update p(t)
            for t in range(self.n_clusters):
                p_t[t] = np.sum(p_t_given_x[t, :] * p_x)
            
            # Calculate change for convergence check
            change = np.mean(np.abs(p_t_given_x - old_p_t_given_x))
            self.convergence.append(change)
            
            if change < self.tol:
                break
        
        self.p_t_given_x = p_t_given_x
        self.p_t = p_t
        self.p_y_given_t = p_y_given_t
        self.n_iterations = iteration + 1
        
        
        # Calculate mutual information I(X;T) and I(T;Y)
        
        for x in range(n_x):
            for t in range(self.n_clusters):
                if p_t_given_x[t, x] > 0:
                    self.i_x_t += p_x[x] * p_t_given_x[t, x] * np.log2(p_t_given_x[t, x] / p_t[t])
        
        
        for t in range(self.n_clusters):
            for y in range(n_y):
                if p_t_y[t, y] > 0:
                    self.i_t_y += p_t_y[t, y] * np.log2(p_y_given_t[t, y] / p_y[y])
        
        return self
        
    def compress(self, x):
        """
        Compress data using the learned mapping
        """
        if x >= self.p_t_given_x.shape[1]:
            raise ValueError("Input x is outside the range of learned distribution")
        return self.p_t_given_x[:, x]


# Demo the Information Bottleneck on synthetic data
def demo_information_bottleneck():
    """
    Demonstrate the Information Bottleneck algorithm on synthetic data.
    Generates synthetic data, applies the algorithm with various beta values,
    and visualizes the results.
    """
    # Generate synthetic data with 5 clusters
    n_samples = 500
    n_features = 10  # X dimension in our joint distribution
    n_classes = 5    # Y dimension in our joint distribution
    
    x, y = make_blobs(n_samples=n_samples, centers=n_classes, 
                      n_features=2, random_state=42)
    
    # Discretize X into bins to create finite alphabet
    n_bins = n_features
    x_discrete = np.zeros((n_samples, 2), dtype=int)
    for i in range(2):  # For each feature dimension
        bins = np.linspace(x[:, i].min(), x[:, i].max(), n_bins+1)
        x_discrete[:, i] = np.digitize(x[:, i], bins[:-1])
    
    # Create joint distribution p(x,y)
    p_x_y = np.zeros((n_bins, n_classes))
    for i in range(n_samples):
        x_idx = x_discrete[i, 0]  # Using just the first feature for simplicity
        y_idx = y[i]
        # Ensure indices are within bounds
        x_idx = min(x_idx, n_bins - 1)
        y_idx = min(y_idx, n_classes - 1)
        p_x_y[x_idx, y_idx] += 1
    
    # Normalize to get probabilities
    p_x_y /= np.sum(p_x_y)
    
    # Apply Information Bottleneck with different beta values
    beta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    i_x_t_values = []
    i_t_y_values = []
    
    for beta in beta_values:
        ib = InformationBottleneck(n_clusters=7, beta=beta, max_iter=100)
        ib.fit(p_x_y)
        i_x_t_values.append(ib.i_x_t)
        i_t_y_values.append(ib.i_t_y)
        print(f"Beta={beta}: I(X;T)={ib.i_x_t:.4f}, I(T;Y)={ib.i_t_y:.4f}, Iterations={ib.n_iterations}")
    
    # Plot the information plane (I(X;T) vs I(T;Y))
    plt.figure(figsize=(10, 6))
    plt.scatter(i_x_t_values, i_t_y_values, s=100)
    
    # Add beta values as labels
    for i, beta in enumerate(beta_values):
        plt.annotate(f"β={beta}", (i_x_t_values[i], i_t_y_values[i]), 
                    xytext=(10, 5), textcoords='offset points')
    
    plt.title('Information Bottleneck: Information Plane')
    plt.xlabel('I(X;T) - Complexity')
    plt.ylabel('I(T;Y) - Relevance')
    plt.grid(True)
    plt.savefig('information_bottleneck_plane.png')
    
    # Plot convergence for the last beta value
    plt.figure(figsize=(10, 6))
    plt.plot(ib.convergence)
    plt.title(f'Convergence with β={beta}')
    plt.xlabel('Iteration')
    plt.ylabel('Change in p(t|x)')
    plt.grid(True)
    plt.savefig('information_bottleneck_convergence.png')
    
    print("Information Bottleneck visualizations saved.")

if __name__ == "__main__":
    print("Demonstrating Information Bottleneck method...")
    demo_information_bottleneck() 
    
