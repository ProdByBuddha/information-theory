"""
This module provides basic information theory concepts, including entropy,
mutual information, and channel capacity calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def entropy(p):
    """
    Calculate the Shannon entropy of a probability distribution.
    
    Args:
        p: probability distribution (numpy array or list)
    
    Returns:
        entropy value in bits
    
    Real-World Context:
        Entropy is a measure of uncertainty or randomness. In data compression,
        it represents the minimum number of bits needed to encode information.
        High entropy indicates more randomness, which means more bits are needed
        to represent the data without loss. In cybersecurity, entropy is used to
        assess the strength of encryption keys.
    """
    # Filter out zero probabilities to avoid log(0)
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def mutual_information(joint_prob, marginal_x, marginal_y):
    """
    Calculate mutual information between two random variables.
    
    Args:
        joint_prob: joint probability matrix P(X,Y)
        marginal_x: marginal probability P(X)
        marginal_y: marginal probability P(Y)
    
    Returns:
        mutual information I(X;Y) in bits
    
    Real-World Context:
        Mutual information quantifies the amount of information obtained about
        one random variable through another. In machine learning, it is used
        for feature selection, where features that provide the most information
        about the target variable are chosen. In telecommunications, it helps
        in optimizing channel capacity by understanding dependencies between
        transmitted and received signals.
    """
    # Calculate entropies
    h_x = entropy(marginal_x)
    h_y = entropy(marginal_y)
    h_xy = entropy(joint_prob.flatten())
    
    # Mutual information is I(X;Y) = H(X) + H(Y) - H(X,Y)
    return h_x + h_y - h_xy

def binary_channel_capacity(p_error):
    """
    Calculate the capacity of a binary symmetric channel with error probability p.
    
    Args:
        p_error: probability of bit flip
    
    Returns:
        channel capacity in bits per channel use
    
    Real-World Context:
        Channel capacity is a fundamental concept in communication systems,
        representing the maximum rate at which information can be reliably
        transmitted over a channel. In wireless communications, understanding
        channel capacity helps in designing systems that maximize data throughput
        while minimizing error rates.
    """
    if p_error <= 0 or p_error >= 1:
        return 0
    
    # BSC capacity is 1 - H(p)
    return 1 - entropy([p_error, 1-p_error])

# Example: Plot entropy of a binary distribution
def plot_binary_entropy():
    """
    Plot the entropy of a binary distribution as a function of probability.
    """
    p_values = np.linspace(0.01, 0.99, 100)
    h_values = [entropy([p, 1-p]) for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, h_values)
    plt.title('Entropy of Binary Distribution')
    plt.xlabel('Probability p')
    plt.ylabel('Entropy H(p) [bits]')
    plt.grid(True)
    plt.savefig('binary_entropy.png')
    
# Example: Channel capacity vs. noise
def plot_bsc_capacity():
    """
    Plot the capacity of a binary symmetric channel as a function of error probability.
    """
    p_values = np.linspace(0.01, 0.99, 100)
    capacity_values = [binary_channel_capacity(p) for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, capacity_values)
    plt.title('Binary Symmetric Channel Capacity')
    plt.xlabel('Error Probability p')
    plt.ylabel('Capacity [bits/channel use]')
    plt.grid(True)
    plt.savefig('bsc_capacity.png')

if __name__ == "__main__":
    # Example 1: Calculate entropy of various distributions
    # Real-World Context: Understanding the entropy of a distribution helps in
    # data compression and encryption. For instance, a uniform distribution like
    # a fair dice roll has maximum entropy, indicating maximum uncertainty.
    uniform_dist = [1/6] * 6  # Dice roll
    print(f"Entropy of uniform distribution (dice): {entropy(uniform_dist):.4f} bits")
    
    # Real-World Context: A biased distribution, such as a loaded dice, has lower
    # entropy, indicating less uncertainty and more predictability.
    biased_dist = [0.5, 0.25, 0.125, 0.125]  # Non-uniform distribution
    print(f"Entropy of biased distribution: {entropy(biased_dist):.4f} bits")
    
    # Example 2: Binary symmetric channel
    # Real-World Context: In digital communications, understanding the capacity
    # of a channel with noise (error probability) is crucial for efficient data
    # transmission.
    ERROR_PROB = 0.1
    capacity = binary_channel_capacity(ERROR_PROB)
    print(f"BSC capacity with p_error={ERROR_PROB}: {capacity:.4f} bits/use")
    
    # Example 3: Create visualizations
    # Real-World Context: Visualizing entropy and channel capacity helps in
    # understanding how probability and noise affect information transmission.
    plot_binary_entropy()
    plot_bsc_capacity()
    
    print("Visualizations saved as binary_entropy.png and bsc_capacity.png")
    
