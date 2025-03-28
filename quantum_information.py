"""
This module demonstrates key concepts in quantum information theory,
including density matrices, von Neumann entropy, and quantum mutual information.
It also provides visualizations of quantum states under noise.
"""
import numpy as np
from numpy import linalg as LA
try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for plotting. Please install it using 'pip install matplotlib'."
    ) from exc

# Real-World Context: Density matrices are used in quantum computing to describe the state of a quantum system.
# They are crucial in quantum cryptography and quantum error correction.
# Example: In quantum key distribution (QKD) protocols like BB84, density matrices help ensure secure communication.

def density_matrix(state_vector):
    """
    Create a density matrix from a pure state vector.
    
    Args:
        state_vector: A normalized quantum state vector
    
    Returns:
        Density matrix ρ = |ψ⟩⟨ψ|
    """
    state = np.array(state_vector).reshape(-1, 1)  # Ensure column vector
    return np.dot(state, state.conj().T)

# Real-World Context: Von Neumann entropy is a measure of quantum uncertainty and is used in quantum thermodynamics and information theory.
# Example: It helps determine the efficiency of quantum engines or the security of quantum communication channels.

def von_neumann_entropy(rho):
    """
    Calculate the von Neumann entropy of a density matrix.
    S(ρ) = -Tr(ρ log₂ ρ) = -∑ λᵢ log₂ λᵢ
    
    Args:
        rho: Density matrix
    
    Returns:
        Von Neumann entropy in bits
    """
    eigenvalues = LA.eigvalsh(rho)
    # Filter out very small eigenvalues (numerical stability)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace(rho, system_dims, traced_systems):
    """
    Perform partial trace over specified subsystems.
    
    Args:
        rho: Density matrix of the composite system
        system_dims: List of dimensions for each subsystem
        traced_systems: List of subsystem indices to trace out
    
    Returns:
        Reduced density matrix
    """
    n_systems = len(system_dims)
    traced_systems = sorted(traced_systems)

    # Calculate dimensions
    d = np.prod(system_dims)
    
    # Reshape density matrix to tensor form
    tensor_dims = system_dims + system_dims
    rho_tensor = rho.reshape(tensor_dims)
    
    # Perform partial trace
    for i, system in enumerate(traced_systems):
        # Adjust index for already traced systems
        system -= i
        
        # Contract the corresponding indices
        contracted_indices = list(range(n_systems))
        contracted_indices.pop(system)
        
        # Add shifted indices for bra part
        for index in contracted_indices:
            contracted_indices.append(index + n_systems - i - 1)
        
        # Trace the current subsystem
        rho_tensor = np.trace(rho_tensor, axis1=system, axis2=system + n_systems - i - 1)
    
    # Calculate new shape
    remaining_systems = [d for i, d in enumerate(system_dims) if i not in traced_systems]
    new_d = np.prod(remaining_systems)
    
    # Reshape to matrix form
    return rho_tensor.reshape((new_d, new_d))

# Real-World Context: Quantum mutual information measures the total correlations between parts of a quantum system.
# It is used in quantum teleportation and entanglement swapping.
# Example: Calculating quantum mutual information in quantum networks optimizes data transfer.

def quantum_mutual_information(rho_ab, dims):
    """
    Calculate quantum mutual information I(A:B) for a bipartite system.
    I(A:B) = S(ρA) + S(ρB) - S(ρAB)
    
    Args:
        rho_ab: Density matrix of the composite system
        dims: List [dA, dB] with dimensions of subsystems A and B
    
    Returns:
        Quantum mutual information in bits
    """
    # Reduced density matrices
    rho_a = partial_trace(rho_ab, dims, [1])
    rho_b = partial_trace(rho_ab, dims, [0])
    
    # Calculate entropies
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)
    
    return s_a + s_b - s_ab

# Real-World Context: Depolarizing channels model noise in quantum systems, critical for developing robust quantum algorithms.
# Example: They simulate noise in quantum circuits, impacting quantum algorithm performance.

def depolarizing_channel(rho, p):
    """
    Apply a depolarizing channel to a quantum state.
    E(ρ) = (1-p)ρ + p/d I
    
    Args:
        rho: Input density matrix
        p: Depolarizing probability
    
    Returns:
        Output density matrix after the channel
    """
    d = rho.shape[0]
    identity = np.eye(d) / d
    return (1 - p) * rho + p * identity

# Real-World Context: Quantum channel capacity determines the maximum rate of reliable quantum information transmission.
# Important for quantum internet development.
# Example: Channel capacity bounds help design quantum communication protocols that maximize data throughput.

def quantum_channel_capacity_bounds(p, d=2):
    """
    Calculate bounds on the quantum capacity of a depolarizing channel.
    
    Args:
        p: Depolarizing probability
        d: Dimension of the Hilbert space
    
    Returns:
        (lower_bound, upper_bound) for the quantum capacity
    """
    # Coherent information maximized for the maximally mixed input
    # Lower bound from hashing bound
    if p >= 1 - 1/d:  # Above this, the channel is entanglement-breaking
        lower_bound = 0
    else:
        # Simplified lower bound for depolarizing channel
        lower_bound = max(0, np.log2(d) - 2 * binary_entropy((d-1)*p/d))
    
    # Upper bound from no-cloning bound
    upper_bound = max(0, np.log2(d) - binary_entropy(p) - p * np.log2(d-1))
    
    return lower_bound, upper_bound

def binary_entropy(p):
    """
    Calculate the binary entropy function H(p) = -p log₂(p) - (1-p) log₂(1-p)
    """
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def demonstrate_bell_state():
    """
    Demonstrate quantum information properties of a Bell state
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    """
    # Create the Bell state vector
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    # Density matrix of the Bell state
    rho = density_matrix(psi)
    
    # Dimensions of the subsystems
    dims = [2, 2]  # Two qubits
    
    # Calculate reduced density matrices
    rho_a = partial_trace(rho, dims, [1])
    rho_b = partial_trace(rho, dims, [0])
    
    print("Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print("\nDensity matrix of the Bell state:")
    print(np.round(rho, 3))
    
    print("\nReduced density matrix of qubit A:")
    print(np.round(rho_a, 3))
    
    print("\nReduced density matrix of qubit B:")
    print(np.round(rho_b, 3))
    
    # Calculate entropies
    s_ab = von_neumann_entropy(rho)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    
    print(f"\nVon Neumann entropy of the Bell state: {s_ab:.4f} bits")
    print(f"Von Neumann entropy of qubit A: {s_a:.4f} bits")
    print(f"Von Neumann entropy of qubit B: {s_b:.4f} bits")
    
    # Calculate quantum mutual information
    mi = quantum_mutual_information(rho, dims)
    print(f"\nQuantum mutual information I(A:B): {mi:.4f} bits")
    
    # Entanglement under noise
    print("\nEntanglement under depolarizing noise:")
    noise_levels = np.linspace(0, 1, 11)
    entropies = []
    
    for noise in noise_levels:
        # Apply depolarizing channel to the Bell state
        rho_noisy = depolarizing_channel(rho, noise)
        s = von_neumann_entropy(rho_noisy)
        entropies.append(s)
        print(f"p = {noise:.1f}, S(ρ) = {s:.4f} bits")
    
    # Plot entropy vs noise
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, entropies, 'o-')
    plt.xlabel('Depolarizing Probability p')
    plt.ylabel('Von Neumann Entropy (bits)')
    plt.title('Entropy of Bell State Under Depolarizing Noise')
    plt.grid(True)
    plt.savefig('quantum_entropy_vs_noise.png')
    
    # Plot quantum capacity bounds for depolarizing channel
    noise_range = np.linspace(0, 1, 100)
    lower_bounds = []
    upper_bounds = []
    
    for p in noise_range:
        lower, upper = quantum_channel_capacity_bounds(p)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_range, lower_bounds, 'b-', label='Lower Bound')
    plt.plot(noise_range, upper_bounds, 'r-', label='Upper Bound')
    plt.xlabel('Depolarizing Probability p')
    plt.ylabel('Quantum Capacity (qubits)')
    plt.title('Bounds on Quantum Channel Capacity')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_capacity_bounds.png')

if __name__ == "__main__":
    print("Quantum Information Theory Demonstration\n")
    demonstrate_bell_state()
    print("\nVisualizations saved as quantum_entropy_vs_noise.png and quantum_capacity_bounds.png")
    
