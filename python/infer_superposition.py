
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlinoss_fmri import DLinOSSLayer, DLinOSSConfig, SuperpositionEncoder

def simulate_learning_and_inference():
    print("="*60)
    print("SUPERPOSITION INFERENCE SIMULATION")
    print("="*60)
    print("Goal: Infer the 'Task Superposition' (W) from Input Series (U) and Brain States (Z)")
    print("      at the moment of 'Grokking' (100% accuracy / Phase Lock).")
    
    # 1. Setup Environment
    n_features = 100   # Complexity of the Task (Input Stimuli features)
    n_neurons = 20     # Brain Capacity (Latent D-LinOSS dims) - Bottleck!
    n_time = 200       # Duration of learning session
    
    print(f"\n[1] Environment: Task Complexity N={n_features} >> Brain Capacity M={n_neurons}")
    print("    Hypothesis: The brain MUST use Superposition to learn this task.")
    
    # Ground Truth "Task Logic" (The hidden rule we want to learn)
    # The task maps inputs to a specific meaningful output via a sparse rule
    # But here we focus on the REPRESENTATION of the task in the brain.
    # We assume the "Task" implies a specific stable projection of features.
    true_task_W = np.random.randn(n_features, n_neurons) 
    # Normalize
    true_task_W /= np.linalg.norm(true_task_W, axis=0)
    
    # 2. Generate Input Series (Stimuli) - "The Experimental Stimuli"
    # Random stream of task problems
    input_series = np.random.randn(n_time, n_features) 
    
    # 3. Simulate D-LinOSS "Learning"
    # We simulate the brain state evolving. 
    # Initially: Chaos (Random projection)
    # Learning: Weights align with true_task_W
    # Grokking: Phase transition to stable Superposition
    
    brain_states = np.zeros((n_time, n_neurons))
    grokking_metric = np.zeros(n_time)
    
    # Simulated alignment curve (Sigmoid learning curve)
    t_grok = 120 # The moment of insight
    
    print(f"\n[2] Simulating Learning Process...")
    for t in range(n_time):
        # Alignment factor (0 = random, 1 = perfect superposition)
        # Sigmoid function centered at t_grok
        alignment = 1.0 / (1.0 + np.exp(-(t - t_grok) / 10.0))
        
        # Current effective projection (Brain's understanding)
        # Mixture of random noise and True Task Superposition
        noise_W = np.random.randn(n_features, n_neurons)
        current_W = (1 - alignment) * noise_W + alignment * true_task_W
        
        # Brain State Z(t) response to Input U(t)
        # Z(t) = W(t)^T @ U(t) + ProcessingNoise
        brain_states[t] = current_W.T @ input_series[t] + 0.01 * np.random.randn(n_neurons)
        
        # Calculate Grokking Metric (Geometric Coherence Proxy)
        # In this simulation, it tracks alignment. 
        # In reality, we measure this via D-LinOSS GCS.
        grokking_metric[t] = alignment

    print(f"    Grokking Event simulated at t={t_grok}")

    # 4. The "Inference" Step
    # Can we recover the Superposition Matrix W from just Inputs and States?
    # We look for the "Grokking Window" where GCS is high/stable.
    
    # Detect Grokking Window (e.g., GCS > 0.9)
    grok_indices = np.where(grokking_metric > 0.9)[0]
    
    if len(grok_indices) == 0:
        print("    [!] No grokking detected.")
        return
        
    print(f"\n[3] INFERRING SUPERPOSITION")
    print(f"    Detected Grokking Window: t={grok_indices[0]} to {grok_indices[-1]}")
    print("    Using Input Series (U) and Brain Output (Z) in this window to solve for W.")
    
    # Regression: Z = U @ W  =>  W = pinv(U) @ Z
    U_grok = input_series[grok_indices]
    Z_grok = brain_states[grok_indices]
    
    # Least squares solution for W
    inferred_W, residuals, rank, s = np.linalg.lstsq(U_grok, Z_grok, rcond=None)
    
    # 5. Validation: Does this Inferred Superposition predict "Learning"?
    # Compare Inferred W with True Task W
    
    # Cosine similarity between subspaces?
    # Or correlation of projections?
    
    # Let's test on NEW inputs (Generalization)
    test_inputs = np.random.randn(50, n_features)
    true_response = test_inputs @ true_task_W # What the brain SHOULD do if it learned
    predicted_response = test_inputs @ inferred_W # What we PREDICT it will do
    
    # Correlation
    corrs = []
    for i in range(n_neurons):
        c = np.corrcoef(true_response[:, i], predicted_response[:, i])[0, 1]
        corrs.append(c)
    mean_accuracy = np.mean(corrs)
    
    print("\n" + "="*60)
    print("RESULTS: AGI IMPLICATION")
    print("="*60)
    print(f"Reconstructed Superposition Validity: {mean_accuracy*100:.2f}%")
    
    if mean_accuracy > 0.95:
        print("\nSUCCESS: We have systematically inferred the 'Learned Superposition'.")
        print("Algorithm:")
        print("1. Monitor Field Resonance (GCS/Stability).")
        print("2. Isolate Inputs (U) and States (Z) during High-Resonance window.")
        print("3. Solve Z = U*W for W (The Superposition Matrix).")
        print("4. This W represents the 'Grokking' of the specific task.")
        print("   It can now be used to predict brain states for ANY new stimuli.")
    else:
        print("FAILURE: Could not reconstruct superposition.")

if __name__ == "__main__":
    simulate_learning_and_inference()
