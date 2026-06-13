import numpy as np
import matplotlib.pyplot as plt

def plot_sinusoidal_signals():
    # 1. Setup time domain
    t = np.linspace(0, 2 * np.pi, 500)
    
    # 2. Define clean signal (A * sin(omega * t))
    clean_signal = np.sin(5 * t)
    
    # 3. Get user input for noise variance
    try:
        variance = float(input("Enter the variance for Gaussian noise (e.g., 0.1): "))
        std_dev = np.sqrt(variance)
    except ValueError:
        print("Invalid input. Using default variance of 0.1.")
        std_dev = np.sqrt(0.1)
        
    # 4. Generate Gaussian noise
    # Mean is 0, standard deviation is sqrt(variance)
    noise = np.random.normal(0, std_dev, size=t.shape)
    noisy_signal = clean_signal + noise
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, clean_signal, label='Clean Signal', color='blue', linewidth=2)
    plt.plot(t, noisy_signal, label=f'Noisy Signal (Var={variance})', 
             color='red', alpha=0.5, linestyle='--')
    
    plt.title('Sinusoidal Signal vs. Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_sinusoidal_signals()