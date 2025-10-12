import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attention_matrix, title='Attention'):
    plt.imshow(attention_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Memory Slots')
    plt.ylabel('Query')
    plt.show()

def plot_reservoir_states(reservoirs, title='Reservoir States'):
    for i, r in enumerate(reservoirs):
        plt.plot(r.detach().cpu().numpy(), label=f'Res {i}')
    plt.title(title)
    plt.legend()
    plt.show()

