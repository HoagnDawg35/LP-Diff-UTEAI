import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Parse the data
data = [
    {'time': '25-12-31 22:06:10.254', 'epoch': 41, 'iter': 20000, 'psnr': 11.500, 'loss': 0.29982},
    {'time': '26-01-01 08:19:58.130', 'epoch': 82, 'iter': 40000, 'psnr': 11.288, 'loss': 0.32177},
    {'time': '26-01-01 16:16:59.436', 'epoch': 123, 'iter': 60000, 'psnr': 11.478, 'loss': 0.30047},
    {'time': '26-01-02 01:49:28.231', 'epoch': 164, 'iter': 80000, 'psnr': 11.535, 'loss': 0.30024},
    {'time': '26-01-02 12:16:59.803', 'epoch': 204, 'iter': 100000, 'psnr': 10.586, 'loss': 0.36003},
    {}
]

# Extract data for plotting
iterations = [d['iter'] for d in data]
psnr_values = [d['psnr'] for d in data]
loss_values = [d['loss'] for d in data]
epochs = [d['epoch'] for d in data]

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot PSNR
ax1.plot(iterations, psnr_values, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('PSNR (dB)', fontsize=12, color='b')
ax1.set_title('PSNR vs Iterations', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='b')

# Add epoch annotations on top
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks(iterations)
ax1_top.set_xticklabels([f'Epoch {e}' for e in epochs], rotation=45, ha='left')
ax1_top.set_xlabel('Epoch Progression', fontsize=12)

# Add value annotations
for i, (iter_val, psnr_val) in enumerate(zip(iterations, psnr_values)):
    ax1.annotate(f'{psnr_val:.3f}', 
                xy=(iter_val, psnr_val), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='b')

# Plot Loss
ax2.plot(iterations, loss_values, 'r-s', linewidth=2, markersize=8)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12, color='r')
ax2.set_title('Loss vs Iterations', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='r')

# Add epoch annotations on top for loss plot
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks(iterations)
ax2_top.set_xticklabels([f'Epoch {e}' for e in epochs], rotation=45, ha='left')
ax2_top.set_xlabel('Epoch Progression', fontsize=12)

# Add value annotations for loss
for i, (iter_val, loss_val) in enumerate(zip(iterations, loss_values)):
    ax2.annotate(f'{loss_val:.5f}', 
                xy=(iter_val, loss_val), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center', 
                fontsize=9,
                color='r')

plt.tight_layout()
plt.show()

# Also create a combined plot for comparison
fig2, ax3 = plt.subplots(figsize=(12, 6))

# Plot both PSNR and Loss on same axes (with different y-axes)
color1 = 'tab:blue'
ax3.set_xlabel('Iterations', fontsize=12)
ax3.set_ylabel('PSNR (dB)', fontsize=12, color=color1)
line1 = ax3.plot(iterations, psnr_values, 'b-o', linewidth=2, markersize=8, label='PSNR')
ax3.tick_params(axis='y', labelcolor=color1)

# Create second y-axis for loss
ax4 = ax3.twinx()
color2 = 'tab:red'
ax4.set_ylabel('Loss', fontsize=12, color=color2)
line2 = ax4.plot(iterations, loss_values, 'r-s', linewidth=2, markersize=8, label='Loss')
ax4.tick_params(axis='y', labelcolor=color2)

# Add title and grid
ax3.set_title('PSNR and Loss vs Iterations', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add epoch labels on top
ax5 = ax3.twiny()
ax5.set_xlim(ax3.get_xlim())
ax5.set_xticks(iterations)
ax5.set_xticklabels([f'Epoch {e}' for e in epochs], rotation=45, ha='left')
ax5.set_xlabel('Epoch Progression', fontsize=10)

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left')

fig2.tight_layout()
plt.show()