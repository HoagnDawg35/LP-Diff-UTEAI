import numpy as np
import matplotlib.pyplot as plt

# Data (x-axis in thousands)
iters = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
psnr = np.array([6.3980, 6.4207, 6.4311, 6.3797, 6.4325, 6.4348,
                 6.4574, 6.4264, 6.4361, 6.4340])
loss = np.array([1.4114, 1.2962, 1.1840, 1.6794, 1.1998, 1.1691,
                 1.0377, 1.2358, 1.1876, 1.2222])

# Linear regression
psnr_fit = np.polyval(np.polyfit(iters, psnr, 1), iters)
loss_fit = np.polyval(np.polyfit(iters, loss, 1), iters)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---- Left: PSNR ----
axes[0].plot(iters, psnr, marker='o', label="PSNR", color="green")
axes[0].plot(iters, psnr_fit, linestyle='--', label="Linear fit", color="orange")
axes[0].set_title("Validation PSNR")
axes[0].set_xlabel("Iteration (x1k)")
axes[0].set_ylabel("PSNR (dB)")
axes[0].grid(True)
axes[0].legend()

# ---- Right: Loss ----
axes[1].plot(iters, loss, marker='o', label="Loss")
axes[1].plot(iters, loss_fit, linestyle='--', label="Linear fit")
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Iteration (x1k)")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
