import re
import matplotlib.pyplot as plt

# Path tới file log
log_path = r"H:\ICPR2026\LP-Diff\LP-Diff-kaggle\experiments\LP-Diff_251224_042735\logs\train.log"   # đ

iters = []
l_pix = []

# Regex để bắt iter và l_pix
pattern = re.compile(r"iter:\s*([\d,]+)> l_pix:\s*([\deE\.-]+)")

with open(log_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            it = int(match.group(1).replace(",", ""))
            loss = float(match.group(2))
            iters.append(it)
            l_pix.append(loss)

print(f"Parsed {len(iters)} points")

# Vẽ
plt.figure(figsize=(8, 4))
plt.plot(iters, l_pix)
plt.xlabel("Iteration")
plt.ylabel("l_pix")
plt.title("Training l_pix over iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
