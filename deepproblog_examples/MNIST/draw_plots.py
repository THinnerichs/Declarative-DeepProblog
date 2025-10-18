import re
import matplotlib.pyplot as plt

# Use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
})

# File paths
vae_log_file = 'log/vae_losses.log'
digits_log_file = 'log/vanilla_losses.log'

# Function to parse log files
def parse_log(file_path):
    iterations = []
    losses = []
    pattern = re.compile(r"Iteration:\s+(\d+)\s+.*Loss:\s+([\d.]+)")
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Epoch"):
                continue
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                if iteration <= 10000:
                    loss = float(match.group(2))
                    iterations.append(iteration)
                    losses.append(loss)
    return iterations, losses

# Parse both logs
vae_iterations, vae_losses = parse_log(vae_log_file)
digits_iterations, digits_losses = parse_log(digits_log_file)

# Plot the data
plt.figure(figsize=(4.5, 2.5))
plt.plot(vae_iterations, vae_losses, linewidth=1.0, label=r"Declarative DPL")
plt.plot(digits_iterations, digits_losses, linewidth=1.0, label=r"DPL")
plt.title(r'\textbf{VAE Training Loss}', fontsize=10)
plt.xlabel(r'\textit{Iteration}')
plt.ylabel(r'\textit{Loss}')
plt.ylim(bottom=0)
plt.grid(True, linewidth=0.3)
plt.legend()
plt.tight_layout()

# Save as PDF file for publication
plt.savefig("figures/vae_loss_plot.pdf")
plt.close()
