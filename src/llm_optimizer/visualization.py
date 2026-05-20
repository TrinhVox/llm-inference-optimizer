import json
import matplotlib.pyplot as plt

with open("../../results/benchmarks.json", "r") as f:
    data = json.load(f)["results"]

SEQUENCE_LENGTHS = [256, 512, 1024, 2048]
METHODS = ["fp16", "bnb_4bit", "gptq_4bit", "awq_4bit"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for method in METHODS:
    speeds = []
    memory = []
    for seq_len in SEQUENCE_LENGTHS:
        speeds.append(data[method][str(seq_len)]["tokens_per_sec"])
        memory.append(data[method][str(seq_len)]["peak_memory_mb"])
    ax1.plot(SEQUENCE_LENGTHS, speeds, marker="o", label=method)
    ax2.plot(SEQUENCE_LENGTHS, memory, marker="x", label=method)





ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Tokens/sec")
ax1.set_title("Throughput vs Sequence Length")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Memory(mb)")
ax2.set_title("Peak memory vs Sequence Length")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("../../results/profiling_plots.png", dpi=150)
print("Plots saved")