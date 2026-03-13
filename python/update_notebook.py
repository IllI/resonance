import json
from pathlib import Path

notebook_path = Path("c:/Users/cityz/IllI/newer_all/Chronos_Resonance_Stream_Dataset_Colab.ipynb")
worker_path = Path("c:/Users/cityz/IllI/newer_all/python/updated_colab_cuda_worker.py")
output_path = Path("c:/Users/cityz/IllI/newer_all/Chronos_Resonance_Stream_Dataset_Colab_v2.ipynb")

# Read the new worker code
new_worker_code = worker_path.read_text(encoding="utf-8")
# We just need the gpu_worker function body
# Let's extract everything from "def gpu_worker(" downwards
worker_lines = new_worker_code.splitlines()
start_idx = -1
for i, line in enumerate(worker_lines):
    if line.startswith("def gpu_worker("):
        start_idx = i
        break

new_gpu_worker_lines = [line + "\n" for line in worker_lines[start_idx:]]

# Read notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "%%writefile workers.py\n" in cell["source"][0]:
        source = cell["source"]
        # Find start and end of gpu_worker inside the notebook source
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(source):
            if line.startswith("def gpu_worker("):
                start_idx = i
            elif start_idx != -1 and line.startswith("def receiver_worker("):
                end_idx = i
                break
        
        if start_idx != -1 and end_idx != -1:
            # Replace
            new_source = source[:start_idx] + new_gpu_worker_lines + ["\n"] + source[end_idx:]
            cell["source"] = new_source
        break

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {output_path}")
