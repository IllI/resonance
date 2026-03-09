import subprocess

# We can query openneuro CLI if it's installed or we can just google it again specifically
import openneuro

try:
    # Get datasets
    datasets = openneuro.download(dataset="ds000000") # Dummy usage
except Exception as e:
    pass
