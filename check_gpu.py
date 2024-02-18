import torch
import subprocess

try:
    subprocess.run("nvcc --version", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed command: {e}")

print("Cuda is available: ",torch.cuda.is_available())
try:
    print(torch.zeros(1).cuda())
except Exception as e:
    print(f"Failed cuda : {e}")