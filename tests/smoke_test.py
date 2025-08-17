# Quick smoke test to ensure data download & minimal training loop run.
import subprocess, sys, os, json, pathlib

repo = pathlib.Path(__file__).resolve().parents[1]
os.chdir(repo)
print("Running minimal train...")
subprocess.check_call([sys.executable, "-m", "src.train", "--epochs", "1", "--batch-size", "8", "--features", "mfcc"])
print("OK!")
