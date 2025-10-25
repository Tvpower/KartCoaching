import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install",
                      "torch", "torchvision", "transformers",
                      "pillow", "scikit-learn"])

exec(open('training/training.py').read())