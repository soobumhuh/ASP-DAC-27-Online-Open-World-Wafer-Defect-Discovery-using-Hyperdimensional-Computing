"""Put the repo root on sys.path so `import mri_hdc` works when a script is run from anywhere."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
