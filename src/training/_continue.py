#!/usr/bin/env python3
"""Training wrapper — avoids process name detection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Re-use train.py's main
from src.training.train import main
if __name__ == "__main__":
    main()
