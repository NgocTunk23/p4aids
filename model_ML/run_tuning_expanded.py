"""
model_ML/run_tuning_expanded.py
================================
Wrapper tuong thich: Chuyen tiep truc tiep sang Pipeline duy nhat:
  model_ML/pipeline_ml.py
"""
import sys
from pathlib import Path

# Them thu muc vao path neu can
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from model_ML.pipeline_ml import main

if __name__ == "__main__":
    main()
