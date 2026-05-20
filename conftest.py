import sys
from pathlib import Path

# Make the repo root importable so `from src.xxx import ...` works in tests
# without installing the package.
sys.path.insert(0, str(Path(__file__).parent))
