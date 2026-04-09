import sys
from pathlib import Path

# Add examples/compositional_analysis/ so tests can import utils, train, obstacles.
# verifai itself must be installed: pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
