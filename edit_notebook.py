import json
from pathlib import Path
nb_path = Path('data_analysis.ipynb')
nb = json.loads(nb_path.read_text())
print('cells', len(nb.get('cells', [])))
