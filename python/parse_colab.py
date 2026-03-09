import json
import sys

def parse_notebook(filename):
    print(f"\n--- {filename} ---")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') in ['markdown', 'code']:
                src = "".join(cell.get('source', []))
                # Only print interesting parts if it's long
                if len(src.strip()) > 0:
                    print(f"[{cell['cell_type'].upper()}]\n{src[:1000]}\n...")
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

parse_notebook('notebook1.ipynb')
parse_notebook('notebook2.ipynb')
