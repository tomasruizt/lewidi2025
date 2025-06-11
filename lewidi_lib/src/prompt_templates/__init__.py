from pathlib import Path


templates_root = Path(__file__).parent
all_templates = list(templates_root.glob("*.txt"))
