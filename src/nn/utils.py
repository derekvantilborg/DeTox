import yaml
import re

def load_config(path: str) -> dict:
    """Load YAML config and convert scientific notation strings to floats."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    def convert_values(obj):
        if isinstance(obj, dict):
            return {k: convert_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_values(item) for item in obj]
        elif isinstance(obj, str):
            # Match scientific notation like 1e-3, 2.5e-4, etc.
            if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', obj):
                return float(obj)
        return obj
    
    return convert_values(cfg)

