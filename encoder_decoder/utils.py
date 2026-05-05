import yaml
from pathlib import Path
from dataclasses import dataclass

out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)

@dataclass
class TextVariantConfig:
    positionFixed: bool = True
    textFixed: bool = True
    textLenFixed: bool = True

config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    raw = yaml.safe_load(f)

textVariant = TextVariantConfig(**raw["textVariant"])
cloud_fixed = raw['cloud']['fixed']
cloud_dynamic = raw['cloud']['dynamic']
model_name = raw['model_name']
