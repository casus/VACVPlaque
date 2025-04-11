import json
import numpy as np
from pathlib import Path

import click
from PIL import Image

from stardist.src.utils.config import read_json_config

def zoom_at(img, zoom, n):
    w, h = img.size
    x, y = w//2, h//2
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w//(2**n), h//(2**n)), Image.NEAREST)

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    if config['project_result_dir']:
        current_dir = Path(config['project_result_dir'])
    else:
        current_dir = Path(__file__).parent
    print(current_dir)

    source_files = Path(config['conversion']['source_dir']) \
                    .glob('*.'+config['conversion']['source_type'])
    target_dir = Path(config['conversion']['target_dir'])
    zoom = config['conversion']['zoom']
    n = int(config['conversion']['n'])

    for source in source_files:
        image = Image.open(source)
        z_adj_image = zoom_at(image, zoom, n)
        z_adj_image.save(target_dir / (source.stem + '.' + config['conversion']['target_type']))

    print("Zoom adjusted resized image written")

if __name__ == "__main__":
    main()