import json
import numpy as np
from pathlib import Path
from skimage.exposure import adjust_gamma

import click
from PIL import Image

from stardist.src.utils.config import read_json_config

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

    for source in source_files:
        image = np.array(Image.open(source))
        g_adjusted_image = adjust_gamma(image, gamma=config['conversion']['gamma'])
        Image.fromarray(g_adjusted_image).save(target_dir / (source.stem + '.' + 
        config['conversion']['target_type']))

    print("Gamma adjusted images written")

if __name__ == "__main__":
    main()