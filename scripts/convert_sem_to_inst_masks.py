import json
from pathlib import Path
from skimage.measure import label
from typing import Union

import click
import numpy as np
from PIL import Image

from stardist.src.utils.config import read_json_config

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)

    current_dir	= config['project_result_dir']
    if current_dir:
        current_dir = Path(current_dir)
    else:
        current_dir = Path(__file__).parent

    print(current_dir)

    source_files = list(Path(config['conversion']['source_dir']) \
                    .glob('*.'+config['conversion']['source_type']))
    target_dir = Path(config['conversion']['target_dir'])

    print(len(source_files))
    for source in source_files:
        image = Image.open(source)
        img_array = np.array(image)


        inst_mask = label(img_array)
        print(np.unique(inst_mask))

        inst_mask = Image.fromarray(np.uint16(inst_mask))
        inst_mask.save(target_dir / (source.stem.split("_")[0] + '.' + config['conversion']['target_type']))

if __name__ == "__main__":
    main()
