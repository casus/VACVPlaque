import json
import numpy as np
from pathlib import Path
from typing import Union

import click
from PIL import Image

def read_json_config(config_file_path: Union[str, Path]) -> dict:
    with open(config_file_path, "r", encoding="UTF-8") as config_file:
        config = json.load(config_file)
    return config

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

    source_files2 = Path(config['conversion']['source_dir2']) \
                    .glob('*.'+config['conversion']['source_type'])

    source_dir1 = Path(config['conversion']['source_dir1'])
    target_dir2 = Path(config['conversion']['target_dir2'])
    target_dir1 = Path(config['conversion']['target_dir1'])

    # set seed
    np.random.seed(int(config['conversion']['seed']))

    for source in source_files2:
        image2 = Image.open(source)
        image1 = Image.open(source_dir1 / (source.stem + '.' + config['conversion']['source_type']))

        image2 = np.array(image2)
        image1 = np.array(image1)

        #set random numbers
        # This decides how many wells to hide
        # k = np.random.randint(low=0, high =3, size=1)

        # This decides which wells to hide
        # k_ls = list(np.random.choice(np.arange(1,7), k, replace=False))
        k_ls = [4,5,6]
        # find unique wells to mask, since we are using instance masks we know each unique value
        # in the mask corresponds to a well and making these as background solves our task

        # Change chosen well pixels all to background, the corresponding plaque pixels are also set to background
        image1[np.isin(image2, k_ls)] = 0
        image2[np.isin(image2, k_ls)] = 0
        
        #Save the new images
        image2 = Image.fromarray(image2)
        image1 = Image.fromarray(image1)
        print(image1.size)
        image2.save(target_dir2 / (source.stem + '.' + config['conversion']['target_type']))
        image1.save(target_dir1 / (source.stem + '.' + config['conversion']['target_type']))

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
