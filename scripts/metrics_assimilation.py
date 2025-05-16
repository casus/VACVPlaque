import numpy as np

from stardist.src.utils.hydranet import create_assimilated_dict

def main():
    l1 = []
    l2 = []
    l3 = []

    metric = 'mean_true_score' 
    #'accuracy' is the AP
    #'mean_true_score' is the IoU_R.

    asd = create_assimilated_dict(l1,l2,l3,metric)
    print(asd) 

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter