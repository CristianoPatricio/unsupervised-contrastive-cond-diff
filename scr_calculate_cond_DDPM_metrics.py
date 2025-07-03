import pickle
import numpy as np
import os
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--run_path', type=str,
                        help='path to the checkpoints')

    parser.add_argument('--folds', type=int, default=5,
                        help='number of folds')

    opt = parser.parse_args()

    return opt

def main(run_path, datasets, folds):

    for ds in datasets:

        AUPRCPerVolMeans = []
        AUPRCPerVolStds = []
        DicePerVolMeans = []
        DicePerVolStds = []
        l1recoErrorAllMeans = []
        l1recoErrorAllStds = []

        for fold in range(folds):

            # Load preds file
            with open(os.path.join(run_path, f'{fold+1}_preds_dict.pkl'),'rb') as f:
                unpickled_array = pickle.load(f)

                print(f"------------------------- Fold: {fold+1} -------------------------")

                if ds != 'IXI':
                    print(f"Results for {ds}:")
                    print(f"AUPRCPerVolMean: {unpickled_array['test'][f'Datamodules_eval.{ds}']['AUPRCPerVolMean']}")
                    AUPRCPerVolMeans.append(unpickled_array['test'][f'Datamodules_eval.{ds}']['AUPRCPerVolMean'])
                    print(f"AUPRCPerVolStd: {unpickled_array['test'][f'Datamodules_eval.{ds}']['AUPRCPerVolStd']}")
                    AUPRCPerVolStds.append(unpickled_array['test'][f'Datamodules_eval.{ds}']['AUPRCPerVolStd'])
                    print(f"DicePerVolMean: {unpickled_array['test'][f'Datamodules_eval.{ds}']['DicePerVolMean']}")
                    DicePerVolMeans.append(unpickled_array['test'][f'Datamodules_eval.{ds}']['DicePerVolMean'])
                    print(f"DicePerVolStd: {unpickled_array['test'][f'Datamodules_eval.{ds}']['DicePerVolStd']}")
                    DicePerVolStds.append(unpickled_array['test'][f'Datamodules_eval.{ds}']['DicePerVolStd'])
                    print("---")
                else:
                    print(f"Results for {ds}:")
                    print(f"l1recoErrorAllMean: {unpickled_array['test'][f'Datamodules_train.{ds}']['l1recoErrorAllMean']}")
                    l1recoErrorAllMeans.append(unpickled_array['test'][f'Datamodules_train.{ds}']['l1recoErrorAllMean'])
                    print(f"l1recoErrorAllStd: {unpickled_array['test'][f'Datamodules_train.{ds}']['l1recoErrorAllStd']}")
                    l1recoErrorAllStds.append(unpickled_array['test'][f'Datamodules_train.{ds}']['l1recoErrorAllStd'])
                    print("---")

        print(f"Averaged results across all folds:")
        if ds != 'IXI':
            print(f"############# Results for {ds}#############")
            print(f"AUPRCPerVolMean: {np.mean(np.array(AUPRCPerVolMeans))}")
            print(f"AUPRCPerVolStd: {np.std(np.array(AUPRCPerVolMeans))}")
            print(f"DicePerVolMean: {np.mean(np.array(DicePerVolMeans))}")
            print(f"DicePerVolStd: {np.std(np.array(DicePerVolMeans))}")
            print("###############################################")
        else:
            print(f"############# Results for {ds}#############")
            print(f"l1recoErrorAllMean: {np.mean(np.array(l1recoErrorAllMeans))}")
            print(f"l1recoErrorAllStd: {np.std(np.array(l1recoErrorAllMeans))}")
            print("###############################################")

if __name__ == '__main__':
    opt = parse_option()
    datasets = ['Brats21', 'IXI', 'MSLUB']
    folds = 5
    main(run_path=opt.run_path, datasets=datasets, folds=folds)