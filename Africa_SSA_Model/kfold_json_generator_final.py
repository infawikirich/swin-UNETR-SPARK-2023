from glob import glob
import json
import random
import  nibabel as nib
import numpy as np


def view_data_shape(path_2_img:str):
    r''' Given path to a nifti file, this function will load the contents as numpy array and prints its shape. 
    '''
    img:np.array = nib.load(path_2_img).get_fdata()
    print(img.shape)
    
    return 0

def kfold_data_dict(data_dir:str, num_folds:int):
    r''' Given a directory of images from BraTS challege, it will map the patient files and stores them in a dictionary
    inputs:
        - data_dir: path to the training data
        - num_folds: number of k fold cross validation 

    output: 
        k_fold_dict: a dictionary holding the following keys:
            fold
            image
            label
            training
    '''
    patient_dir:list = glob(data_dir + "*/")
    random.shuffle(patient_dir)
    # print(patient_dir)

    # Initialize the dictionary holding k fold cross validation
    kfold_dict:dict = {"training":[]}
    # figure out how many patients are in a fold
    num_patient_per_fold:int = int(len(patient_dir) / num_folds)
    # print(num_patient_per_fold)

    #count = 0

    for k in range(num_folds):
        # print(type(k))
        
        for patient in patient_dir[k*num_patient_per_fold: (k+1)*num_patient_per_fold]:
            # print(patient)
            # print()
            try:

                temp_dict = {'fold':k}
                temp_dict["image"] = glob(patient+"/*t*")
                temp_dict["label"] = glob(patient+"/*seg*")[0]

                kfold_dict["training"].append(temp_dict)

            except:
                print(patient)
            #print(temp_dict)
            # break
        # break
            #count += 1

    #print(count)

    return kfold_dict

def main():

    # check the shape of a single image or segmentation file. 
    # for brats 2021 
    # path_2_img = "/scratch/guest183/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/BraTS-SSA-00002-000_seg.nii.gz"
    # for brats 2023 africa 
    # path_2_img = "/scratch/guest185/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00151-000/BraTS-GLI-00151-000-seg.nii.gz"
    # view_data_shape(path_2_img)

    # generate json file containing the path to n fold cross validation data 
    # get the number of folds
    num_folds:int = 5
    # get the path to the patient image folders
    data_dir:str = "/scratch/guest185/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/"
    kfold_dict:dict = kfold_data_dict(data_dir, num_folds)
    # print(len(kfold_dict["training"]))
    with open("./brats2023_ssa_data.json", 'w') as outfile:
        json.dump(kfold_dict, outfile, indent=4)

# DO NOT DELETE
if __name__ == "__main__":
    main()
