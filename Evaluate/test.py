############### predit ###################
import torch 
import torch.nn as nn
import numpy as np
import time
import nibabel as nib
import scipy.io as scio
from argparse import ArgumentParser
import os
from IR2Unet import *

##########################################

def Read_nii(path):
    """
    read local field map from nifti file
    """
    nibField = nib.load(path)
    Field = nibField.get_fdata() 
    aff = nibField.affine
    Field = np.array(Field)
    return Field, aff

def zero_padding(Field, factor):
    # Get the size of the input Field
    im_size = torch.tensor(Field.size())
    
    # Calculate the expanded size
    up_size = torch.ceil(im_size / factor).to(torch.int) * factor
    
    # Calculate the initial and end positions
    pos_init = torch.ceil((up_size - im_size) / 2).to(torch.int)
    pos_end = pos_init + im_size - 1
    
    # Create a zero-filled tensor of the expanded size
    dtype = Field.dtype
    tmp_field = torch.zeros(tuple(up_size), dtype=dtype)
    
    # Copy the original data to the appropriate position in the expanded tensor
    tmp_field[pos_init[0]:pos_end[0]+1, pos_init[1]:pos_end[1]+1, pos_init[2]:pos_end[2]+1] = Field
    
    result = torch.tensor(tmp_field,dtype=dtype)

    return result
    
def process_nii_file(file_path, output_dir, model, device):

    image, aff = nib.load(file_path).get_fdata(), nib.load(file_path).affine

    image = torch.from_numpy(image).float()
    print(image.size())
    image = zero_padding(image, 8)
    print(image.size())
    mask = image != 0
    
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image = image.float()
    print(image.size())

    # reconstruction
    image = image.to(device)
    time_start = time.time()
    latest_out,all_out = model(image)

    pred = latest_out
    #pred = all_out[t] #it is possible to choose one of the intermediate outputs,but it is recommended to choose the final output
    time_end = time.time()
    print('used time:')
    print(time_end - time_start)
    pred = torch.squeeze(pred, 0)
    pred = torch.squeeze(pred, 0)
    pred = pred.to('cpu')
    print(pred.size())
    pred = pred * mask
    pred = pred.numpy()
    nii_image = nib.Nifti1Image(pred, affine=np.eye(4))
    nib.save(nii_image, output_dir)

if __name__ == '__main__':
    with torch.no_grad():        
        ## load trained network 
        net = IR2Unet()
        net = nn.DataParallel(net)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load('/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/CC/QSM_Recon/CC_code/codes_for_cc/git_IR2QSM/PythonCodes/Evaluation/model_IR2Unet.pth'),False)
        net.to(device)
        net.eval()
       
        input_path = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/CC/QSM_Recon/CC_code/codes_for_cc/git_IR2QSM/PythonCodes/Evaluation/lfs1.nii'
        output_path = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/CC/QSM_Recon/CC_code/codes_for_cc/git_IR2QSM/PythonCodes/Evaluation/chi1.nii'
        process_nii_file(input_path, output_path, net, device)
        print('end2')
