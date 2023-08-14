import time
import os
import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, apply_orientation

def generate_segmentation_nifti(segmentation_data, case_id, timepoint, output_dir):
    
    # Convert segmentation data to NumPy array with a compatible data type
    segmentation_np = segmentation_data.cpu().detach().numpy().astype(np.float32)
    
    # Create a NIfTI image object from the segmentation data
    segmentation_nifti = nib.Nifti1Image(segmentation_np, affine=np.eye(4))

    # Set the NIfTI header information
    affine = np.eye(4)
    affine[:3, 3] = [0, -239, 0]  # Setting up the correct offset
    segmentation_nifti.set_qform(affine, code=1)
    segmentation_nifti.set_sform(affine, code=1)
    segmentation_nifti.header.set_data_shape((240, 240, 155))
    segmentation_nifti.header.set_xyzt_units('mm', 'sec')

    # Create the output filename
    output_filename = f"BraTS-GLI-seg-{case_id}-{timepoint.zfill(3)}.nii.gz"
    output_filepath = os.path.join(output_dir, output_filename)

    # Save the NIfTI file
    nib.save(segmentation_nifti, output_filepath)

    return output_filepath
