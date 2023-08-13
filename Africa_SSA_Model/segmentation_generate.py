import time
import os
import nibabel as nib
import numpy as np

def generate_segmentation_nifti(segmentation_data, case_id, timepoint, output_dir):
    
    # Convert segmentation data to NumPy array with a compatible data type
    segmentation_np = segmentation_data.cpu().numpy().astype(np.float32)
    
    # Create a NIfTI image object from the segmentation data
    segmentation_nifti = nib.Nifti1Image(segmentation_data, affine=np.eye(4))

    # Set the NIfTI header information
    segmentation_nifti.header.set_data_shape((240, 240, 155))
    segmentation_nifti.header.set_xyzt_units('mm', 'sec')
    segmentation_nifti.header.set_qform(np.diag([-1, 1, -1, 1]), code=1)
    segmentation_nifti.header.set_sform(np.diag([-1, 1, -1, 1]), code=1)
    segmentation_nifti.header['qoffset_x'] = 0
    segmentation_nifti.header['qoffset_y'] = -239
    segmentation_nifti.header['qoffset_z'] = 0

    # Create the output filename
    output_filename = f"BraTS-GLI-seg-{case_id}-{timepoint.zfill(3)}.nii.gz"
    output_filepath = os.path.join(output_dir, output_filename)

    # Save the NIfTI file
    nib.save(segmentation_nifti, output_filepath)
