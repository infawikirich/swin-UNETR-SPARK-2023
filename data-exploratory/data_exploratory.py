#importing libraries

import pydicom
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import json
from ipywidgets import interact, interactive
from typing import List, Tuple, Union

print("Libraries imported successfully")

# set the path to the data
dir_path = '1010_brain_mr_04_lee'

# read the dicom files from a directory
def load_dicom_slices(dir_path: str, force = False):
  """
    Load and sort a series of dicom files inside the provided folder
    path.
  """
  files = os.listdir(dir_path)
  slices = []
  for file in files:
    if file.endswith('.dcm'):
      ds = pydicom.dcmread(os.path.join(dir_path, file))
      slices.append(ds)

    slices.sort(key = lambda x: int(x.InstanceNumber))
    return slices

slices = load_dicom_slices(dir_path)
print("Number of slices: ", len(slices))
print("Slices dtype: ", type(slices[0]))

# viewing slices shape
print("Volume Shape (Row, Column)", slices[0].Rows, slices[0].Rows)

# interactive slides for viewing dicom slides

from ipywidgets import interact

plt.figure(1, figsize = (10,10))
def dicom_animation(x):
  plt.imshow(slices[x].pixel_array, cmap = plt.cm.bone)
  plt.colorbar()
  plt.show()
  return x


interact(dicom_animation, x = (0, len(slices)-1))

# accessing the patient id.
# accessing the patient ID using indexing:
patient_id = slices[0].PatientID
print("Patient ID:", patient_id)

# Accessing the patient ID using attributes (tags):
patient_id = slices[0].get("PatientID")
print("Patient ID", patient_id)


#tranform the slices into Housefield scale
def to_hu(slices: List):
  """
    Transform a list of slices to a Housefield Unit Scale.
    This function takes the loaded slices and return a list of 
    transformed

  """

  hu_slices = []
  intercept = slices[0].RescaleIntercept if "RescaleIntercept" in slices[0] else 0
  slope = slices[0].RescaleSlope if "RescaleSlope" in slices[0] else 1

  for sli in slices:
    hu_image = sli.pixel_array * slope + intercept
    hu_slices.append(hu_image)
  
  return hu_slices


def window_clip(slices: List, window_cent: int, window_width: int):
  """
    Clip a list of slices pixels, one by one, into a specific intensity
    range basedon the provided window location size.
    All the pixels inside each single slice with a intensity below and over
     the window range will be clipped into the main and max intensity range
     window covers.
     This function returns a list clipped Numpy array slices
  """

  cliped_slices = []
  for sli in slices:
    cliped = np.clip(sli, window_cent - (window_width/2), window_cent +(window_width/2))
    cliped_slices.append(cliped)
  return cliped_slices


def to_3d_numpy(slices: List, dtype = None):
  """
    Stack up all slices into a single NumPy array of the provided
    data type
  """
  image = np.stack(slices)
  if dtype:
    image = image.astype(dtype)
  return image


def min_max_scaler(image: np.array, dtype: Union[type, None] = None):
  """
    Scale a single Numpy array image intensity into range `0` and `1`
  """
  min_val = np.min(image)
  max_val = np.max(image)
  image = (image - min_val)/(max_val - min_val)

  if dtype:
    image = image.astype(dtype)

  return image

def visualizer(slice: np.ndarray, title=""):
  """Visualize a slice of type numpy array with the provided title"""
  plt.imshow(slice, cmap = plt.cm.bone)
  plt.title(title)
  plt.show()


processed_slices_hu = to_hu(slices)
processed_slices_cliped = window_clip(processed_slices_hu, 500, 1000)
image = to_3d_numpy(processed_slices_cliped, dtype = None)

print("Numpy array image shape is: ", image.shape)
print("Transformed image pixel value range (min, max): ", (image.min(), image.max()))

scaled_image = min_max_scaler(image, dtype = np.float32)
print("Scaled image pixel value range (min, max): ", (scaled_image.min(), scaled_image.max()))

# You can see the scaled version of your selected region using the `window_clip`
# function here
visualizer(processed_slices_hu[0], "HU transform")
visualizer(scaled_image[0], "Scaled")


# Convert the original slices into a 3d nmpy array and save it as a .npy file format. 
def to_3d_numpy(slices, output_file):
    
    # Sort the DICOM files by InstanceNumber
    slices.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

    # Read the first DICOM file to get the shape of the pixel data
    first_dicom = pydicom.dcmread(slices[0])
    rows = int(first_dicom.Rows)
    cols = int(first_dicom.Columns)
    slices = len(slices)

    # Create a 3D Numpy array to store the pixel data
    pixel_data = np.zeros((slices, rows, cols), dtype=np.int16)

    # Read the pixel data from each DICOM file and store it in the Numpy array
    for i, file in enumerate(slices):
        ds = pydicom.dcmread(file)
        pixel_data[i, :, :] = ds.pixel_array

    # Save the pixel data as a .npy file
    np.save(output_file, pixel_data)

    return pixel_data


# Create a folder and save the scaled slices one by one. 
# create the output directory if it doesn't exist
if not os.path.exists('scaled_slices'):
    os.makedirs('scaled_slices')

# loop through the DICOM datasets and save them as individual slices
for i, dataset in enumerate(scaled_image):
    ds = pydicom.dataset.Dataset({'PixelData': dataset.tobytes()})
    filename = f'slice_{i}.dcm'
    filepath = os.path.join('scaled_slices', filename)
    np.save(filepath, ds)


# create the output directory if it doesn't exist
if not os.path.exists('scaled_slices_dicom'):
    os.makedirs('scaled_slices_dicom')

# read the DICOM series into a list of slices
slices = []
for filename in os.listdir(dir_path):
    filepath = os.path.join(dir_path, filename)
    ds = pydicom.dcmread(filepath)
    slices.append(ds.pixel_array)

# convert the list of slices to a numpy array and scale the pixel values
image = np.array(slices)
scaled_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 65535
scaled_image = scaled_image.astype(np.uint16)

# loop through the slices and save them as individual DICOM files
for i, slice_data in enumerate(scaled_image):
    ds = pydicom.dataset.Dataset({'PixelData': slice_data.tobytes()})
    filename = f'slice_{i}.dcm'
    filepath = os.path.join('scaled_slices_dicom', filename)
    np.save(filepath, ds)


