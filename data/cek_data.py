import os
import glob
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import nibabel as nib
files = '/datasets/IXI/100_Guys/T1/NIfTI/IXI100-Guys-0747-T1.nii.gz'

try:
    img = nib.load(files).get_fdata()
    print(img.shape)
except Exception as e:
    print(f"[ERROR] Failed to load {files}: {e}")
    