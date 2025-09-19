import os
import glob
import random
import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
import yaml
from tqdm import tqdm
from scipy.ndimage import zoom
from nilearn.image import resample_to_img, resample_img
from nibabel.affines import apply_affine, voxel_sizes

# Suppress future warnings and other common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*resample.*')
warnings.filterwarnings('ignore', message='.*interpolation.*')
warnings.filterwarnings('ignore', message='.*force_resample.*')
warnings.filterwarnings('ignore', message='.*copy_header.*')
warnings.filterwarnings('ignore', message='.*Nilearn.*')

# Set environment variables to suppress warnings from underlying libraries
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['NILEARN_SKIP_GLOBAL_WARNINGS'] = '1'

# Suppress numpy warnings
np.seterr(all='ignore')

# Additional nilearn-specific warning suppression
try:
    import nilearn
    nilearn.settings.log_level = 'ERROR'
except:
    pass


# ---------- helpers ----------

def _collect_subjects(ixi_root, subject_dirs):
    """Collect one T1 and one T2 NIfTI per subject directory."""
    subject_files = []
    for subject in subject_dirs:
        t1_files = glob.glob(os.path.join(ixi_root, subject, "T1", "NIfTI", "*.nii*"))
        t2_files = glob.glob(os.path.join(ixi_root, subject, "T2", "NIfTI", "*.nii*"))
        if t1_files and t2_files:
            subject_files.append({
                "subject": subject,
                "T1": t1_files[0],
                "T2": t2_files[0]
            })
    return subject_files


def _to_ras(img):
    """Nibabel image → RAS canonical (reorders/flips axes only)."""
    return nib.as_closest_canonical(img)


def _resample_to_isotropic_centered(img, voxel_mm=1.0):
    """
    Resample 'img' to an orthogonal isotropic grid **while preserving world-space center**.
    Avoids off-center black middle slices.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        aff = img.affine
        shape = np.array(img.shape, dtype=float)
        cur_vox = np.array(voxel_sizes(aff))        # current spacing (mm)
        new_vox = np.array([voxel_mm, voxel_mm, voxel_mm], dtype=float)

        # Old center in voxel coords → world coords
        c_old_vox = (shape - 1) / 2.0
        c_old_world = apply_affine(aff, c_old_vox)

        # New shape that preserves FOV (rounded up)
        new_shape = np.ceil(shape * (cur_vox / new_vox)).astype(int)

        # New affine: diag(spacing) with translation so that centers match
        R = np.diag(np.append(new_vox, 1.0))
        c_new_vox = (new_shape - 1) / 2.0
        t = c_old_world - (R[:3, :3] @ c_new_vox)
        R[:3, 3] = t

        return resample_img(img, target_affine=R, target_shape=tuple(new_shape),
                            interpolation="continuous", copy_header=True, force_resample=True)


def _resample_to_reference(moving_img, reference_img, interpolation="continuous"):
    """Resample 'moving_img' into the voxel grid of 'reference_img'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resample_to_img(moving_img, reference_img, interpolation=interpolation, 
                              copy_header=True, force_resample=True)


def _save_slice(slice_data, out_path, rotate_clockwise=True, scale_to_unit=True, target_size=256):
    """Save a single 2D slice as .npy with optional rotate/normalize and resize to target_size."""
    if rotate_clockwise:
        slice_data = np.rot90(slice_data, -1)  # 90° CW
    
    # Resize to target_size x target_size
    current_h, current_w = slice_data.shape
    zoom_factors = (target_size / current_h, target_size / current_w)
    slice_data = zoom(slice_data, zoom_factors, order=1)  # Linear interpolation
    
    if scale_to_unit:
        vmin, vmax = float(slice_data.min()), float(slice_data.max())
        if vmax > vmin:
            slice_data = (slice_data - vmin) / (vmax - vmin)
        else:
            slice_data = slice_data - vmin
    np.save(str(out_path), slice_data.astype(np.float32, copy=False))


def _extract_slice(vol, plane='axial', idx=None):
    """Extract a single slice from 3D array by plane."""
    if plane == 'axial':      # z
        k = vol.shape[2] // 2 if idx is None else idx
        return vol[:, :, k]
    elif plane == 'coronal':  # y
        k = vol.shape[1] // 2 if idx is None else idx
        return vol[:, k, :]
    elif plane == 'sagittal': # x
        k = vol.shape[0] // 2 if idx is None else idx
        return vol[k, :, :]
    else:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")


def best_index_by_energy(vol, plane='axial', frac=0.5):
    """
    Pick a slice index with high 'energy' (sum of intensities) to avoid blank slices.
    'frac' = fraction along the axis (0..1); default 0.5 = near middle but chooses among top-5.
    """
    if plane == 'axial':
        scores = vol.sum(axis=(0, 1))
    elif plane == 'coronal':
        scores = vol.sum(axis=(0, 2))
    elif plane == 'sagittal':
        scores = vol.sum(axis=(1, 2))
    else:
        raise ValueError
    # pick an index near the requested fraction but within top-5 energetic slices
    n = len(scores)
    target = int(frac * (n - 1))
    top5 = np.argsort(scores)[-5:]
    return top5[np.argmin(np.abs(top5 - target))]


# ---------- main pipeline ----------

def create_ixi_dataset(
    ixi_root,
    output_root,
    slice_range=(27, 127),
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    rotate_clockwise=True,
    scale_to_unit=True,
    isotropic_mm=None  # set to 1.0 to enforce orthogonal isotropic grid; None to skip
):
    """
    Subject-split dataset with alignment:
      - RAS canonical
      - optional centered isotropic resample on T1
      - T2 resampled to T1 grid
      - train/val: many axial slices (z-range)
      - test: 3 orientations (axial, coronal, sagittal) with robust slice selection
    """

    # your 5 subjects
    subject_dirs = ["100_Guys", "101_Guys", "102_HH", "103_Guys", "104_HH", "105_HH", 
                    "106_Guys", "107_Guys", "108_Guys", "109_Guys", "110_Guys"]
    subjects = _collect_subjects(ixi_root, subject_dirs)
    if not subjects:
        raise RuntimeError("No subjects found. Check ixi_root and folder structure.")

    # subject-level split
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_subjects = subjects[:n_train]
    val_subjects   = subjects[n_train:n_train + n_val]
    test_subjects  = subjects[n_train + n_val:]

    print(f"[Split] Train={len(train_subjects)}  Val={len(val_subjects)}  Test={len(test_subjects)}")

    # Make directories
    for modality in ("T1", "T2"):
        for split in ("train", "val", "test"):
            Path(output_root, modality.lower(), split).mkdir(parents=True, exist_ok=True)

    # ------- Train/Val -------
    for split, subjs in (("train", train_subjects), ("val", val_subjects)):
        slice_idx = 0  # Sequential numbering across all subjects
        
        for subj in subjs:
            # Load and align both T1 and T2 for this subject
            t1_img = _to_ras(nib.load(subj["T1"]))
            t2_img = _to_ras(nib.load(subj["T2"]))

            # Optional: centered isotropic; then align T2 to T1
            if isotropic_mm is not None:
                t1_img = _resample_to_isotropic_centered(t1_img, voxel_mm=isotropic_mm)
            t2_img_aligned = _resample_to_reference(t2_img, t1_img, interpolation="continuous")

            # Get aligned volumes
            t1_vol = t1_img.get_fdata()
            t2_vol = t2_img_aligned.get_fdata()

            # Ensure both volumes have the same shape
            assert t1_vol.shape == t2_vol.shape, f"Shape mismatch: T1 {t1_vol.shape} vs T2 {t2_vol.shape}"

            zmax = t1_vol.shape[2]
            start = max(0, slice_range[0])
            end   = min(zmax, slice_range[1])

            # Process both modalities for the same slice range
            for i in tqdm(range(start, end), desc=f"{split}-{subj['subject']}", leave=False):
                # Extract the same slice from both modalities
                t1_sl = t1_vol[:, :, i]
                t2_sl = t2_vol[:, :, i]
                
                # Save T1 slice
                t1_name = f"slice_{slice_idx}.npy"
                _save_slice(t1_sl, Path(output_root, "t1", split) / t1_name,
                            rotate_clockwise=rotate_clockwise,
                            scale_to_unit=scale_to_unit,
                            target_size=256)
                
                # Save T2 slice with the same index
                t2_name = f"slice_{slice_idx}.npy"
                _save_slice(t2_sl, Path(output_root, "t2", split) / t2_name,
                            rotate_clockwise=rotate_clockwise,
                            scale_to_unit=scale_to_unit,
                            target_size=256)
                
                slice_idx += 1

    # ------- Test (3 orientations, robust index) -------
    test_slice_idx = 0  # Sequential numbering for test set
    for subj in test_subjects:
        # Load and align both T1 and T2 for this subject
        t1_img = _to_ras(nib.load(subj["T1"]))
        t2_img = _to_ras(nib.load(subj["T2"]))
        if isotropic_mm is not None:
            t1_img = _resample_to_isotropic_centered(t1_img, voxel_mm=isotropic_mm)
        t2_img_aligned = _resample_to_reference(t2_img, t1_img, interpolation="continuous")

        # Get aligned volumes
        t1_vol = t1_img.get_fdata()
        t2_vol = t2_img_aligned.get_fdata()
        
        # Ensure both volumes have the same shape
        assert t1_vol.shape == t2_vol.shape, f"Shape mismatch: T1 {t1_vol.shape} vs T2 {t2_vol.shape}"

        # Process each orientation
        for plane in ("axial", "coronal", "sagittal"):
            # Use the same slice index for both modalities
            idx = best_index_by_energy(t1_vol, plane=plane, frac=0.5)  # Use T1 for slice selection
            
            # Extract the same slice from both modalities
            t1_sl = _extract_slice(t1_vol, plane=plane, idx=idx)
            t2_sl = _extract_slice(t2_vol, plane=plane, idx=idx)

            # 🔹 Extra rotation: flip coronal/sagittal 180°
            if plane in ("coronal", "sagittal"):
                t1_sl = np.rot90(t1_sl, 2)   # 180° CCW
                t2_sl = np.rot90(t2_sl, 2)   # 180° CCW

            # Save T1 slice
            t1_name = f"slice_{test_slice_idx}.npy"
            _save_slice(t1_sl, Path(output_root, "t1", "test") / t1_name,
                        rotate_clockwise=rotate_clockwise,
                        scale_to_unit=scale_to_unit,
                        target_size=256)
            
            # Save T2 slice with the same index
            t2_name = f"slice_{test_slice_idx}.npy"
            _save_slice(t2_sl, Path(output_root, "t2", "test") / t2_name,
                        rotate_clockwise=rotate_clockwise,
                        scale_to_unit=scale_to_unit,
                        target_size=256)
            
            test_slice_idx += 1

    # Create subject_ids.yaml for evaluation
    _create_subject_ids_file(output_root, train_subjects, val_subjects, test_subjects, slice_range)


def _create_subject_ids_file(output_root, train_subjects, val_subjects, test_subjects, slice_range=(27, 127)):
    """Create subject_ids.yaml file for evaluation purposes - ONLY for test data."""
    
    # Create mapping of slice indices to subject IDs - ONLY for test data
    subject_ids = []
    
    # Test subjects (3 orientations per subject)
    for subj in test_subjects:
        for _ in range(3):  # 3 orientations per subject
            subject_ids.append(subj['subject'])
    
    # Save to file
    subject_ids_path = Path(output_root, "subject_ids.yaml")
    with open(subject_ids_path, 'w') as f:
        yaml.dump(subject_ids, f, default_flow_style=False)
    
    print(f"[INFO] Created subject_ids.yaml with {len(subject_ids)} entries (test data only)")
    print(f"[INFO] Test: {len(test_subjects)} subjects × 3 orientations each")


def main():
    # Comprehensive warning suppression context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        ixi_root = "../datasets/IXI"          # adjust to your path
        output_root = "../datasets/IXI_processed"
        slice_range = (27, 127)

        create_ixi_dataset(
            ixi_root=ixi_root,
            output_root=output_root,
            slice_range=slice_range,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
            rotate_clockwise=True,
            scale_to_unit=True,
            isotropic_mm=1.0   # set to None if you prefer to skip isotropic resample
        )


if __name__ == "__main__":
    main()