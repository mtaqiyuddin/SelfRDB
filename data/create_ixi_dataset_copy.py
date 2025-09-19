import os
import glob
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import nibabel as nib


def _collect_ixi_files(ixi_root, modalities=("PD", "T1", "T2")):
    """
    Recursively collect IXI NIfTI file paths for given modalities.

    Expected IXI folder structure (example):
    <ixi_root>/
      ├── ixi/
      │   ├── 100_Guys/           # or 100_HH, 100_IOP, etc.
      │   │   ├── PD/
      │   │   │   └── NIfTI/
      │   │   │       └── IXI101-Guys-0749-PD.nii.gz
      │   │   ├── T1/
      │   │   │   └── NIfTI/
      │   │   │       └── IXI101-Guys-0749-T1.nii.gz
      │   │   └── T2/
      │   │       └── NIfTI/
      │   │           └── IXI101-Guys-0749-T2.nii.gz

    Returns:
        dict(modality_upper -> list of file paths)
    """
    files_by_mod = {m.upper(): [] for m in modalities}
    for m in modalities:
        # Accept different capitalization for NIfTI
        print(m)
        patterns = [
            os.path.join(ixi_root, "**", m, "NIfTI", "*.nii*")
        ]
        print(patterns)
        for pat in patterns:
            files_by_mod[m.upper()].extend(glob.glob(pat, recursive=True))
        # De-duplicate and sort
        files_by_mod[m.upper()] = sorted(set(files_by_mod[m.upper()]))
    return files_by_mod


def create_ixi_dataset(
    ixi_root,
    output_root,
    slice_range=(27, 127),
    modalities=None,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    rotate_clockwise=True,
    scale_to_unit=True
):
    """
    Create a 2D-slice dataset from the IXI dataset using only NIfTI files under PD/T1/T2.

    Output format:
    <output_root>/
      ├── pd/
      │   ├── train/
      │   ├── val/
      │   └── test/
      ├── t1/
      └── t2/
    """
    if modalities is None:
        modalities = ("PD", "T1", "T2")

    rng = random.Random(seed)
    files_by_mod = _collect_ixi_files(ixi_root, modalities=modalities)
    print(files_by_mod)

    # Split and process per modality
    for modality in modalities:
        mod_upper = modality.upper()
        mod_lower = modality.lower()

        files = files_by_mod.get(mod_upper, [])
        if not files:
            print(f"[WARN] No files found for modality '{mod_upper}' under {ixi_root}")
            continue

        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        # Create directories
        for split in ("train", "val", "test"):
            Path(output_root, mod_lower, split).mkdir(parents=True, exist_ok=True)

        # Process slices
        for split, file_list in split_map.items():
            out_dir = Path(output_root, mod_lower, split)
            slice_idx = 0
            for fpath in tqdm(file_list, desc=f"IXI {mod_upper}->{split}"):
                try:
                    img = nib.load(fpath).get_fdata()
                except Exception as e:
                    print(f"[ERROR] Failed to load {fpath}: {e}")
                    continue

                # Determine valid slice range
                zmax = img.shape[2] if img.ndim == 3 else (img.shape[3] if img.ndim == 4 else None)
                if zmax is None:
                    print(f"[WARN] Unexpected dims for {fpath}: {img.shape}, skipping.")
                    continue
                start, end = slice_range
                start = max(0, start)
                end = min(zmax, end)

                for i in tqdm(range(start, end), desc="Slices", leave=False):
                    slice_data = img[:, :, i] if img.ndim == 3 else img[:, :, i, 0]

                    if rotate_clockwise:
                        slice_data = np.rot90(slice_data, -1)

                    if scale_to_unit:
                        vmin = slice_data.min()
                        vmax = slice_data.max()
                        if vmax > vmin:
                            slice_data = (slice_data - vmin) / (vmax - vmin)
                        else:
                            slice_data = slice_data - vmin

                    np.save(str(out_dir / f"slice_{slice_idx}.npy"), slice_data.astype(np.float32, copy=False))
                    slice_idx += 1


def main():
    # Example usage
    ixi_root = "../datasets/IXI"   # adjust to your IXI root path
    output_root = "../datasets/IXI_processed"
    slice_range = (27, 127)
    modalities = ["PD", "T1", "T2"]

    create_ixi_dataset(
        ixi_root=ixi_root,
        output_root=output_root,
        slice_range=slice_range,
        modalities=modalities,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        rotate_clockwise=True,
        scale_to_unit=True
    )


if __name__ == "__main__":
    main()