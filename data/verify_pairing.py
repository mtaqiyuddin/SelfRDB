#!/usr/bin/env python3
"""
Simple script to verify T1-T2 slice pairing after dataset creation.
"""

import numpy as np
import os
from pathlib import Path

def verify_pairing(dataset_dir):
    """Verify that T1 and T2 slices are properly paired."""
    
    dataset_path = Path(dataset_dir)
    
    for split in ["train", "val", "test"]:
        t1_dir = dataset_path / "t1" / split
        t2_dir = dataset_path / "t2" / split
        
        if not t1_dir.exists() or not t2_dir.exists():
            print(f"[SKIP] {split} directories don't exist yet")
            continue
            
        t1_files = sorted([f for f in t1_dir.glob("*.npy")])
        t2_files = sorted([f for f in t2_dir.glob("*.npy")])
        
        print(f"\n[{split.upper()}] Found {len(t1_files)} T1 files and {len(t2_files)} T2 files")
        
        if len(t1_files) != len(t2_files):
            print(f"[ERROR] Mismatch: {len(t1_files)} T1 vs {len(t2_files)} T2 files")
            continue
            
        # Check first few pairs
        num_to_check = min(5, len(t1_files))
        print(f"Checking first {num_to_check} pairs:")
        
        for i in range(num_to_check):
            t1_file = t1_files[i]
            t2_file = t2_files[i]
            
            # Load and check shapes
            t1_data = np.load(t1_file)
            t2_data = np.load(t2_file)
            
            print(f"  {t1_file.name} ↔ {t2_file.name}")
            print(f"    T1 shape: {t1_data.shape}, range: [{t1_data.min():.3f}, {t1_data.max():.3f}]")
            print(f"    T2 shape: {t2_data.shape}, range: [{t2_data.min():.3f}, {t2_data.max():.3f}]")
            
            if t1_data.shape != t2_data.shape:
                print(f"    [ERROR] Shape mismatch!")
            else:
                print(f"    [OK] Shapes match")

if __name__ == "__main__":
    dataset_dir = "../datasets/IXI_processed"
    verify_pairing(dataset_dir)
