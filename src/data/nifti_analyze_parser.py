import os
import struct
import gzip
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Mapping of NIfTI/Analyze datatype codes to numpy dtypes
DATATYPE_TO_NUMPY = {
    2: (np.uint8, 1),
    4: (np.int16, 2),
    8: (np.int32, 4),
    16: (np.float32, 4),
    64: (np.float64, 8),
    256: (np.int8, 1),
    512: (np.uint16, 2),
    768: (np.uint32, 4),
}

DATATYPE_NAMES = {
    2: "uint8",
    4: "int16",
    8: "int32",
    16: "float32",
    64: "float64",
    256: "int8",
    512: "uint16",
    768: "uint32",
}

class MedicalImageReader:
    """
    Self-contained parser for NIfTI-1 (.nii, .nii.gz) and Analyze 7.5 (.hdr/.img) formats.
    Can read headers, volume shapes, voxel spacings, and extract 2D axial/coronal/sagittal slices
    without external system dependencies. Delegates to nibabel if available.
    """
    
    @staticmethod
    def is_nibabel_available() -> bool:
        try:
            import nibabel as nib
            return True
        except ImportError:
            return False

    @classmethod
    def read_header(cls, file_path: Path) -> Dict[str, Any]:
        """Reads metadata from .nii, .nii.gz, or .hdr without reading full image payload."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if nibabel is available
        if cls.is_nibabel_available():
            try:
                import nibabel as nib
                img = nib.load(str(file_path))
                hdr = img.header
                shape = tuple(int(s) for s in img.shape)
                zooms = tuple(float(z) for z in hdr.get_zooms()[:len(shape)])
                dtype_str = str(img.get_data_dtype())
                return {
                    "shape": shape,
                    "zooms": zooms,
                    "dtype": dtype_str,
                    "format": "nifti" if file_path.suffix in [".nii", ".gz"] else "analyze_7_5",
                    "reader": "nibabel"
                }
            except Exception:
                pass # Fallback to native binary parser

        # Native binary header parser
        is_gz = file_path.name.endswith(".gz") or file_path.suffix == ".gz"
        open_func = gzip.open if is_gz else open

        with open_func(file_path, "rb") as f:
            raw_hdr = f.read(348)

        if len(raw_hdr) < 348:
            raise ValueError(f"Header too short ({len(raw_hdr)} bytes, expected 348) for {file_path.name}")

        # Determine endianness from sizeof_hdr (must be 348)
        sizeof_hdr_le = struct.unpack("<I", raw_hdr[0:4])[0]
        sizeof_hdr_be = struct.unpack(">I", raw_hdr[0:4])[0]

        if sizeof_hdr_le == 348:
            endian = "<"
        elif sizeof_hdr_be == 348:
            endian = ">"
        else:
            raise ValueError(f"Invalid sizeof_hdr ({sizeof_hdr_le}/{sizeof_hdr_be}). Not a valid Analyze/NIfTI header.")

        # Read dimensions: dim[0] = num_dims, dim[1..dim[0]] = sizes
        dim = struct.unpack(f"{endian}8h", raw_hdr[40:56])
        num_dims = dim[0]
        if not (1 <= num_dims <= 7):
            num_dims = 3
        shape = tuple(dim[1:num_dims+1])

        # Read datatype & bitpix
        datatype = struct.unpack(f"{endian}h", raw_hdr[70:72])[0]
        bitpix = struct.unpack(f"{endian}h", raw_hdr[72:74])[0]

        # Read pixdim (voxel spacing in mm)
        pixdim = struct.unpack(f"{endian}8f", raw_hdr[76:108])
        zooms = tuple(float(pixdim[i]) for i in range(1, num_dims+1))

        # Check vox_offset (for single-file NIfTI .nii)
        vox_offset = struct.unpack(f"{endian}f", raw_hdr[108:112])[0]

        # Magic identifier
        magic = raw_hdr[344:348].rstrip(b"\x00")
        is_nifti = magic in (b"n+1", b"ni1")

        dtype_info = DATATYPE_NAMES.get(datatype, f"unknown_{datatype}_{bitpix}bit")

        return {
            "shape": shape,
            "zooms": zooms,
            "datatype_code": datatype,
            "dtype": dtype_info,
            "bitpix": bitpix,
            "vox_offset": int(vox_offset) if is_nifti and vox_offset > 0 else 0,
            "magic": magic.decode("latin1", errors="ignore"),
            "is_nifti": is_nifti,
            "format": "nifti_gz" if is_gz else ("nifti" if is_nifti else "analyze_7_5"),
            "endian": endian,
            "reader": "native_binary"
        }

    @classmethod
    def load_volume(cls, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads the full 3D numpy array and header metadata."""
        file_path = Path(file_path)
        hdr_info = cls.read_header(file_path)
        shape = hdr_info["shape"]

        if cls.is_nibabel_available():
            try:
                import nibabel as nib
                img = nib.load(str(file_path))
                vol = img.get_fdata().astype(np.float32)
                return vol, hdr_info
            except Exception:
                pass

        # Native binary volume reader
        endian = hdr_info.get("endian", "<")
        datatype_code = hdr_info.get("datatype_code", 4)
        np_dtype, item_size = DATATYPE_TO_NUMPY.get(datatype_code, (np.int16, 2))

        # Determine binary data source
        if file_path.suffix == ".hdr":
            img_path = file_path.with_suffix(".img")
            if not img_path.exists():
                raise FileNotFoundError(f"Analyze 7.5 .img file missing for {file_path.name}")
            data_file = img_path
            offset = 0
            open_func = open
        elif file_path.name.endswith(".nii.gz") or file_path.suffix == ".gz":
            data_file = file_path
            offset = hdr_info.get("vox_offset", 352)
            open_func = gzip.open
        else: # .nii
            data_file = file_path
            offset = hdr_info.get("vox_offset", 352)
            open_func = open

        with open_func(data_file, "rb") as f:
            if offset > 0:
                f.seek(offset)
            expected_elements = int(np.prod(shape))
            raw_bytes = f.read(expected_elements * item_size)
            
        data = np.frombuffer(raw_bytes, dtype=np.dtype(np_dtype).newbyteorder(endian))
        vol = data.reshape(shape, order="F").astype(np.float32)
        return vol, hdr_info

    @classmethod
    def extract_slice(cls, volume: np.ndarray, plane: str = "axial", slice_idx: Optional[int] = None) -> np.ndarray:
        """
        Extracts a normalized 2D slice from a 3D MRI volume.
        plane: 'axial' (default, z-axis), 'coronal' (y-axis), or 'sagittal' (x-axis)
        """
        if volume.ndim < 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        if plane == "axial":
            max_idx = volume.shape[2]
            idx = max_idx // 2 if slice_idx is None else max(0, min(slice_idx, max_idx - 1))
            slc = volume[:, :, idx]
        elif plane == "coronal":
            max_idx = volume.shape[1]
            idx = max_idx // 2 if slice_idx is None else max(0, min(slice_idx, max_idx - 1))
            slc = volume[:, idx, :]
        elif plane == "sagittal":
            max_idx = volume.shape[0]
            idx = max_idx // 2 if slice_idx is None else max(0, min(slice_idx, max_idx - 1))
            slc = volume[idx, :, :]
        else:
            raise ValueError(f"Unknown anatomical plane: {plane}")

        # Rotate/orient to standard radiologic viewing (head up)
        slc = np.rot90(slc)
        return slc
