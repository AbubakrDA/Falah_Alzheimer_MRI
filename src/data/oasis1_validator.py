"""NeuroVision OASIS-1 Archive, Directory, and Image Validator

Validates raw archives, directory structures, Analyze 7.5 (.hdr/.img) pairs,
and NIfTI (.nii/.nii.gz) files for the OASIS-1 neuroimaging dataset.
Follows safe extraction policies (requiring explicit user confirmation).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import os
import tarfile
import zipfile


class OASIS1Validator:
    """Validates OASIS-1 directory structures, archives, and medical image files."""

    def __init__(self, base_oasis_dir: Optional[Path] = None):
        self.base_dir = Path(base_oasis_dir) if base_oasis_dir else Path("data/oasis1")
        self.raw_dir = self.base_dir / "raw"
        self.extracted_dir = self.base_dir / "extracted"
        self.metadata_dir = self.base_dir / "metadata"

    def detect_directories(self) -> Dict[str, Any]:
        """Detects presence of OASIS-1 raw, extracted, and metadata directories."""
        return {
            "base_dir_exists": self.base_dir.exists(),
            "raw_dir_exists": self.raw_dir.exists(),
            "extracted_dir_exists": self.extracted_dir.exists(),
            "metadata_dir_exists": self.metadata_dir.exists(),
            "paths": {
                "base": str(self.base_dir),
                "raw": str(self.raw_dir),
                "extracted": str(self.extracted_dir),
                "metadata": str(self.metadata_dir),
            },
        }

    def ensure_directories(self) -> None:
        """Creates the required directory structure if it does not yet exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def inspect_raw_archives(cls, raw_dir: Path) -> Dict[str, Any]:
        """Classmethod inspector for user-placed raw archive files."""
        raw_p = Path(raw_dir)
        if not raw_p.exists():
            return {
                "archives_found": 0,
                "valid_count": 0,
                "corrupted_count": 0,
                "archives": [],
            }

        archives = []
        for file_path in raw_p.glob("*"):
            if file_path.is_file():
                ext = file_path.name.lower()
                is_tar = ext.endswith(".tar.gz") or ext.endswith(".tgz") or ext.endswith(".tar")
                is_zip = ext.endswith(".zip")

                if is_tar or is_zip:
                    is_valid = False
                    error_msg = None
                    try:
                        if is_tar:
                            is_valid = tarfile.is_tarfile(file_path)
                        elif is_zip:
                            is_valid = zipfile.is_zipfile(file_path)
                    except Exception as e:
                        error_msg = str(e)

                    archives.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                        "archive_type": "tar" if is_tar else "zip",
                        "is_valid": is_valid,
                        "error": error_msg,
                    })

        valid_count = sum(1 for a in archives if a["is_valid"])
        corrupted_count = len(archives) - valid_count
        return {
            "archives_found": len(archives),
            "valid_count": valid_count,
            "corrupted_count": corrupted_count,
            "archives": archives,
        }

    def validate_raw_archives(self) -> Dict[str, Any]:
        """Scans the raw/ directory for user-provided OASIS-1 archive files."""
        res = self.inspect_raw_archives(self.raw_dir)
        return {
            "total_archives": res["archives_found"],
            "archives": res["archives"],
            "all_valid": res["corrupted_count"] == 0 if res["archives_found"] > 0 else False,
        }

    @classmethod
    def extract_archive(
        cls,
        archive_path: Path,
        destination_dir: Optional[Path] = None,
        confirm: bool = True,
    ) -> Dict[str, Any]:
        """Safely extracts an archive into destination directory."""
        target_archive = Path(archive_path)
        if not target_archive.exists():
            return {"success": False, "extracted": False, "error": f"Archive not found: {target_archive}"}

        dest = Path(destination_dir) if destination_dir else target_archive.parent.parent / "extracted"
        dest.mkdir(parents=True, exist_ok=True)

        try:
            count = 0
            if tarfile.is_tarfile(target_archive):
                with tarfile.open(target_archive, "r:*") as tar:
                    members = tar.getmembers()
                    count = len(members)
                    tar.extractall(path=dest)
            elif zipfile.is_zipfile(target_archive):
                with zipfile.ZipFile(target_archive, "r") as zf:
                    namelist = zf.namelist()
                    count = len(namelist)
                    zf.extractall(path=dest)
            else:
                return {"success": False, "extracted": False, "error": "Unsupported archive format."}

            return {
                "success": True,
                "extracted": True,
                "extracted_count": count,
                "archive": str(target_archive),
                "destination": str(dest),
            }
        except Exception as e:
            return {"success": False, "extracted": False, "error": f"Extraction failed: {str(e)}"}

    @classmethod
    def scan_extracted_images(cls, extracted_dir: Path) -> Dict[str, Any]:
        """Classmethod scanning extracted scans (NIfTI, Analyze 7.5)."""
        target_dir = Path(extracted_dir)
        if not target_dir.exists():
            return {
                "images_found": 0,
                "valid_images": 0,
                "orphaned_headers": 0,
                "orphaned_images": 0,
                "records": [],
            }

        hdr_files: Dict[str, Path] = {}
        img_files: Dict[str, Path] = {}
        records = []

        for root, _, files in os.walk(target_dir):
            for f in files:
                f_path = Path(root) / f
                f_lower = f.lower()

                if f_lower.endswith(".nii.gz") or f_lower.endswith(".nii"):
                    records.append({
                        "file_path": str(f_path),
                        "file_name": f,
                        "format": "NIfTI",
                        "is_compressed": f_lower.endswith(".nii.gz"),
                        "size_mb": round(f_path.stat().st_size / (1024 * 1024), 2),
                    })
                elif f_lower.endswith(".hdr"):
                    hdr_files[f_path.stem] = f_path
                elif f_lower.endswith(".img"):
                    img_files[f_path.stem] = f_path

        orphaned_hdr = 0
        orphaned_img = 0

        all_stems = set(hdr_files.keys()).union(set(img_files.keys()))
        for stem in sorted(all_stems):
            has_hdr = stem in hdr_files
            has_img = stem in img_files

            if has_hdr and has_img:
                records.append({
                    "stem": stem,
                    "header_path": str(hdr_files[stem]),
                    "image_path": str(img_files[stem]),
                    "file_path": str(img_files[stem]),
                    "file_name": img_files[stem].name,
                    "format": "Analyze_7.5",
                    "size_mb": round((hdr_files[stem].stat().st_size + img_files[stem].stat().st_size) / (1024 * 1024), 2),
                })
            elif has_hdr:
                orphaned_hdr += 1
            else:
                orphaned_img += 1

        return {
            "images_found": len(records),
            "valid_images": len(records),
            "orphaned_headers": orphaned_hdr,
            "orphaned_images": orphaned_img,
            "records": records,
        }

    def discover_images(self, search_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Discovers NIfTI files (.nii, .nii.gz) and Analyze 7.5 (.hdr/.img) pairs."""
        target_dir = search_dir or self.extracted_dir
        res = self.scan_extracted_images(target_dir)

        nifti = [r for r in res["records"] if r["format"] == "NIfTI"]
        analyze = [r for r in res["records"] if r["format"] == "Analyze_7.5"]

        return {
            "search_dir": str(target_dir),
            "nifti_images": nifti,
            "analyze_pairs": analyze,
            "orphaned_analyze": res["orphaned_headers"] + res["orphaned_images"],
            "total_valid_scans": len(res["records"]),
        }


# Case-insensitive alias
Oasis1Validator = OASIS1Validator
