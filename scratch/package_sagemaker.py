import os
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def main():
    print("==================================================================")
    print(" PACKAGING OPTIMIZED OCR_CHALL.ZIP FOR AWS SAGEMAKER ")
    print("==================================================================")

    zip_filename = os.path.join(PROJECT_ROOT, "OCR-chall.zip")

    # Target folders and files to include
    include_files = [
        "Test.csv",
        "SampleSubmission.csv",
        "CRNN.ipynb",
    ]

    include_dirs = [
        "data",
        "src",
        "Starters"
    ]

    # Explicit exclusions
    exclude_dirs = {"images", "scratch", "models", ".git", "__pycache__"}

    file_count = 0
    total_size = 0

    print(f"Creating Zip Archive at: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for fname in include_files:
            fpath = os.path.join(PROJECT_ROOT, fname)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=os.path.join("OCR-chall", fname))
                file_count += 1
                total_size += os.path.getsize(fpath)
                print(f"  + Added file: {fname}")

        # Add directories
        for dname in include_dirs:
            dir_path = os.path.join(PROJECT_ROOT, dname)
            if os.path.exists(dir_path):
                print(f"  + Adding directory: {dname}/ ...")
                for root, dirs, files in os.walk(dir_path):
                    # Filter out excluded subdirectories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for f in files:
                        if f.endswith('.pyc') or f.startswith('.'):
                            continue
                        abs_file = os.path.join(root, f)
                        rel_file = os.path.relpath(abs_file, PROJECT_ROOT)
                        arc_file = os.path.join("OCR-chall", rel_file)
                        zipf.write(abs_file, arcname=arc_file)
                        file_count += 1
                        total_size += os.path.getsize(abs_file)

    zip_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print("\n==================================================================")
    print(" SAGEMAKER PACKAGE CREATED SUCCESSFULLY! ")
    print("==================================================================")
    print(f"[OK] Output Package Location : {zip_filename}")
    print(f"[OK] Total Packaged Files    : {file_count:,}")
    print(f"[OK] Zip File Size           : {zip_mb:.2f} MB")
    print(f"[OK] Excluded Raw Images     : images/ folder omitted (saving ~1.2GB bandwidth)")

if __name__ == '__main__':
    main()
