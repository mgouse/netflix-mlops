import yaml
import sys
import os
import glob

def main():
    """
    Validates all Kubernetes YAML files in a given directory.
    Exits with code 1 if any file is invalid.
    """
    manifest_dir = 'kubernetes/manifests'
    print(f"Searching for YAML files in: {manifest_dir}")

    if not os.path.isdir(manifest_dir):
        print(f"Warning: Directory '{manifest_dir}' not found. Skipping validation.")
        sys.exit(0)

    yaml_files = glob.glob(os.path.join(manifest_dir, '*.yaml'))
    
    if not yaml_files:
        print(f"Warning: No YAML files found in '{manifest_dir}'.")
        sys.exit(0)

    has_errors = False
    for file_path in yaml_files:
        print(f"Validating {file_path}...")
        try:
            with open(file_path, 'r') as f:
                # Use load_all to handle files with multiple documents
                documents = list(yaml.safe_load_all(f))
                if not documents:
                    print(f"  - Warning: {file_path} is empty.")
                else:
                    print(f"  - ✓ {file_path} contains {len(documents)} valid YAML document(s).")
        except Exception as e:
            print(f"  - ❌ Error validating {file_path}: {e}", file=sys.stderr)
            has_errors = True
    
    if has_errors:
        print("\nValidation failed for one or more files.")
        sys.exit(1)
    else:
        print("\nAll Kubernetes YAML files are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()