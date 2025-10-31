#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all components are properly installed and configured.
"""

import sys
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pdfplumber': 'pdfplumber',
        'sklearn': 'scikit-learn',
        'flask': 'flask',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'jupyter': 'jupyter'
    }
    
    missing = []
    installed = []
    
    for module, package in required.items():
        try:
            __import__(module)
            installed.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    return len(missing) == 0, missing

def check_file_structure():
    """Check if all required files exist."""
    required_files = [
        'src/extractor.py',
        'src/rule_engine.py',
        'src/predictor.py',
        'src/model_trainer.py',
        'src/utils.py',
        'src/api.py',
        'src/__init__.py',
        'main.py',
        'demo.py',
        'requirements.txt',
        'README.md',
        'notebooks/data_analysis.ipynb'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    return len(missing) == 0, missing

def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        'data/sample_reports',
        'data/processed',
        'notebooks',
        'src'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            missing.append(dir_path)
    
    return len(missing) == 0, missing

def main():
    """Run all checks."""
    print("=" * 60)
    print("MEDICAL REPORT INTERPRETER - SETUP VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        all_good = False
    
    # Check dependencies
    print("\n2. Checking required packages...")
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_good = False
        print(f"\n⚠ Missing packages: {', '.join(missing_deps)}")
        print(f"   Install with: pip install {' '.join(missing_deps)}")
    
    # Check file structure
    print("\n3. Checking file structure...")
    files_ok, missing_files = check_file_structure()
    if not files_ok:
        all_good = False
        print(f"\n⚠ Missing files: {', '.join(missing_files)}")
    
    # Check directories
    print("\n4. Checking directories...")
    dirs_ok, missing_dirs = check_directories()
    if not dirs_ok:
        all_good = False
        print(f"\n⚠ Missing directories: {', '.join(missing_dirs)}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYour system is ready! Try:")
        print("  python demo.py")
        print("  python main.py --help")
        print("  jupyter notebook notebooks/data_analysis.ipynb")
    else:
        print("❌ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease fix the issues above and run this script again.")
        print("\nQuick fix:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
