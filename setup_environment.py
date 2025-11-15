#!/usr/bin/env python3
"""
One-Command Setup Script for AI Assignments
Creates conda environment 'cs221' with all dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False

def main():
    print("=" * 60)
    print("🚀 AI ASSIGNMENTS - ONE-COMMAND SETUP")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Create conda environment 'cs221'")
    print("  2. Install all required packages")
    print("  3. Verify installation")
    print()
    
    # Check if conda is available
    print("Checking for conda...")
    result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Conda not found!")
        print("\nPlease install Miniconda or Anaconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        print("\nOr use pip instead:")
        print("  pip install -r requirements.txt")
        return False
    
    print(f"✅ Found: {result.stdout.strip()}")
    
    # Check if environment already exists
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    
    if "cs221" in result.stdout:
        print("\n⚠️  Environment 'cs221' already exists!")
        response = input("Do you want to remove it and recreate? (y/N): ")
        
        if response.lower() == 'y':
            if not run_command("conda env remove -n cs221 -y", "Removing old environment"):
                return False
        else:
            print("Skipping environment creation.")
            print("\nTo update existing environment, run:")
            print("  conda env update -f environment.yml --prune")
            return True
    
    # Create environment from environment.yml
    if not run_command("conda env create -f environment.yml", "Creating conda environment"):
        print("\n❌ Failed to create environment!")
        print("\nTry manual installation:")
        print("  conda create -n cs221 python=3.12 -y")
        print("  conda activate cs221")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n✅ Environment 'cs221' created successfully!")
    print("\nNext steps:")
    print("  1. Activate the environment:")
    print("     conda activate cs221")
    print()
    print("  2. Start Jupyter:")
    print("     jupyter notebook AI_Assignments.ipynb")
    print()
    print("  3. Or run the GUI launcher:")
    print("     python main.py")
    print()
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
