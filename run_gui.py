"""
Quick Launcher - Automatically runs the GUI menu from the notebook
This script opens cell 22 of AI_Assignments.ipynb which contains the main GUI launcher
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("🚀 AI ASSIGNMENTS - QUICK LAUNCHER")
    print("=" * 60)
    print("\nStarting Jupyter Notebook and opening GUI menu...")
    print("Cell 22 will run automatically to show the assignment menu.\n")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, "AI_Assignments.ipynb")
    
    # Check if notebook exists
    if not os.path.exists(notebook_path):
        print("❌ Error: AI_Assignments.ipynb not found!")
        print(f"Looking in: {script_dir}")
        print("Make sure you're running this from the AI ASSIGNMENT folder.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Open Jupyter notebook
    try:
        print("Opening Jupyter Notebook...")
        print("\n📝 Instructions:")
        print("  1. Notebook will open in your browser")
        print("  2. Navigate to Cell 22")
        print("  3. Click 'Run' or press Shift+Enter")
        print("  4. The GUI menu will appear!")
        print("\n" + "=" * 60)
        
        # Change to the notebook directory
        os.chdir(script_dir)
        subprocess.run(["jupyter", "notebook", "AI_Assignments.ipynb"], check=True)
        
    except FileNotFoundError:
        print("\n❌ Jupyter not found!")
        print("\nPlease install Jupyter:")
        print("  conda activate cs221")
        print("  conda install jupyter -y")
        print("\nOr use pip:")
        print("  pip install jupyter")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Launcher cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
