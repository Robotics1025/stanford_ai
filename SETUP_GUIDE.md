# AI Programming Assignments - Setup Guide

**Course:** Artificial Intelligence  
**Instructor:** Dr. Nakibule Mary  
**Group:** Group C  
**Submission Date:** November 15, 2025

---

## 📦 Installation Instructions

### Prerequisites
- Python 3.12 (tested on Python 3.12.12)
- **Option A:** Conda (Miniconda or Anaconda) - RECOMMENDED
- **Option B:** pip (Python package manager)
- Jupyter Notebook or JupyterLab

---

## 🎯 ONE-COMMAND SETUP (RECOMMENDED)

### For Conda Users (Windows):
```bash
setup_environment.bat
```

### For Conda Users (Mac/Linux):
```bash
python setup_environment.py
```

This will automatically:
- Create conda environment named 'cs221'
- Install all required packages
- Verify installation

**Then just activate and run:**
```bash
conda activate cs221
jupyter notebook AI_Assignments.ipynb
```

---

## 📦 Manual Installation Methods

### Step 1: Install Required Packages

**Option A - Using Conda Environment File (BEST - Installs Everything):**
```bash
conda env create -f environment.yml
conda activate cs221
```

**Option B - Using pip:**
```bash
pip install -r requirements.txt
```

**Option C - Using conda packages directly:**
```bash
conda install pygame numpy plotly matplotlib pandas scipy gymnasium -c conda-forge
```

**Option D - Manual installation:**
```bash
pip install pygame>=2.5.0 numpy>=1.24.0 plotly>=5.14.0 matplotlib>=3.7.0 pandas>=2.0.0 scipy>=1.10.0 gymnasium>=0.29.0
```

### Step 2: Verify Installation

Run this command to verify all packages are installed correctly:
```bash
python -c "import pygame, numpy, plotly, matplotlib, pandas, scipy, gymnasium; print('All packages installed successfully!')"
```

### Step 3: Open the Notebook

```bash
jupyter notebook AI_Assignments.ipynb
```
or
```bash
jupyter lab AI_Assignments.ipynb
```

---

## 📁 Project Structure

```
AI_Assignments.ipynb    # Main notebook with all assignments
requirements.txt        # External dependencies
DOCUMENTATION.md        # Detailed documentation
SETUP_GUIDE.md          # This file
main.py                 # GUI launcher (optional)

Assignment Folders:
├── foundations/        # Assignment 1: Foundations
├── sentiment/          # Assignment 2: Sentiment Analysis
├── route/              # Assignment 3: Route Planning
├── mountaincar/        # Assignment 4: Reinforcement Learning
├── pacman/             # Assignment 5: Pacman AI
├── scheduling/         # Assignment 6: CSP Scheduling
└── car/                # Assignment 7: Autonomous Driving
```

---

## 🚀 Running the Assignments

### Method 1: Using Jupyter Notebook (Recommended)
1. Open `AI_Assignments.ipynb` in Jupyter
2. Run the first cell to see the assignment menu
3. Click on any assignment to navigate to it
4. Each assignment has interactive menus for testing and grading

### Method 2: Using GUI Launcher (Optional)
```bash
python main.py
```
This opens a Pygame GUI with buttons for all 7 assignments.

---

## 📝 Assignment Details

### Assignment 1: Foundations
- **Topics:** Python basics, data structures, algorithms
- **Files:** `foundations/submission.py`
- **Grader:** Run grader from notebook menu

### Assignment 2: Sentiment Analysis
- **Topics:** Text classification, machine learning
- **Files:** `sentiment/submission.py`
- **Data:** `polarity.train`, `polarity.dev`

### Assignment 3: Route Planning
- **Topics:** Search algorithms (UCS, A*), heuristics
- **Files:** `route/submission.py`, `route/visualization.py`
- **Features:** Interactive map visualization

### Assignment 4: Mountain Car (Reinforcement Learning)
- **Topics:** Q-Learning, value iteration, policy optimization
- **Files:** `mountaincar/submission.py`
- **Environment:** Custom OpenAI Gymnasium environment

### Assignment 5: Pacman AI
- **Topics:** Game AI, minimax, expectimax, evaluation functions
- **Files:** `pacman/submission.py`
- **Layouts:** Multiple game layouts available

### Assignment 6: Scheduling (CSP)
- **Topics:** Constraint satisfaction, backtracking, arc consistency
- **Files:** `scheduling/submission.py`
- **Problems:** N-Queens, course scheduling

### Assignment 7: Autonomous Car
- **Topics:** Particle filters, Bayes nets, probabilistic reasoning
- **Files:** `car/submission.py`
- **Simulation:** Interactive car driving environment

---

## 🔧 Important Notes

### Path Handling
- **All paths are relative** - The notebook works from any directory
- No hardcoded paths - works on any computer
- Assignment folders must be in the same directory as the notebook

### No Additional Configuration Needed
- Python environment is automatically detected
- No conda environment activation required
- All imports use relative paths

### Troubleshooting

**Problem:** Package import errors
**Solution:** Make sure all packages in requirements.txt are installed

**Problem:** Assignment folder not found
**Solution:** Ensure all assignment folders are in the same directory as the notebook

**Problem:** Visualization doesn't open
**Solution:** Check that your browser allows file:// URLs, or manually open the generated HTML file

**Problem:** Grader doesn't run
**Solution:** Navigate to the assignment section in the notebook and use the interactive menu

---

## ✅ Verification Checklist

Before grading, please verify:

- [ ] All packages installed successfully
- [ ] Notebook opens without errors
- [ ] All 7 assignment folders are present
- [ ] Team member information is visible at the top
- [ ] Each assignment has proper numbering (Assignment 1-7)
- [ ] Interactive menus work for all assignments
- [ ] Graders run successfully
- [ ] Visualizations display correctly (route, mountaincar)

---

## 👥 Team Members

**Group C:**
1. Kigozi Allan - 2023/dcse/0101/ps
2. Keith Paul Kato - 2023/dcse/0092/ps
3. Mugole Joel - 2023/dcse/0103/ps
4. Nalubega Shadiah - 2023/dcse/0036/ps
5. Ageno Elizabeth - 2023/dcse/0081/ps

---

## 📧 Support

If you encounter any issues during setup or grading:
- Check the DOCUMENTATION.md file for detailed code explanations
- All assignment code is well-documented with inline comments
- Each algorithm implementation includes explanatory comments

---

## 🎓 Academic Integrity

This submission represents original work completed by Group C for the Artificial Intelligence course. All code implementations follow the assignment specifications provided in the Stanford CS221 curriculum.

**Submission Date:** November 15, 2025  
**Course Instructor:** Dr. Nakibule Mary  
**Institution:** [Your University Name]

---

Thank you for reviewing our work! 🙏
