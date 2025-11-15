#!/usr/bin/env python3
"""
Pre-Submission Verification Script
Checks that everything is ready for submission to Dr. Nakibule Mary
"""

import os
import json
from pathlib import Path

print("🔍 AI ASSIGNMENT SUBMISSION VERIFICATION")
print("=" * 60)

# Track issues
issues = []
warnings = []
passed = []

# 1. Check main files
print("\n📄 Checking Main Files...")
required_files = [
    'AI_Assignments.ipynb',
    'requirements.txt',
    'SETUP_GUIDE.md',
    'DOCUMENTATION.md',
    'README.md',
    'main.py'
]

for file in required_files:
    if os.path.exists(file):
        passed.append(f"✅ {file} exists")
        print(f"  ✅ {file}")
    else:
        issues.append(f"❌ Missing: {file}")
        print(f"  ❌ Missing: {file}")

# 2. Check assignment folders
print("\n📁 Checking Assignment Folders...")
assignment_folders = [
    'foundations',
    'sentiment', 
    'route',
    'mountaincar',
    'pacman',
    'scheduling',
    'car'
]

for folder in assignment_folders:
    if os.path.exists(folder):
        # Check for submission.py
        submission_file = os.path.join(folder, 'submission.py')
        if os.path.exists(submission_file):
            passed.append(f"✅ {folder}/ with submission.py")
            print(f"  ✅ {folder}/ with submission.py")
        else:
            warnings.append(f"⚠️  {folder}/ missing submission.py")
            print(f"  ⚠️  {folder}/ missing submission.py")
    else:
        issues.append(f"❌ Missing folder: {folder}/")
        print(f"  ❌ Missing folder: {folder}/")

# 3. Check notebook content
print("\n📓 Checking Notebook Content...")
try:
    with open('AI_Assignments.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check for team members
    has_team_info = False
    for cell in notebook['cells'][:10]:  # Check first 10 cells
        source = ''.join(cell.get('source', []))
        if 'Kigozi Allan' in source or 'Group C' in source:
            has_team_info = True
            break
    
    if has_team_info:
        passed.append("✅ Team information found in notebook")
        print("  ✅ Team information present")
    else:
        warnings.append("⚠️  Team information not found in first 10 cells")
        print("  ⚠️  Team information not clearly visible")
    
    # Count code cells
    code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
    markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    
    passed.append(f"✅ Notebook has {code_cells} code cells and {markdown_cells} markdown cells")
    print(f"  ✅ {code_cells} code cells, {markdown_cells} markdown cells")
    
except Exception as e:
    issues.append(f"❌ Error reading notebook: {str(e)}")
    print(f"  ❌ Error reading notebook: {str(e)}")

# 4. Check requirements.txt
print("\n📦 Checking Dependencies...")
try:
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = ['pygame', 'numpy', 'plotly', 'gymnasium']
    found_packages = []
    
    for package in required_packages:
        if package in content:
            found_packages.append(package)
            print(f"  ✅ {package} in requirements.txt")
        else:
            issues.append(f"❌ {package} missing from requirements.txt")
            print(f"  ❌ {package} missing from requirements.txt")
    
    # Check for local modules (shouldn't be there)
    bad_modules = ['foundations', 'sentiment', 'pacman', 'submission', 'grader']
    found_bad = []
    for module in bad_modules:
        if module in content and not content.split(module)[0].endswith('#'):
            found_bad.append(module)
    
    if found_bad:
        warnings.append(f"⚠️  Local modules in requirements.txt: {', '.join(found_bad)}")
        print(f"  ⚠️  Found local modules (should be filtered): {', '.join(found_bad)}")
    else:
        passed.append("✅ No local modules in requirements.txt")
        print("  ✅ No local modules in requirements.txt")
        
except Exception as e:
    issues.append(f"❌ Error reading requirements.txt: {str(e)}")
    print(f"  ❌ Error reading requirements.txt: {str(e)}")

# 5. Check documentation
print("\n📚 Checking Documentation...")
docs = ['SETUP_GUIDE.md', 'DOCUMENTATION.md', 'README.md']
for doc in docs:
    if os.path.exists(doc):
        size = os.path.getsize(doc)
        if size > 1000:  # At least 1KB
            passed.append(f"✅ {doc} has content ({size} bytes)")
            print(f"  ✅ {doc} ({size} bytes)")
        else:
            warnings.append(f"⚠️  {doc} seems too small ({size} bytes)")
            print(f"  ⚠️  {doc} seems small ({size} bytes)")

# 6. Print summary
print("\n" + "=" * 60)
print("📊 VERIFICATION SUMMARY")
print("=" * 60)

print(f"\n✅ Passed Checks: {len(passed)}")
print(f"⚠️  Warnings: {len(warnings)}")
print(f"❌ Issues: {len(issues)}")

if warnings:
    print("\n⚠️  WARNINGS:")
    for warning in warnings:
        print(f"  {warning}")

if issues:
    print("\n❌ CRITICAL ISSUES:")
    for issue in issues:
        print(f"  {issue}")
    print("\n🚨 Please fix these issues before submission!")
else:
    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED!")
    print("=" * 60)
    print("\n✅ Your submission is ready!")
    print("✅ All required files present")
    print("✅ All assignment folders included")
    print("✅ Dependencies correctly configured")
    print("✅ Documentation complete")
    print("\n📦 Ready to submit to Dr. Nakibule Mary!")
    print("\nSubmission Deadline: November 15, 2025")
    print("\n🚀 Good luck with your submission!")

print("\n" + "=" * 60)
