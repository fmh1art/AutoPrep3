#!/usr/bin/env python3
import re

# Read main log
with open("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/main_log.ansi", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# The 4 cases we care about
target_cases = [
    'django__django-13279',
    'django__django-13670',
    'scikit-learn__scikit-learn-14983',
    'sphinx-doc__sphinx-8551'
]

print("="*80)
print("ANALYZING GIT PATCHES FROM MAIN LOG")
print("="*80)

for case in target_cases:
    print(f"\n\n{'='*80}")
    print(f"CASE: {case}")
    print('='*80)
    
    # Find where this case starts and ends
    start_pattern = re.compile(r'\[Worker-PE\] Starting ' + re.escape(case))
    end_pattern = re.compile(r'\[Worker-PE\] Completed ' + re.escape(case))
    
    start_match = start_pattern.search(content)
    end_match = end_pattern.search(content)
    
    if start_match and end_match:
        case_content = content[start_match.end():end_match.start()]
        
        # Look for git_patch in this content, or look for the part around when they run git diff
        print("Looking for git diff in this case's execution...")
        
        # Find any diff output
        diff_matches = re.findall(r'(diff --git.*?)(?=\[|$)', case_content, re.DOTALL)
        
        if diff_matches:
            for i, diff in enumerate(diff_matches):
                print(f"\n--- DIFF #{i+1} ---")
                print(diff[:3000])
        else:
            print("No diff output found!")
            
        # Also check if there's any git commit or git add output
        print("\nChecking for git commit or git add...")
        git_add_matches = re.findall(r'(git add.*?)(?=\[|$)', case_content, re.DOTALL)
        git_commit_matches = re.findall(r'(git commit.*?)(?=\[|$)', case_content, re.DOTALL)
        git_diff_matches = re.findall(r'(git diff.*?)(?=\[|$)', case_content, re.DOTALL)
        
        if git_add_matches:
            print("\nGIT ADD")
            for m in git_add_matches:
                print(m[:1000])
                
        if git_commit_matches:
            print("\nGIT COMMIT")
            for m in git_commit_matches:
                print(m[:1000])
                
        if git_diff_matches:
            print("\nGIT DIFF")
            for m in git_diff_matches:
                print(m[:3000])
