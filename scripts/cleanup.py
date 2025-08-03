#!/usr/bin/env python3
"""
Repository cleanup script for HIPAA QA System.
Removes temporary files, caches, and build artifacts.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_repository():
    """Clean up temporary files and build artifacts."""
    
    print("🧹 Cleaning up repository...")
    
    # Get repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    removed_count = 0
    
    # Python cache and compiled files
    patterns_to_remove = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/*.pyd",
        "**/.pytest_cache",
        "**/*.egg-info",
        "**/build",
        "**/dist"
    ]
    
    for pattern in patterns_to_remove:
        for path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")
                else:
                    os.remove(path)
                    print(f"  Removed file: {path}")
                removed_count += 1
            except OSError as e:
                print(f"  Warning: Could not remove {path}: {e}")
    
    # Temporary and backup files
    temp_patterns = [
        "*.tmp",
        "*.bak", 
        "*.orig",
        "*~",
        "*.log",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    for pattern in temp_patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                os.remove(path)
                print(f"  Removed temp file: {path}")
                removed_count += 1
            except OSError as e:
                print(f"  Warning: Could not remove {path}: {e}")
    
    # Clean logs directory but keep structure
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*"):
            if log_file.name != ".gitkeep":
                try:
                    log_file.unlink()
                    print(f"  Removed log: {log_file}")
                    removed_count += 1
                except OSError as e:
                    print(f"  Warning: Could not remove {log_file}: {e}")
    
    # Ensure .gitkeep exists in logs
    gitkeep_path = logs_dir / ".gitkeep"
    if not gitkeep_path.exists():
        gitkeep_path.touch()
        print(f"  Created: {gitkeep_path}")
    
    # Clean test results and generated files
    test_artifacts = [
        "enhanced_test_results.json",
        "enhanced_ingestion.log", 
        "enhanced_parser.log",
        "pdf_extraction.log",
        "section_parser.log"
    ]
    
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            try:
                os.remove(artifact)
                print(f"  Removed test artifact: {artifact}")
                removed_count += 1
            except OSError as e:
                print(f"  Warning: Could not remove {artifact}: {e}")
    
    print(f"\n✅ Cleanup complete! Removed {removed_count} items.")
    print("\n📝 Files preserved:")
    print("  • Source code (.py files)")
    print("  • Configuration files") 
    print("  • Documentation")
    print("  • Docker files")
    print("  • Data structure (empty directories with .gitkeep)")

if __name__ == "__main__":
    cleanup_repository()