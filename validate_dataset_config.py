#!/usr/bin/env python3
"""
Quick validator for DATASET_URL configuration
Run this to verify setup before deploying to Streamlit Cloud
"""

import os
import sys
from pathlib import Path

# Test 1: Check if .env file exists
print("=" * 60)
print("DATASET_URL Configuration Validator")
print("=" * 60)

project_root = Path(__file__).parent
env_file = project_root / ".env"

print("\n[Test 1] Checking .env file...")
if env_file.exists():
    print(f"  ✓ .env file found at {env_file}")
else:
    print(f"  ⚠ .env file NOT found at {env_file}")
    print(f"    → Copy from .env.example: cp .env.example .env")

# Test 2: Try loading from backend
print("\n[Test 2] Testing DATASET_URL retrieval...")
sys.path.insert(0, str(project_root / "backend" / "src"))

try:
    from dotenv import load_dotenv
    if env_file.exists():
        load_dotenv(env_file)
    print("  ✓ python-dotenv imported successfully")
except ImportError:
    print("  ⚠ python-dotenv not installed")
    print("    → Run: pip install python-dotenv")

# Test 3: Check environment variable
print("\n[Test 3] Checking environment variables...")
dataset_url = os.getenv("DATASET_URL", "").strip()

if not dataset_url:
    print("  ❌ DATASET_URL not configured")
    print("    → Add DATASET_URL to .env or environment")
elif dataset_url.startswith("https://YOUR_PUBLIC"):
    print("  ❌ DATASET_URL is still set to placeholder")
    print(f"    → Current value: {dataset_url[:50]}...")
    print("    → Replace with real URL in .env")
else:
    print(f"  ✓ DATASET_URL configured: {dataset_url[:60]}...")

# Test 4: Validate URL format
print("\n[Test 4] Validating URL format...")
if dataset_url:
    if dataset_url.startswith("https://"):
        print(f"  ✓ URL uses HTTPS")
    else:
        print(f"  ❌ URL does not use HTTPS")
    
    if any(x in dataset_url for x in ["huggingface.co", "drive.google.com", ".s3."]):
        print(f"  ✓ URL from recognized hosting provider")
    else:
        print(f"  ⚠ URL from unknown provider (may still work)")

# Test 5: Try to import and test the actual function
print("\n[Test 5] Testing get_dataset_url() function...")
try:
    from airpollution.multi_city_data import get_dataset_url
    
    try:
        url = get_dataset_url()
        print(f"  ✓ get_dataset_url() returned: {url[:60]}...")
        print("\n🎉 Configuration VALID! Ready to deploy.")
    except ValueError as e:
        print(f"  ❌ get_dataset_url() raised ValueError:")
        print(f"     {str(e)[:200]}...")
        
except ImportError as e:
    print(f"  ⚠ Could not import get_dataset_url(): {e}")
    print("    → This is OK if running outside the backend context")

print("\n" + "=" * 60)
print("Next steps:")
print("  1. Update .env with real DATASET_URL")
print("  2. Run: ./run_project.sh (local test)")
print("  3. If OK, add DATASET_URL to Streamlit Secrets")
print("  4. Reboot app on Streamlit Cloud")
print("=" * 60)
