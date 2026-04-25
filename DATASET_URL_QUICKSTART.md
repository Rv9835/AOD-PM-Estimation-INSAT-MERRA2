# ⚡ DATASET_URL Setup - Quick Start (5 mins)

## 🎯 The Problem
Your Streamlit Cloud app fails with:
```
FileNotFoundError: Failed to download unified dataset from DATASET_URL:
Failed to resolve 'your_public_dataset_url'
```

**Cause:** `DATASET_URL` placeholder not replaced with real download link

---

## ✅ The Solution (Choose Your Path)

### 🚀 Path A: Streamlit Cloud (Cloud Deployment)

**Prerequisites:** Code already pushed to GitHub main branch ✓

#### Step 1: Go to Streamlit Dashboard
```
https://share.streamlit.io → Your App → Settings (⋯ menu) → Secrets
```

#### Step 2: Add Secret
Paste this YAML (replace URL):
```yaml
DATASET_URL: 'https://huggingface.co/datasets/your-username/your-dataset/resolve/main/unified_dataset.csv?download=true'
```

#### Step 3: Reboot App
Click **"Reboot app"** button. You'll see in logs:
```
✓ Loaded DATASET_URL from Streamlit Secrets
```

**✅ Done!** App now downloads dataset automatically on first load.

---

### 🔧 Path B: Local Development (.env File)

#### Step 1: Create .env
```bash
cp .env.example .env
```

#### Step 2: Edit .env
```bash
# Replace URL with your download link
DATASET_URL=https://huggingface.co/datasets/your-username/your-dataset/resolve/main/unified_dataset.csv?download=true
```

#### Step 3: Test Locally
```bash
./run_project.sh
```

**✅ Done!** App auto-loads `.env` and runs locally.

---

## 📊 Where to Get Your Dataset URL

### 🤗 **Hugging Face (Easiest - Recommended)**
1. Sign up at [huggingface.co](https://huggingface.co)
2. Create Dataset → Upload `unified_dataset.csv` → Set Public
3. Copy URL: `https://huggingface.co/datasets/{username}/{repo}/resolve/main/unified_dataset.csv?download=true`

### 📁 **Google Drive**
1. Upload file → Share → Public
2. Format: `https://drive.google.com/uc?export=download&id={FILE_ID}`

### ☁️ **AWS S3**
1. Upload to public bucket
2. Copy: `https://{bucket}.s3.{region}.amazonaws.com/unified_dataset.csv`

---

## 🔍 Validate Your URL (Before Using)

**Test it works:**
```bash
curl -I "https://your-dataset-url-here"
```

**Should show:**
```
HTTP/1.1 200 OK
Content-Type: text/csv (or application/gzip)
Content-Length: 123456789
```

**❌ Bad signs:**
- `Content-Type: text/html` → It's a redirect, not direct download
- `HTTP 403 Forbidden` → Access denied

---

## 🚨 Troubleshooting

| Error | Fix |
|-------|-----|
| `DATASET_URL not configured` | Add to `.env` (local) or Streamlit Secrets (cloud) |
| `Failed to resolve 'your_public_dataset_url'` | Replace placeholder with real URL |
| `.env` not working | Ensure you've done: `cp .env.example .env` |
| Still seeing old URL | Reboot Streamlit Cloud app after updating Secrets |

---

## 🧪 Validation Script

Run this to verify setup:
```bash
python3 validate_dataset_config.py
```

Expected output:
```
[Test 1] Checking .env file...
  ✓ .env file found

[Test 2] Testing DATASET_URL retrieval...
  ✓ python-dotenv imported successfully

[Test 3] Checking environment variables...
  ✓ DATASET_URL configured: https://...

[Test 4] Validating URL format...
  ✓ URL uses HTTPS
  ✓ URL from recognized hosting provider

[Test 5] Testing get_dataset_url() function...
  ✓ get_dataset_url() returned: https://...

🎉 Configuration VALID! Ready to deploy.
```

---

## 📋 Implementation Details

### What Changed
- **Removed:** Hardcoded placeholder `https://YOUR_PUBLIC_DATASET_URL/...`
- **Added:** Secure configuration system with:
  - `get_dataset_url()` → Retrieves from Streamlit Secrets (cloud) or `.env` (local)
  - Validation & error messages → Clear guidance if URL missing/invalid
  - Auto-download → Dataset downloads on first app load if missing
  - Compression support → Handles `.csv.gz` and `.parquet` formats

### Code Flow
```
App starts
  ↓
frontend/app_dashboard.py loads .env (if local)
  ↓
backend/multi_city_data.py calls get_dataset_url()
  ↓
Checks: Streamlit Secrets → Environment Variable → Raise Error
  ↓
If dataset file missing: Auto-downloads from URL
  ↓
App loads & runs
```

---

## 🔐 Security Best Practices

✅ **Do:**
- Use Streamlit Secrets for cloud (encrypted)
- Use .env for local (git-ignored)
- Test URL before committing
- Use HTTPS URLs only
- Consider presigned URLs for S3 (time-limited)

❌ **Don't:**
- Commit `.env` file to Git
- Use HTTP (only HTTPS)
- Share dataset URLs publicly if private data
- Hardcode credentials in code

---

## 📚 Full Documentation

See [docs/DATASET_CONFIGURATION.md](docs/DATASET_CONFIGURATION.md) for:
- Complete setup guide with screenshots
- All hosting options explained
- URL validation tests
- Detailed troubleshooting
- Security recommendations

---

## ✨ Next Steps

1. **Choose hosting:** Hugging Face (easiest) or Drive/S3
2. **Get URL:** Upload dataset, copy direct-download link
3. **Validate URL:** `curl -I "your-url"`
4. **Local setup:** Create `.env` and test with `./run_project.sh`
5. **Cloud setup:** Add to Streamlit Secrets and reboot app
6. **Verify:** Check logs for `✓ Loaded DATASET_URL from...`

**🎉 You're ready to deploy!**
