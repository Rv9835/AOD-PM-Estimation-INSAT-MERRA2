# DATASET_URL Configuration Guide

## Overview

The `DATASET_URL` is now securely managed through configuration rather than hardcoded. This guide explains how to set it up for both **local development** and **Streamlit Cloud deployment**.

---

## 🌐 Method 1: Local Development (.env File)

### Step 1: Create `.env` File
In the project root directory, create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

### Step 2: Add Your Dataset URL
Edit `.env` and replace the placeholder:

```bash
# .env
DATASET_URL=https://huggingface.co/datasets/your-username/your-dataset/resolve/main/unified_dataset.csv?download=true
```

### Step 3: Test Locally
Run the dashboard:

```bash
./run_project.sh
```

The app will automatically load `DATASET_URL` from `.env` via `python-dotenv`.

### ✅ Notes for Local Development
- **`.env` is git-ignored** (see `.gitignore`) – never commits credentials
- **python-dotenv auto-loads** `.env` on startup (no manual loading needed)
- **Test URL validity** before committing code:
  ```bash
  curl -I "https://your-url/unified_dataset.csv"
  # Should return HTTP 200, not HTML redirect
  ```

---

## ☁️ Method 2: Streamlit Cloud (Secrets Management)

### Step 1: Deploy to Streamlit Cloud
First, push your code to GitHub (main branch tracked by Streamlit):

```bash
git add -A
git commit -m "Add DATASET_URL configuration management"
git push origin main
```

Then deploy via [Streamlit Community Cloud](https://share.streamlit.io).

### Step 2: Access App Settings
1. Go to **[https://share.streamlit.io](https://share.streamlit.io)**
2. Find your app: `aod-pm-estimation-insat-merra2`
3. Click the **3-dot menu** (⋯) → **Settings**

### Step 3: Configure Secrets
1. Scroll to **"Secrets"** section
2. Click **"Edit secrets"**
3. Paste the following YAML (replace URL with yours):

```yaml
DATASET_URL: 'https://huggingface.co/datasets/your-username/your-dataset/resolve/main/unified_dataset.csv?download=true'
```

**Important:** Do NOT include `https://YOUR_PUBLIC...` placeholder – use real URL

### Step 4: Reboot App
Click **"Reboot app"** button. The app will restart with secrets loaded.

### Step 5: Verify Success
Check the app logs:
```
✓ Loaded DATASET_URL from Streamlit Secrets
```

---

## 📋 Recommended Dataset Hosting Options

### 🤗 **Option A: Hugging Face Datasets (Recommended)**

**Why:** Easiest setup, stable direct links, no auth required

**Steps:**
1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to **Datasets** → **Create new dataset**
3. Upload your `unified_dataset.csv` (or `.csv.gz` for compression)
4. Set visibility to **Public**
5. Go to **Files** tab
6. Right-click file → **Copy browser download link**
7. Add `?download=true` parameter to ensure direct download:

```
https://huggingface.co/datasets/your-username/your-dataset-repo/resolve/main/unified_dataset.csv?download=true
```

**Test it:**
```bash
curl -I "https://huggingface.co/datasets/your-username/your-dataset-repo/resolve/main/unified_dataset.csv?download=true"
# Should return: HTTP/1.1 200 OK
```

---

### 📁 **Option B: Google Drive (Alternative)**

**Steps:**
1. Upload file to Google Drive
2. Right-click file → **Share** → **Anyone with link**  
3. Copy link: `https://drive.google.com/file/d/FILE_ID/view`
4. Extract `FILE_ID` from URL
5. Format as direct-download URL:

```
https://drive.google.com/uc?export=download&id=FILE_ID
```

**Example:** If link is `https://drive.google.com/file/d/1A2B3C4D5E6F7G8H/view`
```
FILE_ID = 1A2B3C4D5E6F7G8H
URL = https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H
```

**Test it:**
```bash
curl -I "https://drive.google.com/uc?export=download&id=FILE_ID"
```

---

### ☁️ **Option C: AWS S3 (Advanced)**

**Steps:**
1. Upload to public S3 bucket
2. Grant **public read** access
3. Copy object URL:

```
https://your-bucket-name.s3.region.amazonaws.com/unified_dataset.csv
```

Or generate presigned URL:
```bash
aws s3 presign s3://your-bucket/unified_dataset.csv --expires-in 604800
# Returns: https://...?X-Amz-SignedHeaders=... (7-day expiry)
```

---

## 🔍 Validating Your URL

Before using a URL, verify it's a **direct download** (not HTML redirect):

```bash
# Test 1: Check HTTP headers
curl -I "your-url-here"
# Should show:
#   HTTP/1.1 200 OK
#   Content-Type: text/csv  (or application/gzip)
#   Content-Length: 123456789

# Test 2: Download first few bytes
curl -r 0-100 "your-url-here" | head -c 100
# Should show CSV header or gzip magic bytes (1f 8b)
```

**❌ Bad signs:**
- `Content-Type: text/html` – It's returning a webpage, not the file
- `HTTP/1.1 301/302` – It's a redirect; follow it to final URL
- `HTTP/1.1 403 Forbidden` – Access denied; check permissions

---

## 🚀 Code Behavior

### How DATASET_URL is Retrieved

The code tries sources in this priority order:

1. **Streamlit Secrets** (`st.secrets["DATASET_URL"]`) – Streamlit Cloud
2. **Environment Variable** (`os.getenv("DATASET_URL")`) – Local .env or shell export
3. **Raise Error** – If missing or invalid

### Error Messages (User-Friendly)

If URL is not configured:
```
❌ DATASET_URL not configured!

For Streamlit Cloud:
  1. Go to https://share.streamlit.io → Select your app → Settings
  2. Scroll to 'Secrets' section
  3. Paste this YAML:
     DATASET_URL: 'https://...'
  4. Click Save & Reboot app

For Local Development:
  • Create .env file in project root with: DATASET_URL=https://...
  • Or export DATASET_URL='https://...' in your shell
```

If URL is still placeholder:
```
❌ DATASET_URL is still set to placeholder value!
Please update it to a real download URL in Streamlit Secrets or environment variables.
```

---

## 🔐 Security Best Practices

1. **Never commit `.env`** – Git automatically ignores it (see `.gitignore`)
2. **Use Streamlit Secrets** on cloud (encrypted at rest, scoped to app)
3. **For production:** Use time-limited presigned URLs (AWS S3) for extra security
4. **Rotate credentials** if dataset URL contains auth tokens
5. **Test endpoint access** before deployment

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `DATASET_URL not configured` error | Add to `.env` (local) or Streamlit Secrets (cloud) |
| `Failed to download` error | Check URL is direct-download (`curl -I`); verify publicly accessible |
| `.env` file not loaded | Ensure `python-dotenv>=1.0.0` in requirements.txt (✓ already there) |
| Streamlit Cloud still uses old URL | Reboot app after updating Secrets |
| Dataset download times out | Check file size; consider uploading compressed `.csv.gz` format |

---

## ✅ Checklist

- [ ] Chosen hosting solution (Hugging Face recommended)
- [ ] Uploaded dataset file and set to public
- [ ] Tested URL with `curl -I` (shows 200 OK, not HTML)
- [ ] Created `.env` file for local dev (copied from `.env.example`)
- [ ] Set `DATASET_URL` in `.env`
- [ ] Tested locally with `./run_project.sh`
- [ ] Pushed code to GitHub main branch
- [ ] Added `DATASET_URL` to Streamlit Cloud Secrets
- [ ] Rebooted Streamlit Cloud app
- [ ] Verified logs show `✓ Loaded DATASET_URL from Streamlit Secrets`
