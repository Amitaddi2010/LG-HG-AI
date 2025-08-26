# Deployment Instructions

## Model File Setup
The model file `final_efficientnet_b0_cv.pth` has been copied to `webapp/backend/` but is excluded from git due to size.

## For Render Deployment:

### Option 1: Upload Model via Render Dashboard
1. Deploy without model file (will use random weights)
2. Upload model file via Render's file manager
3. Restart service

### Option 2: Use Git LFS (Recommended)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pth"
git add .gitattributes

# Add and commit
git add webapp/backend/final_efficientnet_b0_cv.pth
git commit -m "Add model file with LFS"
git push
```

### Option 3: External Storage
Upload model to cloud storage (AWS S3, Google Drive) and modify app.py to download on startup.

## Current Status:
- ✅ Model copied to webapp/backend/
- ✅ App.py updated to use local model path
- ✅ Ready for deployment (with model file handling)