# Deploy to Render Without Git

## Option 1: Direct Upload to Render

1. **Zip the webapp folder:**
   ```bash
   cd /Users/amit/Desktop/LG-HG
   zip -r histoai-webapp.zip webapp/ -x "webapp/backend/final_efficientnet_b0_cv.pth"
   ```

2. **Go to Render.com:**
   - Create account at https://render.com
   - Click "New +" → "Web Service"
   - Choose "Deploy an existing image or build from source code"
   - Select "Upload from computer"
   - Upload the zip file

3. **Configure Service:**
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && gunicorn -c gunicorn_config.py app:app`
   - **Environment**: Python 3.9

## Option 2: Fix Git Authentication

1. **Generate GitHub Personal Access Token:**
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token with repo permissions

2. **Use token as password:**
   ```bash
   git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/Amitaddi2010/LG-HG-AI.git
   git push origin master
   ```

## Option 3: Deploy Demo Version (No Model)

The app will work with random weights for demonstration. Real predictions require the trained model file.

**Current Status:** Ready to deploy without model file for demo purposes.