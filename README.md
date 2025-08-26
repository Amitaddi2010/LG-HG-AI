# HistoAI - Histopathology Classification WebApp

## 🚀 Deploy to Render

### Option 1: One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Option 2: Manual Deploy
1. Push code to GitHub repository
2. Connect to Render: https://render.com
3. Create new Web Service
4. Connect your GitHub repo
5. Use these settings:
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && gunicorn -c gunicorn_config.py app:app`
   - **Environment**: Python 3.9

## 🏠 Local Development

```bash
# Backend
cd webapp/backend
pip install -r requirements.txt
python app.py

# Frontend (separate terminal)
cd webapp/frontend  
python -m http.server 8000
```

## 📊 Features
- EfficientNet-B0 model for LG/HG classification
- Real-time pipeline visualization
- Professional neumorphism UI
- 94% accuracy with 5-fold cross-validation
- Responsive dashboard design

## 🔧 Tech Stack
- **Backend**: Flask, PyTorch, OpenCV
- **Frontend**: Vanilla HTML/CSS/JS
- **Deployment**: Render, Gunicorn