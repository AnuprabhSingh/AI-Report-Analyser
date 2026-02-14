# Deployment Guide - Medical Interpreter System

## 🚀 Deployment Options

This application can be deployed in several ways. Choose the option that best suits your needs.

---

## Option 1: Docker (Recommended for Full Stack)

### Prerequisites
- Docker and Docker Compose installed
- Git repository initialized

### Steps

1. **Build and run locally to test:**
```bash
cd medical_interpreter
docker-compose up --build
```

2. **Access the application:**
   - Open http://localhost:5000

3. **Deploy to any Docker-compatible platform:**
   - **Railway.app** (Free tier available)
   - **Render** (Free tier available)
   - **DigitalOcean App Platform**
   - **AWS ECS/Fargate**
   - **Google Cloud Run**

---

## Option 2: Render.com (Easiest - Free Tier)

### Steps

1. **Push your code to GitHub:**
```bash
cd medical_interpreter
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medical-interpreter.git
git push -u origin main
```

2. **Deploy on Render:**
   - Go to https://render.com
   - Sign up/Login with GitHub
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will detect `render.yaml` and deploy automatically

3. **Access your app:**
   - Render will provide a URL like: `https://medical-interpreter.onrender.com`

**Note:** Free tier may have cold starts (app sleeps after 15 min of inactivity).

---

## Option 3: Railway.app (Easy - Free Tier)

### Steps

1. **Push code to GitHub** (see Option 2, step 1)

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Sign up/Login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will detect Dockerfile and deploy automatically

3. **Configure:**
   - Add environment variables in Railway dashboard if needed
   - Railway auto-assigns a domain

---

## Option 4: Split Deployment (Frontend + Backend Separate)

### Backend on Render/Railway/Heroku

1. **Deploy backend:**
```bash
# On Render or Railway, deploy with Dockerfile
# Or use Heroku:
heroku create medical-interpreter-api
git push heroku main
```

2. **Note your backend URL:** e.g., `https://medical-interpreter-api.onrender.com`

### Frontend on Vercel/Netlify

1. **Update API endpoint in frontend:**
```bash
cd frontend-react
```

Edit your API calls to use the backend URL.

2. **Deploy to Vercel:**
```bash
cd frontend-react
npm install -g vercel
vercel
```

Or connect via Vercel dashboard at https://vercel.com

3. **Deploy to Netlify:**
```bash
cd frontend-react
npm run build
# Drag and drop 'dist' folder to netlify.com
```

---

## Option 5: Local Development

### Quick Start

1. **Backend:**
```bash
cd medical_interpreter
pip install -r requirements.txt
python -m flask --app src/api.py run
```

2. **Frontend (separate terminal):**
```bash
cd frontend-react
npm install
npm run dev
```

3. **Access:**
   - Frontend: http://localhost:5173
   - Backend: http://localhost:5000

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

For production deployment, set these in your hosting platform:

- `FLASK_ENV=production`
- `FLASK_APP=src/api.py`
- `SECRET_KEY=your-secret-key` (generate a random string)
- `MAX_CONTENT_LENGTH=16777216`

---

## Pre-Deployment Checklist

- [ ] Test application locally with Docker
- [ ] Ensure all models are in `models/` directory
- [ ] Update CORS origins in production
- [ ] Set secure `SECRET_KEY` for production
- [ ] Test file upload functionality
- [ ] Verify ML models load correctly
- [ ] Check application health endpoint: `/api/health`

---

## Platform-Specific Notes

### Render
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ Custom domains supported
- ⚠️ Free tier sleeps after 15 minutes of inactivity

### Railway
- ✅ Free $5 credit monthly
- ✅ Simple deployment
- ✅ Persistent storage available
- ⚠️ Credit-based (may need upgrade)

### Vercel (Frontend only)
- ✅ Excellent for React apps
- ✅ Fast CDN
- ✅ Free tier generous
- ❌ Cannot host Python backend

### Docker-based (AWS, GCP, DO)
- ✅ Full control
- ✅ Scalable
- ✅ Professional deployment
- ⚠️ More complex setup
- 💰 Costs vary

---

## Troubleshooting

### Docker build fails
- Check Docker is running
- Ensure all files are present
- Try: `docker system prune -a` then rebuild

### Frontend can't connect to backend
- Check API URL in frontend code
- Verify CORS settings in `api.py`
- Check network/proxy settings

### Models not loading
- Ensure `models/` directory exists
- Check model files are in Docker image
- Verify file paths in code

### Out of memory on free tier
- ML models may be too large for free tier
- Consider upgrading to paid tier
- Or optimize model size

---

## Support

For issues or questions:
1. Check logs in your deployment platform
2. Test locally first with Docker
3. Verify all environment variables are set

---

## Quick Deploy Commands

**Docker:**
```bash
docker-compose up --build -d
```

**Render (after GitHub push):**
```bash
# Just connect GitHub repo - automatic deployment
```

**Railway:**
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway up
```

**Heroku:**
```bash
heroku login
heroku create medical-interpreter
git push heroku main
```

---

## Next Steps

1. Choose your deployment platform
2. Follow the steps for that platform above
3. Test your deployed application
4. Share the URL with users!

**Recommended for beginners:** Start with Render.com (Option 2) - it's the easiest with free tier.
