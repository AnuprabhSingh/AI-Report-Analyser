# 🚀 Deployment Guide

Complete deployment documentation for the Medical Report Interpreter System.

## Table of Contents
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Split Deployment (Backend + Frontend)](#split-deployment)
- [Environment Configuration](#environment-configuration)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### ⚡ 5-Minute Deploy (Recommended)

**Option 1: Render.com** (Easiest - Free Tier)

```bash
# 1. Push to GitHub
cd medical_interpreter
git add .
git commit -m "Ready for deployment"
git push

# 2. Deploy on Render
# - Go to https://render.com
# - Click "New +" → "Blueprint"
# - Connect your GitHub repo
# - Click "Apply" (auto-detects render.yaml)
# - Wait 5-10 minutes
# - Get your URL: https://your-app.onrender.com
```

**Option 2: Railway.app** (Free Credit)

```bash
# 1. Push to GitHub (same as above)

# 2. Deploy on Railway
# - Go to https://railway.app
# - "New Project" → "Deploy from GitHub"
# - Select your repo
# - Auto-deploys via Dockerfile
```

**Option 3: Docker Locally**

```bash
cd medical_interpreter
docker-compose up --build
# Open http://localhost:5000
```

**Test Before Deploy:**
```bash
./test_deployment.sh
```

---

## Deployment Options

### Overview of Platforms

| Platform | Difficulty | Free Tier | Best For |
|----------|-----------|-----------|----------|
| **Render.com** | ⭐ Easy | ✅ Yes | Quick deployment, beginners |
| **Railway.app** | ⭐ Easy | ✅ $5/month credit | Simple deployment |
| **Docker** | ⭐⭐ Medium | ✅ Self-hosted | Full control, testing |
| **Vercel + Render** | ⭐⭐ Medium | ✅ Yes | Split frontend/backend |
| **AWS/GCP/Azure** | ⭐⭐⭐ Hard | 💰 Varies | Enterprise, scalable |

---

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- Git repository initialized

### Local Testing

1. **Build and run:**
```bash
cd medical_interpreter
docker-compose up --build
```

2. **Access the application:**
   - Open http://localhost:5000

3. **Stop the application:**
```bash
docker-compose down
```

### Docker Configuration Files

**docker-compose.yml:**
- Configures both frontend and backend
- Includes volume mounts for data persistence
- Sets up networking between services

**Dockerfile.backend:**
- Multi-stage build for optimization
- Installs Python dependencies
- Copies application code and models

### Deploy to Docker-Compatible Platforms

Once tested locally, deploy to:
- **Railway.app** (Free tier available)
- **Render** (Free tier available)
- **DigitalOcean App Platform**
- **AWS ECS/Fargate**
- **Google Cloud Run**

---

## Cloud Platform Deployment

### Render.com (Recommended)

**✅ Pros:** Free tier, auto-deploy, custom domains
**⚠️ Cons:** Free tier sleeps after 15 min inactivity

**Steps:**

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
   - Render detects `render.yaml` and deploys automatically

3. **Configure Environment Variables:**
   - `FLASK_ENV=production`
   - `FLASK_APP=src/api.py`
   - `MAX_CONTENT_LENGTH=16777216`

4. **Access your app:**
   - Render provides URL: `https://medical-interpreter.onrender.com`

5. **Monitor deployment:**
   - Check logs in Render dashboard
   - Test health endpoint: `https://your-app.onrender.com/health`

### Railway.app

**✅ Pros:** $5 free credit monthly, simple deployment
**⚠️ Cons:** Credit-based (may need upgrade)

**Steps:**

1. **Push code to GitHub** (see Render step 1)

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Sign up/Login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway detects Dockerfile and deploys automatically

3. **Configure (optional):**
   - Add environment variables in Railway dashboard
   - Railway auto-assigns a domain

4. **Railway CLI (Alternative):**
```bash
npm install -g @railway/cli
railway login
railway up
```

### Heroku

```bash
heroku login
heroku create medical-interpreter
git push heroku main
```

---

## Split Deployment

Deploy backend and frontend separately for better scalability.

### Backend on Render + Frontend on Vercel

**Part 1: Deploy Backend to Render**

1. **Prepare Backend:**
```bash
cd medical_interpreter
git add .
git commit -m "Backend ready for Render"
git push
```

2. **Deploy on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `medical_interpreter` repository

3. **Configure Service:**
   ```
   Name: medical-interpreter-backend
   Region: Oregon (US West)
   Branch: main
   Environment: Docker
   Dockerfile Path: Dockerfile.backend
   Plan: Free
   ```

4. **Add Environment Variables:**
   ```
   FLASK_ENV=production
   FLASK_APP=src/api.py
   MAX_CONTENT_LENGTH=16777216
   CORS_ORIGINS=https://your-app.vercel.app
   ```

5. **Get Backend URL:**
   - Copy: `https://medical-interpreter-backend.onrender.com`

6. **Test Backend:**
```bash
curl https://YOUR_BACKEND_URL.onrender.com/health
```

**Part 2: Deploy Frontend to Vercel**

1. **Configure Frontend:**
```bash
cd frontend-react
cp .env.example .env.production
```

2. **Edit `.env.production`:**
```env
VITE_API_URL=https://YOUR_BACKEND_URL.onrender.com
```

3. **Commit and push:**
```bash
cd ..
git add .
git commit -m "Frontend ready for Vercel"
git push
```

4. **Deploy on Vercel:**
   - Go to https://vercel.com
   - Sign up/Login with GitHub
   - Click "Add New..." → "Project"
   - Import your repository

5. **Configure Project:**
   ```
   Framework Preset: Vite
   Root Directory: frontend-react
   Build Command: npm run build
   Output Directory: dist
   ```

6. **Add Environment Variable:**
   ```
   VITE_API_URL=https://YOUR_BACKEND_URL.onrender.com
   ```

7. **Deploy and Get URL:**
   - Vercel builds and deploys
   - Get URL: `https://medical-interpreter.vercel.app`

8. **Update Backend CORS:**
   - Go back to Render dashboard
   - Update `CORS_ORIGINS` to include your Vercel URL
   - Save changes (auto-redeploys)

### Netlify (Alternative to Vercel)

```bash
cd frontend-react
npm run build
# Drag and drop 'dist' folder to netlify.com
```

---

## Environment Configuration

### Development (.env.example)

```bash
FLASK_ENV=development
FLASK_APP=src/api.py
FLASK_DEBUG=1
SECRET_KEY=dev-secret-key
MAX_CONTENT_LENGTH=16777216
```

### Production

Set these in your hosting platform:

```bash
FLASK_ENV=production
FLASK_APP=src/api.py
SECRET_KEY=your-secure-random-key-here
MAX_CONTENT_LENGTH=16777216
CORS_ORIGINS=https://your-frontend-domain.com
```

**Generate Secure SECRET_KEY:**
```python
import secrets
print(secrets.token_hex(32))
```

---

## Pre-Deployment Checklist

### Essential Checks

- [ ] Code committed to Git repository
- [ ] Test application locally with Docker
- [ ] All models present in `models/` directory
- [ ] `requirements.txt` is complete
- [ ] Environment variables configured
- [ ] CORS origins updated for production
- [ ] Secure `SECRET_KEY` set (not dev key)
- [ ] `.env` and secrets not committed to Git
- [ ] Frontend built successfully
- [ ] API endpoints tested locally

### Testing Script

Run the pre-deployment test:
```bash
./test_deployment.sh
```

This checks:
- ✓ Required files exist
- ✓ Dependencies are installable
- ✓ Models are present
- ✓ Docker builds successfully
- ✓ API responds correctly

### Manual Testing

1. **Test file upload:**
   - Upload a sample PDF
   - Verify extraction works
   - Check interpretation generated

2. **Test ML models:**
   - Verify predictions load
   - Check model accuracy endpoint
   - Test advanced features (if enabled)

3. **Test API endpoints:**
   - `/health` - Health check
   - `/api/upload` - File upload
   - `/api/interpret` - Interpretation
   - `/api/predict` - ML predictions

4. **Check logs:**
   - Look for errors or warnings
   - Verify all services starting correctly

---

## Troubleshooting

### Common Issues

#### Docker Build Fails

**Problem:** Docker build fails or times out

**Solutions:**
```bash
# Check Docker is running
docker --version

# Clean Docker system
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check Dockerfile syntax
docker build -f Dockerfile.backend .
```

#### Frontend Can't Connect to Backend

**Problem:** CORS errors or connection refused

**Solutions:**
- Check API URL in frontend code
- Verify CORS settings in `api.py`
- Ensure backend is running and accessible
- Check network/firewall settings
- Test backend health endpoint directly

#### Models Not Loading

**Problem:** ML models fail to load or predictions fail

**Solutions:**
- Ensure `models/` directory exists
- Check model files are in Docker image
- Verify file paths in code match model locations
- Check file permissions
- Look for model loading errors in logs

#### Out of Memory on Free Tier

**Problem:** Application crashes or won't start

**Solutions:**
- ML models may be too large for free tier
- Consider upgrading to paid tier
- Optimize model size (quantization, pruning)
- Use smaller model variants
- Split deployment (backend on paid tier)

#### Slow Performance / Cold Starts

**Problem:** First request takes 30-60 seconds

**Solutions:**
- Free tiers sleep after inactivity (Render, Heroku)
- Keep-alive service (scheduled pings)
- Upgrade to paid tier
- Use platforms without cold starts (Railway)

#### Port Already in Use

**Problem:** Local Docker fails to start

**Solutions:**
```bash
# Find process using port 5000
lsof -ti:5000

# Kill the process
kill -9 $(lsof -ti:5000)

# Or change port in docker-compose.yml
```

#### Environment Variables Not Loading

**Problem:** Config not applied in production

**Solutions:**
- Verify variables set in platform dashboard
- Check variable names match exactly
- Restart service after updating
- Check logs for loading confirmation
- Ensure no typos in variable names

### Platform-Specific Issues

#### Render
- **Sleeps after 15 min:** Free tier limitation
- **Build timeout:** Split into smaller images
- **Disk space:** Models too large for free tier

#### Railway
- **Credit exhausted:** Monitor usage, upgrade plan
- **Build fails:** Check logs for specific errors

#### Vercel (Frontend Only)
- **Cannot host Python:** Backend must be separate
- **Build fails:** Check Node version compatibility
- **Env vars not working:** Must start with `VITE_`

### Getting Help

1. **Check platform logs:**
   - Render: Dashboard → Service → Logs
   - Railway: Project → Deployments → Logs
   - Vercel: Project → Deployments → Build Logs

2. **Test locally first:**
```bash
docker-compose up --build
```

3. **Verify environment:**
```bash
# Check all required files
ls -la models/
ls -la frontend-react/dist/
```

4. **Enable debug mode (temporarily):**
```bash
FLASK_DEBUG=1
```

---

## Quick Reference Commands

### Docker
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose build --no-cache
```

### Git
```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USER/REPO.git
git push -u origin main

# Updates
git add .
git commit -m "Update"
git push
```

### Railway CLI
```bash
npm install -g @railway/cli
railway login
railway up
railway logs
```

### Heroku
```bash
heroku login
heroku create medical-interpreter
git push heroku main
heroku logs --tail
```

---

## Next Steps After Deployment

1. ✅ **Test thoroughly:**
   - Upload sample PDFs
   - Verify all features work
   - Check mobile responsiveness

2. ✅ **Monitor performance:**
   - Set up error tracking (Sentry)
   - Monitor uptime (UptimeRobot)
   - Check response times

3. ✅ **Set up custom domain** (optional):
   - Purchase domain
   - Configure DNS
   - Enable HTTPS

4. ✅ **Configure backups:**
   - Database backups
   - Model versioning
   - Configuration backups

5. ✅ **Set up CI/CD** (optional):
   - GitHub Actions
   - Auto-deploy on push
   - Run tests before deploy

---

## Support

- **Documentation:** See other guides in [docs/](.)
- **Issues:** Check platform documentation
- **Testing:** Always test locally with Docker first
- **Logs:** Platform dashboard has detailed logs

---

## Recommended Approach for Beginners

1. **Start with Render.com** (Option 2 in Quick Start)
2. **Test with free tier** 
3. **Monitor usage and performance**
4. **Upgrade if needed** for production use
5. **Consider split deployment** for better scalability

**Total time:** 10-15 minutes from code to live URL ✨
