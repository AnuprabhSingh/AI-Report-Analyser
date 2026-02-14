# 🚀 Quick Deploy - Medical Interpreter

## ⚡ Fastest Way (5 minutes)

### Option 1: Render.com (Recommended)

1. **Create GitHub repo and push:**
```bash
cd medical_interpreter
git add .
git commit -m "Ready for deployment"
git push
```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repo
   - Click "Apply" (Render auto-detects `render.yaml`)
   - Wait 5-10 minutes for build
   - Get your URL: `https://your-app.onrender.com`

**Done! ✅**

---

### Option 2: Railway.app

1. **Push to GitHub** (see above)

2. **Deploy:**
   - Go to [railway.app](https://railway.app)
   - "New Project" → "Deploy from GitHub"
   - Select your repo
   - Railway auto-deploys
   - Get your URL from Railway dashboard

**Done! ✅**

---

### Option 3: Docker Locally

```bash
cd medical_interpreter
docker-compose up --build
```

Open: http://localhost:5000

---

## 🔧 Test Before Deploy

```bash
./test_deployment.sh
```

---

## 📝 Environment Variables (Production)

Set these in your hosting platform:

- `FLASK_ENV=production`
- `FLASK_APP=src/api.py`
- `SECRET_KEY=random-secret-key-here`

---

## ❗ Common Issues

**Build fails?**
- Ensure all files committed to Git
- Check Dockerfile exists
- Models folder should be present

**Can't connect?**
- Check health endpoint: `https://your-app.com/health`
- Verify CORS settings
- Check logs in platform dashboard

**Slow to start?**
- Free tiers sleep after inactivity
- First request may take 30-60 seconds
- Consider upgrading for production use

---

## 📚 Full Guide

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for:
- Detailed instructions for each platform
- Environment configuration
- Troubleshooting
- Alternative deployment methods

---

## 🎯 Next Steps After Deploy

1. ✅ Test file upload functionality
2. ✅ Verify ML models load correctly  
3. ✅ Check all API endpoints work
4. ✅ Set up custom domain (optional)
5. ✅ Configure monitoring/alerts

---

## 🆘 Support

- Check platform logs for errors
- Test locally with Docker first
- Verify environment variables are set
- Ensure Git repo is up to date

**Most common issue:** Models folder missing → Make sure `models/` is committed to Git
