# 📝 Deployment Checklist - Render + Vercel

## ✅ Step-by-Step Checklist

### Before You Start
- [ ] Code is committed to GitHub
- [ ] You have a GitHub account
- [ ] Backend tests pass locally

---

## 🔧 PART 1: Backend on Render

### 1. Push to GitHub
```bash
cd /Users/anuprabh/Desktop/BTP/medical_interpreter
git add .
git commit -m "Ready for deployment"
git push
```

### 2. Render Setup
- [ ] Go to https://render.com
- [ ] Sign up/Login with GitHub
- [ ] Click "New +" → "Web Service"
- [ ] Connect GitHub repo

### 3. Configure Render
- [ ] Name: `medical-interpreter-backend`
- [ ] Environment: Docker
- [ ] Branch: main
- [ ] Dockerfile path: `Dockerfile.backend`
- [ ] Plan: Free

### 4. Environment Variables (Render)
Add these in Render dashboard:
- [ ] `FLASK_ENV` = `production`
- [ ] `FLASK_APP` = `src/api.py`
- [ ] `MAX_CONTENT_LENGTH` = `16777216`
- [ ] `CORS_ORIGINS` = (leave blank for now, update later)

### 5. Deploy & Wait
- [ ] Click "Create Web Service"
- [ ] Wait 5-10 minutes for build
- [ ] Check logs for "Service is live"

### 6. Save Backend URL
- [ ] Copy URL: `https://______.onrender.com`
- [ ] Test: `curl https://YOUR_URL.onrender.com/health`
- [ ] Should return: `{"status":"healthy"}`

✅ **Backend URL**: _________________________________

---

## 🎨 PART 2: Frontend on Vercel

### 7. Update Environment
```bash
cd frontend-react
cp .env.example .env.production
```

Edit `.env.production`:
- [ ] Set `VITE_API_URL` = (your Render backend URL)

```bash
cd ..
git add .
git commit -m "Configure for Vercel"
git push
```

### 8. Vercel Setup
- [ ] Go to https://vercel.com
- [ ] Sign up/Login with GitHub
- [ ] Click "Add New..." → "Project"
- [ ] Import your repo

### 9. Configure Vercel
- [ ] Framework: Vite
- [ ] Root Directory: `frontend-react`
- [ ] Build Command: `npm run build`
- [ ] Output Directory: `dist`

### 10. Environment Variables (Vercel)
- [ ] Name: `VITE_API_URL`
- [ ] Value: (your Render backend URL from step 6)

### 11. Deploy & Wait
- [ ] Click "Deploy"
- [ ] Wait 3-5 minutes
- [ ] Check for "Deployment Ready"

### 12. Save Frontend URL
- [ ] Copy URL: `https://______.vercel.app`

✅ **Frontend URL**: _________________________________

---

## 🔄 PART 3: Connect Them

### 13. Update CORS on Render
- [ ] Go back to Render dashboard
- [ ] Select your backend service
- [ ] Go to "Environment" tab
- [ ] Update `CORS_ORIGINS` to your Vercel URL
- [ ] Format: `https://your-app.vercel.app,https://your-app-*.vercel.app`
- [ ] Save (Render will auto-redeploy)

---

## 🧪 PART 4: Test

### 14. Test Backend
```bash
curl https://YOUR_BACKEND_URL.onrender.com/health
```
- [ ] Returns: `{"status":"healthy"}`

### 15. Test Frontend
- [ ] Open your Vercel URL in browser
- [ ] Page loads without errors
- [ ] Press F12 → check Console for errors

### 16. Test Full Flow
- [ ] Upload a test PDF
- [ ] File uploads successfully
- [ ] Interpretation appears
- [ ] No CORS errors in console

---

## 🎉 Done!

### Your Live URLs:
- **Backend**: https://_______________.onrender.com
- **Frontend**: https://_______________.vercel.app

### Share with:
- [ ] Save URLs in a safe place
- [ ] Share frontend URL with users
- [ ] Test on different devices

---

## ⚠️ Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CORS error | Update `CORS_ORIGINS` in Render with exact Vercel URL |
| 404 on backend | Check Dockerfile path is `Dockerfile.backend` |
| Slow first load | Free tier sleeps - first request takes 30s |
| Build fails | Check logs, verify all files committed |
| Frontend blank | Check environment variables in Vercel |

---

## 🔄 To Update Later:

**Backend:**
```bash
git add .
git commit -m "Update"
git push
# Render auto-deploys
```

**Frontend:**
```bash
git add .
git commit -m "Update"
git push
# Vercel auto-deploys
```

---

**Need help? See: [DEPLOY_RENDER_VERCEL.md](DEPLOY_RENDER_VERCEL.md)**
