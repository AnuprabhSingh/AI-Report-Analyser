# 🚀 Split Deployment Guide: Render + Vercel

## Backend on Render + Frontend on Vercel

---

## 📋 Overview

- **Backend (API)**: Deploy to Render.com (Free tier)
- **Frontend (React)**: Deploy to Vercel (Free tier)
- **Connection**: Frontend calls backend API via HTTPS

---

## Part 1: Deploy Backend to Render (15 minutes)

### Step 1: Prepare Backend

Your backend is already configured! Just make sure you've committed everything:

```bash
cd /Users/anuprabh/Desktop/BTP/medical_interpreter
git add .
git commit -m "Backend ready for Render"
git push
```

### Step 2: Deploy on Render

1. **Go to Render**: https://render.com
2. **Sign up/Login** with your GitHub account
3. **Create New Web Service**:
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Select `medical_interpreter` repository

4. **Configure Service**:
   ```
   Name: medical-interpreter-backend
   Region: Oregon (US West)
   Branch: main
   Root Directory: (leave blank)
   Environment: Docker
   Docker Command: (leave blank - uses Dockerfile.backend)
   Plan: Free
   ```

5. **Advanced Settings** - Add Environment Variables:
   ```
   FLASK_ENV=production
   FLASK_APP=src/api.py
   MAX_CONTENT_LENGTH=16777216
   CORS_ORIGINS=https://your-app.vercel.app
   ```
   
   ⚠️ **Note**: We'll update `CORS_ORIGINS` with your actual Vercel URL in Step 10

6. **Dockerfile Path**:
   - Click "Advanced"
   - Set **Dockerfile Path**: `Dockerfile.backend`

7. **Click "Create Web Service"**

### Step 3: Wait for Build (5-10 minutes)

Watch the logs. You should see:
```
==> Building...
==> Deploying...
==> Service is live 🎉
```

### Step 4: Get Your Backend URL

Once deployed, Render provides a URL like:
```
https://medical-interpreter-backend.onrender.com
```

**📝 COPY THIS URL - You'll need it for frontend!**

### Step 5: Test Backend

```bash
# Test health endpoint
curl https://YOUR_BACKEND_URL.onrender.com/health

# Should return:
# {"status":"healthy","service":"Medical Report Interpretation API","version":"1.0.0"}
```

---

## Part 2: Deploy Frontend to Vercel (10 minutes)

### Step 6: Prepare Frontend

1. **Create environment file for production**:
```bash
cd frontend-react
cp .env.example .env.production
```

2. **Edit `.env.production`** and add your Render backend URL:
```env
VITE_API_URL=https://YOUR_BACKEND_URL.onrender.com
```

Replace `YOUR_BACKEND_URL` with the URL from Step 4.

3. **Commit changes**:
```bash
cd ..
git add .
git commit -m "Frontend ready for Vercel"
git push
```

### Step 7: Deploy on Vercel

1. **Go to Vercel**: https://vercel.com
2. **Sign up/Login** with your GitHub account
3. **Import Project**:
   - Click **"Add New..."** → **"Project"**
   - Import your `medical_interpreter` repository

4. **Configure Project**:
   ```
   Framework Preset: Vite
   Root Directory: frontend-react
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

5. **Environment Variables**:
   Click "Environment Variables" and add:
   ```
   Name: VITE_API_URL
   Value: https://YOUR_BACKEND_URL.onrender.com
   ```
   (Use your actual Render backend URL from Step 4)

6. **Click "Deploy"**

### Step 8: Wait for Build (3-5 minutes)

Vercel will build and deploy your frontend.

### Step 9: Get Your Frontend URL

Once deployed, Vercel provides a URL like:
```
https://medical-interpreter.vercel.app
```

**🎉 Your frontend is live!**

### Step 10: Update CORS on Backend

Now that you have your Vercel URL, update the backend CORS settings:

1. **Go back to Render dashboard**
2. **Select your backend service**
3. **Go to "Environment"**
4. **Update `CORS_ORIGINS`**:
   ```
   CORS_ORIGINS=https://medical-interpreter.vercel.app,https://medical-interpreter-*.vercel.app
   ```
   (Use your actual Vercel URL)

5. **Save Changes** - Render will redeploy automatically

---

## Part 3: Test Everything (5 minutes)

### Step 11: Test Your Deployed App

1. **Open your Vercel URL** in browser
2. **Upload a test PDF**
3. **Verify interpretations appear**

### Step 12: Check Browser Console

Press F12 → Console tab. You should see:
- No CORS errors
- API calls to your Render backend
- Successful responses

---

## 🎯 Quick Reference

### Your URLs:
```
Backend:  https://YOUR_BACKEND_URL.onrender.com
Frontend: https://YOUR_FRONTEND_URL.vercel.app
```

### Test Endpoints:
```bash
# Backend health
curl https://YOUR_BACKEND_URL.onrender.com/health

# API test
curl https://YOUR_BACKEND_URL.onrender.com/api/test
```

---

## 🐛 Troubleshooting

### CORS Error in Browser?

**Problem**: Frontend can't reach backend

**Solution**:
1. Check `CORS_ORIGINS` in Render includes your Vercel URL
2. Make sure format is: `https://your-app.vercel.app` (no trailing slash)
3. Include wildcard for preview deployments: `https://your-app-*.vercel.app`

### Backend Returns 404?

**Problem**: Backend routes not found

**Solution**:
1. Check Dockerfile path in Render: should be `Dockerfile.backend`
2. Verify build logs - look for "src/api.py" being copied
3. Check environment variables are set

### Frontend Shows "Failed to Fetch"?

**Problem**: API URL not configured correctly

**Solution**:
1. Check `.env.production` has correct `VITE_API_URL`
2. Verify environment variable in Vercel dashboard
3. Redeploy frontend after changing env vars

### Render Free Tier Sleeping?

**Problem**: First request takes 30+ seconds

**Solution**: 
- Free tier sleeps after 15 min inactivity
- First request wakes it up (cold start)
- Consider upgrading for production use

### Vercel Build Fails?

**Problem**: "Build failed" error

**Solution**:
1. Check "Root Directory" is set to `frontend-react`
2. Verify `package.json` and all files exist
3. Check build logs for specific errors

---

## 🔧 Update Deployment

### Update Backend:
```bash
git add .
git commit -m "Update backend"
git push
```
Render auto-deploys on push

### Update Frontend:
```bash
git add .
git commit -m "Update frontend"
git push
```
Vercel auto-deploys on push

---

## 💰 Cost Breakdown

| Service | Free Tier | Limits |
|---------|-----------|--------|
| **Render** | ✅ Yes | 750 hours/month, sleeps after 15min |
| **Vercel** | ✅ Yes | 100GB bandwidth, unlimited deployments |
| **Total** | ✅ $0/month | Perfect for projects & demos |

---

## 🚀 Production Tips

1. **Custom Domains**:
   - Render: Add in "Settings" → "Custom Domain"
   - Vercel: Add in project "Settings" → "Domains"

2. **Environment Variables**:
   - Keep `.env` files in `.gitignore`
   - Set all env vars in platform dashboards
   - Never commit API keys or secrets

3. **Monitoring**:
   - Check Render logs for backend errors
   - Use Vercel Analytics for frontend metrics
   - Set up Render health checks

4. **SSL/HTTPS**:
   - Both platforms provide free SSL
   - Always use HTTPS URLs
   - Check certificate is valid

---

## ✅ Checklist

- [ ] Backend deployed on Render
- [ ] Backend health endpoint works
- [ ] Frontend deployed on Vercel
- [ ] Frontend loads correctly
- [ ] CORS configured properly
- [ ] File upload works
- [ ] API calls succeed
- [ ] No console errors
- [ ] URLs saved for reference

---

## 🎉 You're Done!

Your app is now live with:
- ⚡ Fast frontend (Vercel CDN)
- 🔒 Secure HTTPS
- 🆓 Free hosting
- 🚀 Auto-deployments on git push

**Share your Vercel URL with users!**

---

## 📞 Need Help?

**Render Issues**:
- Render Dashboard → Your Service → Logs
- Render Docs: https://render.com/docs

**Vercel Issues**:
- Vercel Dashboard → Your Project → Deployments → View Build Logs
- Vercel Docs: https://vercel.com/docs

**App Issues**:
- Check browser console (F12)
- Test backend health endpoint
- Verify environment variables
