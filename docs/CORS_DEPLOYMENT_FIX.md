# CORS & Deployment Fix Guide

## Problem Summary

Your deployment was experiencing CORS (Cross-Origin Resource Sharing) errors:
- Frontend at `https://ai-report-analyser.vercel.app` couldn't communicate with backend at `https://medical-interpreter-backend.onrender.com`
- Errors: "No 'Access-Control-Allow-Origin' header is present" 
- Health check and API endpoints returning 500 errors

## Root Causes

1. **Incomplete CORS Configuration**: The backend only allowed specific localhost origins, not the production Vercel domain
2. **Missing Environment Variable**: `CORS_ORIGINS` wasn't set in the Render deployment
3. **Route-Specific CORS**: CORS was only configured for `/api/*` routes, not `/health`

## Solutions Implemented

### 1. Updated Backend CORS Configuration (`src/api.py`)

**Changed from:**
```python
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')
CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=True)
```

**Changed to:**
```python
default_origins = 'http://localhost:5173,http://localhost:3000,https://ai-report-analyser.vercel.app,https://*.vercel.app'
cors_origins = os.environ.get('CORS_ORIGINS', default_origins).split(',')
cors_origins = [origin.strip() for origin in cors_origins]
CORS(app, 
     resources={
         r"/*": {"origins": cors_origins}
     },
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
```

**Key Changes:**
- Added `https://ai-report-analyser.vercel.app` to default origins
- Added wildcard support for Vercel preview deployments `https://*.vercel.app`
- Changed from `/api/*` to `/*` to enable CORS for all routes (including `/health`)
- Explicitly added HTTP methods and headers
- Stripped whitespace from origins for robustness

### 2. Updated Render Deployment Configuration

**`render.backend.yaml`** - Updated environment variables:
```yaml
envVars:
  - key: CORS_ORIGINS
    value: https://ai-report-analyser.vercel.app,https://*.vercel.app,http://localhost:5173,http://localhost:3000
```

**`render.yaml`** - Added missing CORS_ORIGINS variable to main deployment

### 3. Frontend Configuration (Already Correct)

The frontend is properly configured to use the environment variable:
- **Development** (`.env`): `VITE_API_URL=http://localhost:8000`
- **Production** (`.env.production`): `VITE_API_URL=https://medical-interpreter-backend.onrender.com`

## Deployment Steps

### For Vercel Frontend (Already Deployed)
No changes needed - frontend is already configured correctly with the API URL.

### For Render Backend (ACTION REQUIRED)

1. **Set Environment Variable in Render Dashboard:**
   - Go to your backend service settings
   - Find the "Environment" section
   - Add/Update the variable:
     ```
     CORS_ORIGINS=https://ai-report-analyser.vercel.app,https://*.vercel.app,http://localhost:5173,http://localhost:3000
     ```
   - Save and the service will redeploy automatically

2. **Or Use Blueprint Deployment:**
   - Use `render.backend.yaml` which now includes the correct CORS_ORIGINS
   - This ensures consistent configuration across deployments

3. **Verify After Deployment:**
   - Wait 2-3 minutes for the service to redeploy
   - Test the health endpoint: `curl https://medical-interpreter-backend.onrender.com/health`
   - Should return: `{"status":"healthy","service":"Medical Report Interpretation API","version":"1.0.0"}`

## Testing the Fix

### 1. Test Health Endpoint (Browser Console)
```javascript
fetch('https://medical-interpreter-backend.onrender.com/health')
  .then(res => res.json())
  .then(data => console.log('✅ Health check passed:', data))
  .catch(err => console.error('❌ Error:', err))
```

### 2. Test API Call with CORS Headers
```javascript
const formData = new FormData()
formData.append('file', pdfFile)

fetch('https://medical-interpreter-backend.onrender.com/api/interpret', {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => console.log('✅ Interpretation:', data))
  .catch(err => console.error('❌ Error:', err))
```

### 3. Check CORS Headers in Network Tab
- Open Developer Tools → Network tab
- Make a request to the backend
- Check the response headers for:
  - `Access-Control-Allow-Origin: https://ai-report-analyser.vercel.app`
  - `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS`

## Local Development

For local development, the defaults support:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (Alternative dev port)
- `http://localhost:8000` (Backend dev server)

No environment variables needed locally - just run:
```bash
python src/api.py  # Backend
npm run dev        # Frontend (from frontend-react/)
```

## Production Considerations

### Dynamic CORS for Future Deployments
If you deploy to a different Vercel URL or another platform, update the `CORS_ORIGINS` environment variable:
```
https://your-domain.vercel.app,https://*.vercel.app,<any-other-origins>
```

### Security Notes
- In production, remove `http://localhost:*` origins
- Use specific domain names instead of wildcards when possible
- The wildcard `https://*.vercel.app` allows all Vercel preview deployments - consider restricting if needed

## Troubleshooting

### Still Getting CORS Errors?

1. **Check environment variable is set:**
   ```bash
   # In Render dashboard, verify CORS_ORIGINS is visible
   ```

2. **Restart the service:**
   - Go to Render dashboard → select service → click "Manual Deploy"

3. **Check backend logs:**
   - Render dashboard → Logs tab
   - Look for any Flask errors

4. **Verify frontend API URL:**
   - Check `frontend-react/.env.production` has correct backend URL
   - Run frontend build: `npm run build` in `frontend-react/`

5. **Clear browser cache:**
   - Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)
   - Clear site data: DevTools → Application → Clear Site Data

### If /api/model-comparison Returns 500

1. Check that ML model files exist in `models/` directory
2. Check Render logs for specific errors
3. Verify `model_metadata.json` is accessible
4. If using new datasets, retrain models: `python run_training_workflow.py`

## Files Modified

1. `src/api.py` - Enhanced CORS configuration
2. `render.backend.yaml` - Added CORS_ORIGINS environment variable
3. `render.yaml` - Added CORS_ORIGINS to deployment config

## Next Steps

1. ✅ Deploy backend changes to Render
2. ✅ Set CORS_ORIGINS environment variable in Render dashboard
3. ✅ Wait for service to redeploy (2-3 minutes)
4. ✅ Test endpoints from frontend
5. ✅ Monitor Render logs for any issues
