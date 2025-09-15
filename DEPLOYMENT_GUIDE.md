# 🚀 AgriAI Platform - Netlify Deployment Guide

## Overview
Your AgriAI platform is now ready for deployment to Netlify with integrated backend functions and Supabase database connection.

## 📁 What's Included

### Frontend (React + Vite + TypeScript)
- **Dashboard** with real NDVI satellite analysis
- **Fields Management** with geolocation support
- **Device Monitoring** with real-time status
- **Alerts System** with severity levels
- **Authentication Context** ready for integration

### Backend (Netlify Functions + Python)
- **NDVI Analysis API** (`/.netlify/functions/api/ndvi`)
- **Dashboard Stats API** (`/.netlify/functions/api/dashboard-stats`)
- **Fields Management API** (`/.netlify/functions/api/fields`)
- **Devices API** (`/.netlify/functions/api/devices`)
- **Alerts API** (`/.netlify/functions/api/alerts`)

### Database (Supabase PostgreSQL)
- **Connected** to your provided Supabase instance
- **Tables** created for users, fields, devices, NDVI analysis, alerts
- **Real-time** data storage and retrieval

## 🔧 Pre-Deployment Checklist

### ✅ Files Ready
- [ ] `netlify.toml` - Deployment configuration
- [ ] `netlify/functions/api.py` - Backend API functions
- [ ] `dist/` folder - Built frontend (run `npm run build`)
- [ ] Environment variables configured

### ✅ Database Setup
- [ ] Supabase database URL: `postgresql://postgres.wqoykxlithrqewgyxksr:[Gans356air]@aws-1-ap-south-1.pooler.supabase.com:6543/postgres`
- [ ] Tables auto-created on first API call
- [ ] Connection tested

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Commit all changes:**
```bash
git add .
git commit -m "feat: ready for Netlify deployment with integrated backend"
git push origin feature/agri-ai-platform-integration
```

2. **Merge to main (if ready):**
```bash
git checkout main
git merge feature/agri-ai-platform-integration
git push origin main
```

### Step 2: Deploy to Netlify

#### Option A: Connect GitHub Repository (Recommended)

1. **Go to Netlify Dashboard:**
   - Visit: https://app.netlify.com
   - Sign in with your account

2. **Import from Git:**
   - Click "New site from Git"
   - Choose "GitHub"
   - Select your repository: `crop-scan-nexus-main`

3. **Configure Build Settings:**
   ```
   Branch to deploy: main (or feature/agri-ai-platform-integration)
   Build command: npm run build
   Publish directory: dist
   Functions directory: netlify/functions
   ```

4. **Set Environment Variables:**
   ```
   VITE_API_BASE = (leave empty)
   VITE_APP_NAME = AgriAI Platform
   VITE_ENABLE_REAL_ANALYSIS = true
   VITE_DEBUG_MODE = false
   ```

5. **Deploy:**
   - Click "Deploy site"
   - Wait for build to complete (~2-3 minutes)

#### Option B: Manual Deploy

1. **Build locally:**
```bash
npm run build
```

2. **Drag and Drop:**
   - Go to Netlify Dashboard
   - Drag the `dist` folder to the deployment area
   - Note: Manual deploys don't support functions

### Step 3: Configure Custom Domain (Optional)

1. **In Netlify Dashboard:**
   - Go to Site Settings > Domain management
   - Add custom domain: `yourdomain.com`
   - Follow DNS configuration instructions

### Step 4: Test Deployment

1. **Visit your deployed site:**
   - URL will be: `https://random-name.netlify.app`
   - Test all functionalities:
     - ✅ Dashboard loads with real stats
     - ✅ NDVI analysis works with coordinates
     - ✅ Fields page shows data
     - ✅ Alerts display correctly
     - ✅ Geolocation "Use My Location" works

2. **Test API Endpoints:**
   ```
   GET https://your-site.netlify.app/.netlify/functions/api/health
   GET https://your-site.netlify.app/.netlify/functions/api/dashboard-stats
   POST https://your-site.netlify.app/.netlify/functions/api/ndvi
   ```

## 🔧 API Endpoints

### Health Check
```
GET /.netlify/functions/api/health
Response: {"status": "healthy", "timestamp": "...", "environment": "netlify"}
```

### NDVI Analysis
```
POST /.netlify/functions/api/ndvi
Body: {"latitude": 12.9716, "longitude": 77.5946, "field_id": 1}
Response: {"success": true, "image": "base64...", "statistics": {...}}
```

### Dashboard Stats
```
GET /.netlify/functions/api/dashboard-stats
Response: {"total_fields": 12, "total_devices": 45, "active_alerts": 8, "recent_analyses": 24}
```

### Fields Management
```
GET /.netlify/functions/api/fields
POST /.netlify/functions/api/fields (with field data)
```

### Devices
```
GET /.netlify/functions/api/devices
POST /.netlify/functions/api/devices (with device data)
```

### Alerts
```
GET /.netlify/functions/api/alerts
Response: {"alerts": [...]}
```

## 🗄️ Database Integration

### Supabase Connection
- **Database URL**: Already configured in functions
- **Auto-tables**: Created automatically on first function call
- **Connection**: Secure via environment variables

### Tables Structure
```sql
- users (authentication)
- fields (farm field management)
- devices (IoT device tracking)
- ndvi_analysis (satellite analysis results)
- alerts (system alerts and notifications)
- device_readings (sensor data)
```

## 🔧 Post-Deployment Configuration

### 1. Environment Variables (Production)
```bash
# In Netlify Dashboard > Site Settings > Environment Variables
VITE_API_BASE=
VITE_APP_NAME=AgriAI Platform
VITE_ENABLE_REAL_ANALYSIS=true
VITE_DEBUG_MODE=false
DATABASE_URL=postgresql://postgres.wqoykxlithrqewgyxksr:[Gans356air]@aws-1-ap-south-1.pooler.supabase.com:6543/postgres
```

### 2. Custom Headers (Security)
Add to `netlify.toml`:
```toml
[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
```

### 3. Performance Optimization
```toml
[build.processing]
  skip_processing = false
[build.processing.css]
  bundle = true
  minify = true
[build.processing.js]
  bundle = true
  minify = true
[build.processing.html]
  pretty_urls = true
```

## 🐛 Troubleshooting

### Common Issues

1. **Functions not working:**
   - Check `netlify/functions/api.py` exists
   - Verify functions directory in `netlify.toml`
   - Check function logs in Netlify Dashboard

2. **Database connection failed:**
   - Verify Supabase URL in environment variables
   - Check network connectivity
   - Ensure database credentials are correct

3. **Build fails:**
   - Run `npm run build` locally first
   - Check for TypeScript errors
   - Verify all dependencies are installed

4. **API calls failing:**
   - Check CORS settings in functions
   - Verify API base URL configuration
   - Test endpoints individually

### Debug Steps

1. **Check Function Logs:**
   - Netlify Dashboard > Functions > View logs
   - Look for Python errors or database connection issues

2. **Test Locally:**
   ```bash
   npm run build
   npm run preview
   ```

3. **Check Network Tab:**
   - Browser DevTools > Network
   - Look for failed API calls
   - Check response status codes

## 🎯 Success Metrics

### ✅ Deployment Successful If:
- [ ] Site loads without errors
- [ ] Dashboard shows real data (not "Loading...")
- [ ] NDVI analysis generates coordinates-based results
- [ ] Geolocation works on mobile devices
- [ ] All API endpoints return valid JSON
- [ ] Database operations (fields, devices, alerts) work
- [ ] No console errors related to API calls

### 📊 Performance Targets:
- **Load Time**: < 3 seconds
- **API Response**: < 2 seconds
- **NDVI Analysis**: < 10 seconds
- **Lighthouse Score**: > 90

## 🔄 Updates and Maintenance

### Code Updates:
1. Make changes locally
2. Test with `npm run build`
3. Commit and push to GitHub
4. Netlify auto-deploys from connected branch

### Database Updates:
- Tables auto-update via migration functions
- Monitor Supabase dashboard for usage
- Scale database as needed

### Function Updates:
- Edit `netlify/functions/api.py`
- Deploy triggers automatic function reload
- Monitor function usage in Netlify dashboard

---

## 🎉 Your AgriAI Platform is Ready!

**Frontend + Backend + Database** integrated and deployed on Netlify with Supabase PostgreSQL.

**Live URL**: `https://your-site-name.netlify.app`
**Admin**: Netlify Dashboard for monitoring and logs
**Database**: Supabase dashboard for data management

Enjoy your fully functional AgriAI crop monitoring platform! 🌱📊