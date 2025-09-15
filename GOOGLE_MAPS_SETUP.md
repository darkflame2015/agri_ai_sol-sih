# 🗺️ Google Maps API Setup Guide

## Overview
The AgriAI platform now includes Google Maps integration for interactive satellite mapping and location selection for NDVI analysis.

## 🔑 Google Maps API Key Setup

### Step 1: Get Google Maps API Key

1. **Go to Google Cloud Console:**
   - Visit: https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create a new project (if needed):**
   - Click "New Project"
   - Name: "AgriAI Platform"
   - Click "Create"

3. **Enable APIs:**
   - Navigate to "APIs & Services" > "Library"
   - Search and enable these APIs:
     - ✅ **Maps JavaScript API**
     - ✅ **Places API** (optional, for location search)
     - ✅ **Geocoding API** (optional, for address lookup)

4. **Create API Key:**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key
   - Click "Restrict Key" for security

5. **Configure API Key Restrictions:**
   ```
   Application restrictions:
   - HTTP referrers (websites)
   - Add your domains:
     - localhost:*
     - your-domain.com/*
     - *.netlify.app/*
   
   API restrictions:
   - Restrict key to specific APIs
   - Select: Maps JavaScript API, Places API, Geocoding API
   ```

### Step 2: Add API Key to Environment

1. **Local Development (.env):**
   ```bash
   VITE_GOOGLE_MAPS_API_KEY=your_actual_api_key_here
   ```

2. **Netlify Deployment:**
   - Go to Netlify Dashboard > Site Settings > Environment Variables
   - Add: `VITE_GOOGLE_MAPS_API_KEY` = `your_actual_api_key_here`

## 🌟 Features Enabled

### Interactive Satellite Map
- **Satellite imagery** view by default
- **Click to select** coordinates for NDVI analysis
- **Real-time marker** updates
- **Multiple map types**: Satellite, Hybrid, Terrain, Roadmap

### Enhanced Location Input
- **Visual coordinate selection** via map clicks
- **Manual coordinate entry** (lat/lng)
- **"Use My Location"** GPS button
- **Real-time map center** updates

### Professional Mapping
- **Google Maps quality** satellite imagery
- **Zoom controls** and full-screen mode
- **Responsive design** for all screen sizes
- **Error handling** with fallback to coordinate mode

## 🔧 Technical Implementation

### Map Initialization
```typescript
// Automatically loads Google Maps API
// Falls back to coordinate input if API unavailable
// Handles errors gracefully
```

### Click Events
```typescript
// Map clicks update coordinate inputs
// Marker shows selected location
// Real-time coordinate display
```

### Integration with NDVI Analysis
```typescript
// Selected coordinates feed directly into analysis
// Visual feedback on analysis location
// Seamless user experience
```

## 🚨 Troubleshooting

### Common Issues

1. **Map not loading:**
   - Check API key is set: `VITE_GOOGLE_MAPS_API_KEY`
   - Verify API key restrictions allow your domain
   - Check browser console for errors

2. **"This page can't load Google Maps correctly":**
   - API key missing or invalid
   - Check billing account is active
   - Verify APIs are enabled

3. **Coordinate input fallback:**
   - Normal behavior when Maps API unavailable
   - Users can still enter coordinates manually
   - All NDVI analysis features remain functional

### Debug Steps

1. **Check console for errors:**
   ```javascript
   // Open Browser DevTools > Console
   // Look for Google Maps related errors
   ```

2. **Verify API key:**
   ```bash
   # In browser console
   console.log(import.meta.env.VITE_GOOGLE_MAPS_API_KEY)
   ```

3. **Test API key manually:**
   ```
   https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY
   ```

## 💰 Cost Considerations

### Google Maps Pricing (as of 2024)
- **Maps JavaScript API**: $7 per 1,000 requests
- **First 28,000 requests/month**: FREE
- **Typical usage**: ~100-500 requests/day for small farms

### Cost Optimization
- **API key restrictions** prevent abuse
- **Caching** reduces redundant requests
- **Fallback mode** works without API calls
- **Free tier** sufficient for most users

## 🔒 Security Best Practices

### API Key Protection
1. **Never commit API keys** to public repositories
2. **Use environment variables** for all environments
3. **Restrict API key** to specific domains/APIs
4. **Monitor usage** in Google Cloud Console
5. **Rotate keys** periodically

### Domain Restrictions
```
Allowed domains:
- localhost:* (development)
- your-production-domain.com/*
- *.netlify.app/* (if using Netlify)
```

## 🎯 Benefits for Users

### Enhanced User Experience
- **Visual location selection** vs manual coordinate entry
- **Satellite imagery context** for crop analysis
- **Professional mapping interface**
- **Mobile-friendly** touch controls

### Improved Accuracy
- **Precise coordinate selection** via map clicks
- **Visual verification** of analysis location
- **Satellite imagery** shows actual field boundaries
- **GPS integration** for field work

### Professional Features
- **Multiple map layers** (satellite, hybrid, terrain)
- **Zoom controls** for detailed location selection
- **Full-screen mode** for detailed mapping
- **Responsive design** for all devices

---

## 🚀 Next Steps

1. **Get Google Maps API Key** (15 minutes)
2. **Add to environment variables** (5 minutes)
3. **Deploy to Netlify** (automatic)
4. **Test on production** (5 minutes)

Your AgriAI platform will then have professional-grade mapping capabilities! 🌍🛰️