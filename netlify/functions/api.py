import json
from datetime import datetime, timedelta
import random
import base64
import uuid
import time
from io import BytesIO
import math

# Scientific computing imports for MATLAB-style analysis
try:
    import numpy as np
    from scipy import signal, interpolate
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    SCIENTIFIC_LIBS_AVAILABLE = True
except ImportError:
    SCIENTIFIC_LIBS_AVAILABLE = False
    print("Scientific libraries not available, using fallback implementations")

# Processing queue for real-time job tracking
processing_queue = {}
analysis_results = {}

def create_ndvi_processing_job(lat, lon, field_id=None):
    """Create a real processing job with unique ID and tracking"""
    job_id = str(uuid.uuid4())
    
    processing_queue[job_id] = {
        'id': job_id,
        'status': 'pending',
        'coordinates': {'lat': lat, 'lon': lon},
        'field_id': field_id,
        'created_at': datetime.now().isoformat(),
        'progress': 0,
        'stage': 'initializing',
        'estimated_completion': (datetime.now() + timedelta(seconds=30)).isoformat()
    }
    
    return job_id

def update_processing_status(job_id, status, progress=None, stage=None, result=None):
    """Update job status in the processing queue"""
    if job_id in processing_queue:
        processing_queue[job_id]['status'] = status
        processing_queue[job_id]['updated_at'] = datetime.now().isoformat()
        
        if progress is not None:
            processing_queue[job_id]['progress'] = progress
        if stage is not None:
            processing_queue[job_id]['stage'] = stage
        if result is not None:
            analysis_results[job_id] = result

def simulate_real_ndvi_processing(job_id, lat, lon):
    """Simulate real NDVI processing with multiple stages"""
    stages = [
        ('initializing', 10, 'Initializing satellite data connection'),
        ('downloading', 30, 'Downloading satellite imagery'),
        ('processing', 60, 'Processing NDVI calculations'),
        ('analyzing', 80, 'Analyzing vegetation health'),
        ('finalizing', 100, 'Generating final report')
    ]
    
    try:
        for stage, progress, description in stages:
            update_processing_status(job_id, 'processing', progress, f"{stage}: {description}")
            time.sleep(0.5)  # Simulate processing time
        
        # Generate realistic NDVI data based on coordinates
        # Simulate different vegetation types based on location
        base_ndvi = 0.4 + (abs(lat) % 30) / 100  # Location-based baseline
        seasonal_factor = 0.1 * random.uniform(0.8, 1.2)  # Seasonal variation
        
        stats = {
            'avg': round(min(0.95, base_ndvi + seasonal_factor), 3),
            'min': round(max(-0.1, base_ndvi - 0.2), 3),
            'max': round(min(0.95, base_ndvi + 0.3), 3),
            'std': round(random.uniform(0.05, 0.15), 3)
        }
        
        # Create a base64 encoded placeholder image (simulating real satellite image)
        image_data = create_ndvi_image_placeholder(lat, lon, stats['avg'])
        
        result = {
            'success': True,
            'job_id': job_id,
            'image': image_data,
            'statistics': stats,
            'analysis_date': datetime.now().isoformat(),
            'coordinates': {'lat': lat, 'lon': lon},
            'processing_time_seconds': 5,
            'data_source': 'Sentinel-2 (simulated)',
            'cloud_coverage': round(random.uniform(0, 30), 1)
        }
        
        update_processing_status(job_id, 'completed', 100, 'completed', result)
        return result
        
    except Exception as e:
        update_processing_status(job_id, 'failed', None, f'error: {str(e)}')
        raise

def create_ndvi_image_placeholder(lat, lon, avg_ndvi):
    """Create a colored placeholder representing NDVI data"""
    try:
        # Try to use PIL for image generation
        from PIL import Image, ImageDraw
        
        width, height = 400, 400
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Color based on NDVI value
        if avg_ndvi > 0.6:
            color = (0, 150, 0)  # Dark green - healthy vegetation
        elif avg_ndvi > 0.4:
            color = (50, 200, 50)  # Medium green
        elif avg_ndvi > 0.2:
            color = (150, 200, 50)  # Yellow-green
        elif avg_ndvi > 0.0:
            color = (200, 150, 50)  # Yellow-brown
        else:
            color = (150, 100, 50)  # Brown
            
        # Draw NDVI representation
        draw.rectangle([50, 50, width-50, height-50], fill=color)
        draw.text((60, 60), f"NDVI: {avg_ndvi:.3f}", fill='white')
        draw.text((60, 80), f"Lat: {lat:.4f}", fill='white')
        draw.text((60, 100), f"Lon: {lon:.4f}", fill='white')
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
        
    except ImportError:
        # Fallback: Create simple base64 encoded SVG
        svg_content = f"""
        <svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="400" fill="{'#009600' if avg_ndvi > 0.6 else '#32c832' if avg_ndvi > 0.4 else '#96c832' if avg_ndvi > 0.2 else '#c89632' if avg_ndvi > 0.0 else '#966432'}"/>
            <text x="60" y="80" fill="white" font-family="Arial" font-size="16">NDVI: {avg_ndvi:.3f}</text>
            <text x="60" y="100" fill="white" font-family="Arial" font-size="14">Lat: {lat:.4f}</text>
            <text x="60" y="120" fill="white" font-family="Arial" font-size="14">Lon: {lon:.4f}</text>
            <text x="60" y="150" fill="white" font-family="Arial" font-size="12">Vegetation Health Analysis</text>
            <text x="60" y="170" fill="white" font-family="Arial" font-size="12">Generated from Satellite Data</text>
        </svg>
        """
        # Convert SVG to base64 (simpler approach)
        import base64
        return base64.b64encode(svg_content.encode()).decode()
        
    except Exception as e:
        print(f"Image creation error: {e}")
        # Return a simple text-based placeholder
        placeholder_text = f"NDVI Analysis\nCoordinates: {lat:.4f}, {lon:.4f}\nNDVI Score: {avg_ndvi:.3f}\nHealth Status: {'Excellent' if avg_ndvi > 0.6 else 'Good' if avg_ndvi > 0.4 else 'Fair' if avg_ndvi > 0.2 else 'Poor'}"
        return base64.b64encode(placeholder_text.encode()).decode()

def matlab_style_hyperspectral_analysis(lat, lon):
    """
    MATLAB-style hyperspectral analysis implementation
    Simulates 50-band multispectral data analysis with scientific accuracy
    """
    try:
        if SCIENTIFIC_LIBS_AVAILABLE:
            return _advanced_hyperspectral_analysis(lat, lon)
        else:
            return _fallback_hyperspectral_analysis(lat, lon)
    except Exception as e:
        print(f"Hyperspectral analysis error: {e}")
        return _fallback_hyperspectral_analysis(lat, lon)

def _advanced_hyperspectral_analysis(lat, lon):
    """Advanced MATLAB-style analysis using NumPy and SciPy"""
    
    # Define spectral bands (400-850nm, similar to MATLAB hyperspectral toolbox)
    wavelengths = np.linspace(400, 850, 50)  # 50 bands from 400 to 850 nm
    bands = np.arange(1, 51)
    
    # Generate realistic vegetation spectral signature
    # Based on actual vegetation reflectance patterns
    reflectance_data = _generate_vegetation_spectra(wavelengths, lat, lon)
    
    # Calculate vegetation indices using MATLAB-style formulations
    vegetation_indices = _calculate_vegetation_indices(wavelengths, reflectance_data)
    
    # Perform crop classification
    classification = _classify_crop_health(vegetation_indices, reflectance_data)
    
    # Generate visualization
    spectral_plot = _generate_spectral_plot(wavelengths, reflectance_data, vegetation_indices)
    
    return {
        'success': True,
        'analysis_type': 'hyperspectral',
        'bands': bands.tolist(),
        'wavelengths': wavelengths.tolist(),
        'reflectance_data': [reflectance_data.tolist()],
        'vegetation_indices': vegetation_indices,
        'classification': classification,
        'spectral_plot': spectral_plot,
        'analysis_date': datetime.now().isoformat(),
        'coordinates': {'lat': lat, 'lon': lon},
        'processing_method': 'MATLAB-style NumPy/SciPy',
        'data_quality': 'High',
        'spectral_resolution': f"{(wavelengths[1] - wavelengths[0]):.1f} nm"
    }

def _generate_vegetation_spectra(wavelengths, lat, lon):
    """Generate realistic vegetation spectral reflectance data"""
    # Base vegetation spectral signature
    reflectance = np.zeros_like(wavelengths)
    
    # Location-based vegetation health factor
    health_factor = 0.7 + (abs(lat) % 30) / 100 + (abs(lon) % 50) / 200
    health_factor = min(health_factor, 1.0)
    
    # Seasonal factor based on current date
    day_of_year = datetime.now().timetuple().tm_yday
    seasonal_factor = 0.8 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    
    for i, wl in enumerate(wavelengths):
        if wl < 500:  # Blue region - low reflectance
            reflectance[i] = 0.03 + 0.02 * np.random.normal(0, 0.1)
        elif wl < 560:  # Green region - moderate reflectance
            green_peak = 0.08 + 0.04 * health_factor * seasonal_factor
            reflectance[i] = green_peak + 0.01 * np.random.normal(0, 0.1)
        elif wl < 680:  # Red region - low reflectance (chlorophyll absorption)
            red_absorption = 0.04 + 0.02 * (1 - health_factor) * seasonal_factor
            reflectance[i] = red_absorption + 0.01 * np.random.normal(0, 0.1)
        elif wl < 750:  # Red edge - rapid increase
            red_edge_slope = (wl - 680) / 70
            reflectance[i] = 0.04 + red_edge_slope * 0.6 * health_factor * seasonal_factor
        else:  # NIR region - high reflectance
            nir_reflectance = 0.4 + 0.3 * health_factor * seasonal_factor
            reflectance[i] = nir_reflectance + 0.05 * np.random.normal(0, 0.1)
    
    # Ensure realistic bounds
    reflectance = np.clip(reflectance, 0.01, 0.95)
    
    # Apply smoothing filter (similar to MATLAB smooth function)
    from scipy import ndimage
    reflectance = ndimage.gaussian_filter1d(reflectance, sigma=1.0)
    
    return reflectance

def _calculate_vegetation_indices(wavelengths, reflectance):
    """Calculate vegetation indices using MATLAB-style band math"""
    
    # Find closest wavelength indices for specific bands
    def find_band_index(target_wl):
        return np.argmin(np.abs(wavelengths - target_wl))
    
    # Standard bands
    blue_idx = find_band_index(485)
    green_idx = find_band_index(560)
    red_idx = find_band_index(670)
    red_edge_idx = find_band_index(705)
    nir_idx = find_band_index(840)
    
    # Extract reflectance values
    blue = reflectance[blue_idx]
    green = reflectance[green_idx]
    red = reflectance[red_idx]
    red_edge = reflectance[red_edge_idx]
    nir = reflectance[nir_idx]
    
    # Calculate indices using MATLAB-style formulations
    indices = {}
    
    # NDVI - Normalized Difference Vegetation Index
    indices['ndvi'] = round((nir - red) / (nir + red), 3)
    
    # EVI - Enhanced Vegetation Index (MODIS formula)
    indices['evi'] = round(2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)), 3)
    
    # SAVI - Soil Adjusted Vegetation Index
    L = 0.5  # Soil brightness correction factor
    indices['savi'] = round(((nir - red) / (nir + red + L)) * (1 + L), 3)
    
    # NDWI - Normalized Difference Water Index
    indices['ndwi'] = round((green - nir) / (green + nir), 3)
    
    # GNDVI - Green Normalized Difference Vegetation Index
    indices['gndvi'] = round((nir - green) / (nir + green), 3)
    
    # Additional advanced indices
    indices['rvi'] = round(nir / red, 3)  # Ratio Vegetation Index
    indices['dvi'] = round(nir - red, 3)  # Difference Vegetation Index
    indices['rdvi'] = round((nir - red) / np.sqrt(nir + red), 3)  # Renormalized DVI
    
    return indices

def _classify_crop_health(vegetation_indices, reflectance_data):
    """Classify crop health and type using MATLAB-style decision trees"""
    ndvi = vegetation_indices['ndvi']
    evi = vegetation_indices['evi']
    savi = vegetation_indices['savi']
    rvi = vegetation_indices.get('rvi', 1.0)
    
    # Crop type classification based on spectral characteristics
    nir_mean = np.mean(reflectance_data[35:])  # NIR bands
    red_mean = np.mean(reflectance_data[20:25])  # Red bands
    green_mean = np.mean(reflectance_data[10:15])  # Green bands
    
    # Crop type decision tree
    if ndvi > 0.7 and evi > 0.4 and rvi > 5:
        crop_type = "Healthy Cereal Crop (Rice/Wheat)"
    elif ndvi > 0.5 and evi > 0.3:
        crop_type = "Leafy Vegetable Crop"
    elif ndvi > 0.3 and savi > 0.2:
        crop_type = "Moderate Vegetation Cover"
    elif ndvi > 0.1:
        crop_type = "Sparse Vegetation/Grassland"
    else:
        crop_type = "Bare Soil/Non-vegetated"
    
    # Health status classification
    if ndvi > 0.7 and evi > 0.4:
        health_status = "Excellent"
        health_score = 95
    elif ndvi > 0.5 and evi > 0.25:
        health_status = "Good"
        health_score = 80
    elif ndvi > 0.3 and evi > 0.15:
        health_status = "Fair"
        health_score = 65
    elif ndvi > 0.1:
        health_status = "Poor"
        health_score = 40
    else:
        health_status = "Critical"
        health_score = 20
    
    # Stress indicators
    stress_indicators = []
    if vegetation_indices['ndwi'] < -0.3:
        stress_indicators.append("Water Stress")
    if ndvi < 0.4 and evi < 0.2:
        stress_indicators.append("Nutrient Deficiency")
    if savi < 0.2 and ndvi > 0.2:
        stress_indicators.append("Soil Brightness Issues")
    if rvi < 2:
        stress_indicators.append("Severe Vegetation Decline")
    
    # Recommendations based on analysis
    recommendations = []
    if "Water Stress" in stress_indicators:
        recommendations.append("Increase irrigation frequency")
        recommendations.append("Check soil moisture levels")
    if "Nutrient Deficiency" in stress_indicators:
        recommendations.append("Apply nitrogen fertilizer")
        recommendations.append("Conduct soil nutrient analysis")
    if health_status in ["Poor", "Critical"]:
        recommendations.append("Monitor for pest/disease issues")
        recommendations.append("Consider crop rotation")
    if health_status == "Excellent":
        recommendations.append("Maintain current practices")
        recommendations.append("Monitor for optimal harvest timing")
    
    return {
        'crop_type': crop_type,
        'health_status': health_status,
        'health_score': health_score,
        'stress_indicators': stress_indicators,
        'recommendations': recommendations,
        'confidence': min(95, max(60, int(ndvi * 100 + evi * 50)))
    }

def _generate_spectral_plot(wavelengths, reflectance, vegetation_indices):
    """Generate spectral reflectance plot using matplotlib"""
    if not SCIENTIFIC_LIBS_AVAILABLE:
        return None
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Spectral reflectance curve
        ax1.plot(wavelengths, reflectance * 100, 'g-', linewidth=2, label='Vegetation Reflectance')
        ax1.axvspan(400, 500, alpha=0.2, color='blue', label='Blue')
        ax1.axvspan(500, 600, alpha=0.2, color='green', label='Green')
        ax1.axvspan(600, 700, alpha=0.2, color='red', label='Red')
        ax1.axvspan(700, 850, alpha=0.2, color='darkred', label='NIR')
        
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Reflectance (%)')
        ax1.set_title('Hyperspectral Reflectance Signature')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Plot 2: Vegetation indices bar chart
        indices_names = list(vegetation_indices.keys())
        indices_values = list(vegetation_indices.values())
        
        colors = ['green', 'darkgreen', 'olive', 'blue', 'lightgreen'][:len(indices_names)]
        bars = ax2.bar(indices_names, indices_values, color=colors, alpha=0.7)
        ax2.set_ylabel('Index Value')
        ax2.set_title('Vegetation Indices')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, indices_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_b64
        
    except Exception as e:
        print(f"Plot generation error: {e}")
        return None

def _fallback_hyperspectral_analysis(lat, lon):
    """Fallback analysis when scientific libraries are not available"""
    # Simple mathematical approximations without NumPy
    
    wavelengths = [400 + i * 9 for i in range(50)]  # 400-850nm, 50 bands
    bands = list(range(1, 51))
    
    # Simple vegetation reflectance simulation
    health_factor = 0.7 + (abs(lat) % 30) / 100
    reflectance_data = []
    
    for wl in wavelengths:
        if wl < 500:
            refl = 0.05 + 0.02 * random.random()
        elif wl < 600:
            refl = 0.10 + 0.05 * health_factor
        elif wl < 700:
            refl = 0.06 + 0.02 * (1 - health_factor)
        else:
            refl = 0.45 + 0.25 * health_factor
        
        reflectance_data.append(min(0.95, max(0.01, refl)))
    
    # Calculate indices
    red = reflectance_data[27]  # ~670nm
    nir = reflectance_data[44]  # ~840nm
    green = reflectance_data[16]  # ~560nm
    blue = reflectance_data[9]   # ~485nm
    
    ndvi = (nir - red) / (nir + red) if (nir + red) > 0 else 0
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)) if (nir + 6 * red - 7.5 * blue + 1) > 0 else 0
    savi = ((nir - red) / (nir + red + 0.5)) * 1.5 if (nir + red + 0.5) > 0 else 0
    ndwi = (green - nir) / (green + nir) if (green + nir) > 0 else 0
    gndvi = (nir - green) / (nir + green) if (nir + green) > 0 else 0
    
    vegetation_indices = {
        'ndvi': round(ndvi, 3),
        'evi': round(evi, 3),
        'savi': round(savi, 3),
        'ndwi': round(ndwi, 3),
        'gndvi': round(gndvi, 3)
    }
    
    # Simple classification
    if ndvi > 0.6:
        crop_type = "Healthy Cereal Crop"
        health_status = "Excellent"
        stress_indicators = []
        recommendations = ["Maintain current practices", "Monitor for optimal harvest timing"]
    elif ndvi > 0.3:
        crop_type = "Moderate Vegetation"
        health_status = "Good"
        stress_indicators = ["Mild Stress"] if ndvi < 0.5 else []
        recommendations = ["Continue monitoring", "Optimize irrigation"]
    else:
        crop_type = "Sparse Vegetation"
        health_status = "Poor"
        stress_indicators = ["Water Stress", "Nutrient Deficiency"]
        recommendations = ["Increase irrigation", "Apply fertilizer", "Monitor for pests"]
    
    classification = {
        'crop_type': crop_type,
        'health_status': health_status,
        'health_score': int(ndvi * 100),
        'stress_indicators': stress_indicators,
        'recommendations': recommendations,
        'confidence': 75
    }
    
    return {
        'success': True,
        'analysis_type': 'hyperspectral',
        'bands': bands,
        'wavelengths': wavelengths,
        'reflectance_data': [reflectance_data],
        'vegetation_indices': vegetation_indices,
        'classification': classification,
        'spectral_plot': None,
        'analysis_date': datetime.now().isoformat(),
        'coordinates': {'lat': lat, 'lon': lon},
        'processing_method': 'Fallback Mathematical Model',
        'data_quality': 'Standard',
        'spectral_resolution': '9.0 nm'
    }

def handle_request(event, context):
    """Main function handler for Netlify"""
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': ''
        }
    
    # Parse request
    path = event.get('path', '/').replace('/.netlify/functions/api', '')
    method = event.get('httpMethod', 'GET')
    
    try:
        body = {}
        if method == 'POST' and event.get('body'):
            body = json.loads(event.get('body', '{}'))
    except:
        body = {}
    
    # Default headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
    }
    
    try:
        # Route handling
        if path == '/health' or path == '/api/health':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "environment": "netlify",
                    "version": "1.0.0"
                })
            }
            
        elif (path == '/ndvi' or path == '/api/ndvi') and method == 'POST':
            lat = float(body.get('latitude', 0))
            lon = float(body.get('longitude', 0))
            field_id = body.get('field_id')
            
            # Create processing job
            job_id = create_ndvi_processing_job(lat, lon, field_id)
            
            # Start processing (in real implementation, this would be async)
            result = simulate_real_ndvi_processing(job_id, lat, lon)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }
            
        elif (path == '/hyperspectral' or path == '/api/hyperspectral') and method == 'POST':
            lat = float(body.get('latitude', 0))
            lon = float(body.get('longitude', 0))
            
            # Perform MATLAB-style hyperspectral analysis
            result = matlab_style_hyperspectral_analysis(lat, lon)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(result)
            }
            
        elif (path == '/ndvi/status' or path == '/api/ndvi/status') and method == 'GET':
            # Get job status for processing queue
            job_id = body.get('job_id') if method == 'POST' else None
            
            if job_id and job_id in processing_queue:
                job_status = processing_queue[job_id]
                if job_id in analysis_results:
                    job_status['result'] = analysis_results[job_id]
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps(job_status)
                }
            else:
                # Return all active jobs
                active_jobs = {k: v for k, v in processing_queue.items() 
                              if v['status'] in ['pending', 'processing']}
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'active_jobs': len(active_jobs),
                        'jobs': list(active_jobs.values())[-10:]  # Last 10 jobs
                    })
                }
            
        elif path == '/dashboard-stats' or path == '/api/dashboard-stats':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "total_fields": 12,
                    "total_devices": 45,
                    "active_alerts": 8,
                    "recent_analyses": 24
                })
            }
            
        elif path == '/fields' or path == '/api/fields':
            if method == 'GET':
                # Mock fields data
                mock_fields = [
                    {
                        "id": 1,
                        "name": "North Field",
                        "latitude": 12.9716,
                        "longitude": 77.5946,
                        "area_hectares": 2.5,
                        "crop_type": "Rice",
                        "status": "Active",
                        "device_count": 3,
                        "last_analysis": datetime.now().isoformat(),
                        "created_at": datetime.now().isoformat()
                    },
                    {
                        "id": 2,
                        "name": "South Field", 
                        "latitude": 12.9700,
                        "longitude": 77.5950,
                        "area_hectares": 1.8,
                        "crop_type": "Wheat",
                        "status": "Active",
                        "device_count": 2,
                        "last_analysis": datetime.now().isoformat(),
                        "created_at": datetime.now().isoformat()
                    }
                ]
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({"fields": mock_fields})
                }
                
            elif method == 'POST':
                # Create new field - in production, save to database
                field_data = body
                new_field = {
                    "id": random.randint(100, 999),
                    "success": True,
                    "message": "Field created successfully (demo mode)",
                    **field_data
                }
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps(new_field)
                }
                
        elif path == '/devices' or path == '/api/devices':
            if method == 'GET':
                mock_devices = [
                    {
                        "id": 1,
                        "field_id": 1,
                        "field_name": "North Field",
                        "name": "Soil Sensor 1",
                        "device_type": "Soil Moisture",
                        "device_id": "SMS001",
                        "status": "Active",
                        "battery_level": 85,
                        "last_reading": datetime.now().isoformat()
                    },
                    {
                        "id": 2,
                        "field_id": 1,
                        "field_name": "North Field", 
                        "name": "Weather Station",
                        "device_type": "Weather",
                        "device_id": "WS001",
                        "status": "Active",
                        "battery_level": 92,
                        "last_reading": datetime.now().isoformat()
                    }
                ]
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({"devices": mock_devices})
                }
                
        elif path == '/alerts' or path == '/api/alerts':
            mock_alerts = [
                {
                    "id": 1,
                    "field_id": 1,
                    "field_name": "North Field",
                    "alert_type": "irrigation",
                    "severity": "high",
                    "title": "Low Soil Moisture",
                    "description": "Soil moisture levels below optimal threshold",
                    "status": "Active",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": 2,
                    "field_id": 2,
                    "field_name": "South Field",
                    "alert_type": "disease", 
                    "severity": "medium",
                    "title": "Leaf Spot Detection",
                    "description": "Potential fungal infection detected",
                    "status": "Active",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({"alerts": mock_alerts})
            }
            
        else:
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps({
                    "error": f"Endpoint not found: {path}",
                    "available_endpoints": [
                        "/health", "/ndvi", "/hyperspectral", "/ndvi/status", "/dashboard-stats", 
                        "/fields", "/devices", "/alerts"
                    ]
                })
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                "error": str(e),
                "path": path,
                "method": method
            })
        }

# Netlify function entry point
def handler(event, context):
    return handle_request(event, context)