import json
from datetime import datetime
import random
import base64
from io import BytesIO

def generate_mock_ndvi_data(lat, lon):
    """Generate mock NDVI data for demonstration"""
    try:
        # Create a simple mock NDVI visualization
        # In production, this would connect to Google Earth Engine or similar
        
        # Generate random NDVI-like data
        mock_stats = {
            'avg': round(random.uniform(0.3, 0.7), 3),
            'min': round(random.uniform(-0.1, 0.2), 3), 
            'max': round(random.uniform(0.7, 0.9), 3),
            'std': round(random.uniform(0.1, 0.3), 3)
        }
        
        # For demo purposes, return no image but valid stats
        # In production, integrate with satellite imagery API
        return None, mock_stats
        
    except Exception as e:
        raise Exception(f"NDVI generation failed: {str(e)}")

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
            
            img_data, stats = generate_mock_ndvi_data(lat, lon)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "image": img_data,
                    "statistics": stats,
                    "note": "Demo data - satellite integration in development",
                    "coordinates": {"lat": lat, "lon": lon}
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
                        "/health", "/ndvi", "/dashboard-stats", 
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