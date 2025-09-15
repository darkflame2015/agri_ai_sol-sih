import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MapPin, Satellite, Download, RefreshCw, Map, BarChart3 } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default markers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface NDVIAnalysis {
  image: string;
  statistics: {
    avg: number;
    min: number;
    max: number;
    std: number;
  };
  analysis_id?: number;
  job_id?: string;
  processing_time_seconds?: number;
  data_source?: string;
  cloud_coverage?: number;
  analysis_date?: string;
}

interface HyperspectralAnalysis {
  bands: number[];
  wavelengths: number[];
  reflectance_data: number[][];
  vegetation_indices: {
    ndvi: number;
    evi: number;
    savi: number;
    ndwi: number;
    gndvi: number;
  };
  classification: {
    crop_type: string;
    health_status: string;
    stress_indicators: string[];
    recommendations: string[];
  };
}

interface NDVIMapProps {
  fieldId?: number;
  initialLat?: number;
  initialLon?: number;
  height?: string;
}

// Component for handling map clicks
function MapClickHandler({ onLocationSelect }: { onLocationSelect: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (e) => {
      onLocationSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

const NDVIMap: React.FC<NDVIMapProps> = ({ 
  fieldId, 
  initialLat = 12.9716, 
  initialLon = 77.5946,
  height = "h-96"
}) => {
  const [latitude, setLatitude] = useState<string>(initialLat.toString());
  const [longitude, setLongitude] = useState<string>(initialLon.toString());
  const [analysis, setAnalysis] = useState<NDVIAnalysis | null>(null);
  const [hyperspectralAnalysis, setHyperspectralAnalysis] = useState<HyperspectralAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const [analysisType, setAnalysisType] = useState<'ndvi' | 'hyperspectral'>('ndvi');

  const API_BASE = import.meta.env.VITE_API_BASE || '/.netlify/functions/api';

  const onLocationSelect = (lat: number, lng: number) => {
    setLatitude(lat.toFixed(6));
    setLongitude(lng.toFixed(6));
  };

  const useCurrentLocation = () => {
    if (!('geolocation' in navigator)) {
      setError('Geolocation is not supported by your browser');
      return;
    }

    setError(null);
    setLocating(true);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude: lat, longitude: lon } = position.coords;
        setLatitude(lat.toFixed(6));
        setLongitude(lon.toFixed(6));
        setLocating(false);
      },
      (err) => {
        let message = 'Unable to retrieve your location';
        if (err.code === 1) message = 'Permission denied. Please allow location access';
        else if (err.code === 2) message = 'Position unavailable. Try again';
        else if (err.code === 3) message = 'Location request timed out';
        
        setError(message);
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  };

  const generateHyperspectralAnalysis = async () => {
    if (!latitude || !longitude) {
      setError('Please enter valid coordinates');
      return;
    }

    setLoading(true);
    setError(null);
    setHyperspectralAnalysis(null);

    try {
      const response = await fetch(`${API_BASE}/hyperspectral`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: parseFloat(latitude),
          longitude: parseFloat(longitude),
          field_id: fieldId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      if (data.success) {
        setHyperspectralAnalysis(data);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Hyperspectral analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const getNDVIAnalysis = async () => {
    if (!latitude || !longitude) {
      setError('Please enter valid coordinates');
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      const response = await fetch(`${API_BASE}/ndvi`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: parseFloat(latitude),
          longitude: parseFloat(longitude),
          field_id: fieldId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      if (data.success) {
        setAnalysis(data);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!analysis?.image) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${analysis.image}`;
    link.download = `ndvi-analysis-${latitude}-${longitude}-${new Date().toISOString().split('T')[0]}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatValue = (value: number, decimals: number = 3): string => {
    return value.toFixed(decimals);
  };

  const getHealthStatus = (ndviValue: number): { status: string; color: string } => {
    if (ndviValue > 0.6) return { status: 'Excellent', color: 'text-green-600' };
    if (ndviValue > 0.4) return { status: 'Good', color: 'text-green-500' };
    if (ndviValue > 0.2) return { status: 'Fair', color: 'text-yellow-500' };
    if (ndviValue > 0.0) return { status: 'Poor', color: 'text-orange-500' };
    return { status: 'Critical', color: 'text-red-500' };
  };

  return (
    <div className="space-y-4">
      {/* Analysis Type Selector */}
      <div className="flex gap-2 mb-4">
        <Button
          variant={analysisType === 'ndvi' ? 'default' : 'outline'}
          onClick={() => setAnalysisType('ndvi')}
          className="flex items-center gap-2"
        >
          <Satellite className="w-4 h-4" />
          NDVI Analysis
        </Button>
        <Button
          variant={analysisType === 'hyperspectral' ? 'default' : 'outline'}
          onClick={() => setAnalysisType('hyperspectral')}
          className="flex items-center gap-2"
        >
          <BarChart3 className="w-4 h-4" />
          Hyperspectral Analysis
        </Button>
      </div>

      {/* Map and Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Free Leaflet Map */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Map className="w-5 h-5" />
                Satellite View
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`${height} w-full rounded-lg overflow-hidden border`}>
                <MapContainer
                  center={[parseFloat(latitude), parseFloat(longitude)]}
                  zoom={15}
                  style={{ height: '100%', width: '100%' }}
                  scrollWheelZoom={true}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  
                  <TileLayer
                    attribution='Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    opacity={0.8}
                  />
                  
                  <MapClickHandler onLocationSelect={onLocationSelect} />
                  
                  <Marker position={[parseFloat(latitude), parseFloat(longitude)]}>
                    <Popup>
                      Analysis Location<br />
                      Lat: {parseFloat(latitude).toFixed(4)}<br />
                      Lng: {parseFloat(longitude).toFixed(4)}
                    </Popup>
                  </Marker>
                </MapContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Controls Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="latitude">Latitude</Label>
                <Input
                  id="latitude"
                  type="number"
                  step="any"
                  placeholder="e.g., 12.9716"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="longitude">Longitude</Label>
                <Input
                  id="longitude"
                  type="number"
                  step="any"
                  placeholder="e.g., 77.5946"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                />
              </div>
              
              <Button
                variant="outline"
                onClick={useCurrentLocation}
                disabled={locating}
                className="w-full"
              >
                <MapPin className="w-4 h-4 mr-2" />
                {locating ? 'Locating...' : 'Use My Location'}
              </Button>
              
              <Button
                onClick={analysisType === 'hyperspectral' ? generateHyperspectralAnalysis : getNDVIAnalysis}
                disabled={loading}
                className="w-full"
              >
                {analysisType === 'hyperspectral' ? (
                  <>
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Generate Hyperspectral
                  </>
                ) : (
                  <>
                    <Satellite className="w-4 h-4 mr-2" />
                    Start NDVI Analysis
                  </>
                )}
                {loading && <RefreshCw className="w-4 h-4 ml-2 animate-spin" />}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {loading && (
        <div className="bg-muted rounded-lg p-8 text-center">
          <RefreshCw className="w-8 h-8 text-muted-foreground mx-auto mb-4 animate-spin" />
          <p className="text-muted-foreground mb-2">
            {analysisType === 'hyperspectral' ? 'Generating Hyperspectral Analysis...' : 'Processing NDVI Analysis...'}
          </p>
          <p className="text-sm text-muted-foreground">Processing satellite imagery data</p>
        </div>
      )}

      {/* NDVI Analysis Results */}
      {analysis && analysisType === 'ndvi' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Satellite className="w-5 h-5" />
                  NDVI Analysis Results
                </CardTitle>
                {analysis.image && (
                  <Button variant="outline" size="sm" onClick={downloadImage}>
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                )}
              </CardHeader>
              <CardContent>
                {analysis.image ? (
                  <div className="rounded-lg overflow-hidden border">
                    <img
                      src={`data:image/png;base64,${analysis.image}`}
                      alt="NDVI Analysis"
                      className="w-full h-auto"
                    />
                  </div>
                ) : (
                  <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
                    <p className="text-muted-foreground">Analysis completed - image processing...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div>
            <Card>
              <CardHeader>
                <CardTitle>Health Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center p-4 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground mb-1">Overall Health</p>
                  <p className={`text-lg font-semibold mb-2 ${getHealthStatus(analysis.statistics.avg).color}`}>
                    {getHealthStatus(analysis.statistics.avg).status}
                  </p>
                  <div className="text-3xl font-bold">{formatValue(analysis.statistics.avg)}</div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Minimum:</span>
                    <span className="font-medium">{formatValue(analysis.statistics.min)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Maximum:</span>
                    <span className="font-medium">{formatValue(analysis.statistics.max)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Std Dev:</span>
                    <span className="font-medium">{formatValue(analysis.statistics.std)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Hyperspectral Analysis Results */}
      {hyperspectralAnalysis && analysisType === 'hyperspectral' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Vegetation Indices
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground">NDVI</div>
                  <div className="text-xl font-bold">{hyperspectralAnalysis.vegetation_indices.ndvi}</div>
                </div>
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground">EVI</div>
                  <div className="text-xl font-bold">{hyperspectralAnalysis.vegetation_indices.evi}</div>
                </div>
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground">SAVI</div>
                  <div className="text-xl font-bold">{hyperspectralAnalysis.vegetation_indices.savi}</div>
                </div>
                <div className="text-center p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground">GNDVI</div>
                  <div className="text-xl font-bold">{hyperspectralAnalysis.vegetation_indices.gndvi}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Classification Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="text-sm text-muted-foreground">Crop Type</div>
                <div className="text-lg font-semibold">{hyperspectralAnalysis.classification.crop_type}</div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground">Health Status</div>
                <div className={`text-lg font-semibold ${getHealthStatus(hyperspectralAnalysis.vegetation_indices.ndvi).color}`}>
                  {hyperspectralAnalysis.classification.health_status}
                </div>
              </div>

              {hyperspectralAnalysis.classification.stress_indicators.length > 0 && (
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Stress Indicators</div>
                  <div className="space-y-1">
                    {hyperspectralAnalysis.classification.stress_indicators.map((indicator, index) => (
                      <div key={index} className="text-sm text-orange-600 bg-orange-50 px-2 py-1 rounded">
                        {indicator}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="text-sm text-muted-foreground mb-2">Recommendations</div>
                <div className="space-y-1">
                  {hyperspectralAnalysis.classification.recommendations.map((rec, index) => (
                    <div key={index} className="text-sm text-blue-600 bg-blue-50 px-2 py-1 rounded">
                      {rec}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Default State */}
      {!analysis && !hyperspectralAnalysis && !loading && !error && (
        <div className="bg-muted rounded-lg p-8 text-center">
          <div className="flex justify-center mb-4">
            {analysisType === 'ndvi' ? (
              <Satellite className="w-12 h-12 text-muted-foreground" />
            ) : (
              <BarChart3 className="w-12 h-12 text-muted-foreground" />
            )}
          </div>
          <h3 className="text-lg font-medium text-muted-foreground mb-2">
            {analysisType === 'ndvi' ? 'NDVI Satellite Analysis' : 'Hyperspectral Analysis'}
          </h3>
          <p className="text-sm text-muted-foreground mb-4">
            Click on the map or enter coordinates to analyze crop health using
            {analysisType === 'ndvi' ? ' satellite imagery' : ' MATLAB-style hyperspectral analysis'}
          </p>
          <div className="text-xs text-muted-foreground">
            <p>• Free satellite imagery from OpenStreetMap & Esri</p>
            <p>• {analysisType === 'ndvi' ? 'Real-time NDVI processing' : 'Multi-band spectral analysis'}</p>
            <p>• Professional analysis reports</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NDVIMap;
