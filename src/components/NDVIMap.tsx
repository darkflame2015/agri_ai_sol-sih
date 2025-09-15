import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MapPin, Satellite, Download, RefreshCw, Map } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Declare Google Maps types
declare global {
  interface Window {
    google: any;
  }
}

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

interface ProcessingJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string;
  coordinates: { lat: number; lon: number };
  created_at: string;
  estimated_completion?: string;
}

interface NDVIMapProps {
  fieldId?: number;
  initialLat?: number;
  initialLon?: number;
  height?: string;
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const [processingJob, setProcessingJob] = useState<ProcessingJob | null>(null);
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markerRef = useRef<any>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  const API_BASE = import.meta.env.VITE_API_BASE || '/.netlify/functions/api';

  // Initialize Google Maps
  useEffect(() => {
    const initializeMap = () => {
      if (!mapRef.current || !window.google) return;

      try {
        const map = new window.google.maps.Map(mapRef.current, {
          center: { lat: parseFloat(latitude), lng: parseFloat(longitude) },
          zoom: 15,
          mapTypeId: 'satellite', // Show satellite imagery
          mapTypeControl: true,
          mapTypeControlOptions: {
            style: window.google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
            position: window.google.maps.ControlPosition.TOP_CENTER,
            mapTypeIds: ['roadmap', 'satellite', 'hybrid', 'terrain']
          },
          streetViewControl: false,
          fullscreenControl: true,
        });

        mapInstanceRef.current = map;
        setMapLoaded(true);

        // Add click listener to update coordinates
        map.addListener('click', (event: any) => {
          if (event.latLng) {
            const lat = event.latLng.lat();
            const lng = event.latLng.lng();
            setLatitude(lat.toFixed(6));
            setLongitude(lng.toFixed(6));
            updateMarker(lat, lng);
          }
        });

        // Initial marker
        updateMarker(parseFloat(latitude), parseFloat(longitude));
      } catch (error) {
        console.error('Map initialization error:', error);
        setError('Failed to initialize map. Using coordinate input mode.');
      }
    };

    // Load Google Maps API if not already loaded
    if (!window.google) {
      const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
      if (apiKey) {
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`;
        script.async = true;
        script.defer = true;
        script.onload = initializeMap;
        script.onerror = () => {
          console.error('Failed to load Google Maps API');
          setError('Google Maps API failed to load. Using coordinate input mode.');
        };
        document.head.appendChild(script);
      } else {
        console.warn('Google Maps API key not found. Using coordinate input mode.');
        setError('Google Maps API key not configured. Using coordinate input mode.');
      }
    } else {
      initializeMap();
    }
  }, []);

  // Update map center and marker when coordinates change
  useEffect(() => {
    if (mapInstanceRef.current && mapLoaded) {
      const lat = parseFloat(latitude);
      const lng = parseFloat(longitude);
      if (!isNaN(lat) && !isNaN(lng)) {
        mapInstanceRef.current.setCenter({ lat, lng });
        updateMarker(lat, lng);
      }
    }
  }, [latitude, longitude, mapLoaded]);

  const updateMarker = (lat: number, lng: number) => {
    if (!mapInstanceRef.current || !window.google) return;

    try {
      if (markerRef.current) {
        markerRef.current.setMap(null);
      }

      markerRef.current = new window.google.maps.Marker({
        position: { lat, lng },
        map: mapInstanceRef.current,
        title: 'NDVI Analysis Location',
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: '#3B82F6',
          fillOpacity: 0.8,
          strokeColor: '#1E40AF',
          strokeWeight: 2,
        },
      });
    } catch (error) {
      console.error('Marker update error:', error);
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
    setProcessingJob(null);

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

      if (data.job_id) {
        setProcessingJob({
          id: data.job_id,
          status: data.status || 'processing',
          progress: data.progress || 0,
          stage: data.stage || 'initializing',
          coordinates: { lat: parseFloat(latitude), lon: parseFloat(longitude) },
          created_at: new Date().toISOString()
        });
        
        // Start polling for results if job is processing
        if (data.status === 'processing') {
          pollJobStatus(data.job_id);
        }
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

  const pollJobStatus = async (jobId: string) => {
    const maxPolls = 30; // Maximum 30 polls (30 seconds)
    let pollCount = 0;

    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/ndvi/status`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ job_id: jobId }),
        });

        if (response.ok) {
          const jobData = await response.json();
          
          setProcessingJob(prev => prev ? { ...prev, ...jobData } : jobData);

          if (jobData.status === 'completed' && jobData.result) {
            setAnalysis(jobData.result);
            setLoading(false);
            return;
          } else if (jobData.status === 'failed') {
            setError(jobData.stage || 'Processing failed');
            setLoading(false);
            return;
          }
        }

        pollCount++;
        if (pollCount < maxPolls && (processingJob?.status === 'processing' || processingJob?.status === 'pending')) {
          setTimeout(poll, 1000); // Poll every second
        } else {
          setLoading(false);
          if (pollCount >= maxPolls) {
            setError('Processing timeout - please try again');
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
        setLoading(false);
      }
    };

    poll();
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
      {/* Map and Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Interactive Map */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Map className="w-5 h-5" />
                Satellite Map - Click to Select Location
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div 
                ref={mapRef} 
                className={`${height} w-full rounded-lg ${!mapLoaded ? 'bg-muted' : ''}`}
              >
                {!mapLoaded && (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center space-y-2">
                      <RefreshCw className="w-8 h-8 text-muted-foreground mx-auto animate-spin" />
                      <p className="text-muted-foreground">Loading satellite map...</p>
                    </div>
                  </div>
                )}
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
                onClick={getNDVIAnalysis}
                disabled={loading}
                className="w-full"
              >
                <Satellite className="w-4 h-4 mr-2" />
                {loading ? 'Analyzing...' : 'Start NDVI Analysis'}
              </Button>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {processingJob && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Processing Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Status:</span>
                  <span className={`font-medium ${
                    processingJob.status === 'completed' ? 'text-green-600' :
                    processingJob.status === 'failed' ? 'text-red-600' :
                    'text-blue-600'
                  }`}>
                    {processingJob.status.charAt(0).toUpperCase() + processingJob.status.slice(1)}
                  </span>
                </div>
                
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span>Progress:</span>
                    <span>{processingJob.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${processingJob.progress}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="text-xs text-muted-foreground">
                  {processingJob.stage}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {loading && !processingJob && (
        <div className="bg-muted rounded-lg p-8 text-center">
          <RefreshCw className="w-8 h-8 text-muted-foreground mx-auto mb-4 animate-spin" />
          <p className="text-muted-foreground mb-2">Initializing NDVI Analysis...</p>
          <p className="text-sm text-muted-foreground">Processing satellite imagery data</p>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* NDVI Image */}
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
                  <div className="space-y-4">
                    <div className="rounded-lg overflow-hidden border">
                      <img
                        src={`data:image/png;base64,${analysis.image}`}
                        alt="NDVI Analysis"
                        className="w-full h-auto"
                      />
                    </div>
                    
                    {/* Analysis Metadata */}
                    <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                      <div>
                        <strong>Data Source:</strong> {analysis.data_source || 'Satellite Imagery'}
                      </div>
                      <div>
                        <strong>Processing Time:</strong> {analysis.processing_time_seconds || 0}s
                      </div>
                      <div>
                        <strong>Cloud Coverage:</strong> {analysis.cloud_coverage || 0}%
                      </div>
                      <div>
                        <strong>Analysis Date:</strong> {analysis.analysis_date ? new Date(analysis.analysis_date).toLocaleDateString() : 'Today'}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
                    <p className="text-muted-foreground">Analysis completed - image data processing...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Enhanced Statistics */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Vegetation Health Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Overall Health Score */}
                <div className="text-center p-4 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground mb-1">Overall Health</p>
                  <p className={`text-lg font-semibold mb-2 ${getHealthStatus(analysis.statistics.avg).color}`}>
                    {getHealthStatus(analysis.statistics.avg).status}
                  </p>
                  <div className="relative">
                    <div className="text-3xl font-bold">{formatValue(analysis.statistics.avg)}</div>
                    <div className="text-sm text-muted-foreground">NDVI Score</div>
                  </div>
                </div>

                {/* Detailed Statistics */}
                <div className="space-y-3">
                  <h4 className="font-medium text-sm">Statistical Analysis</h4>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Minimum:</span>
                      <div className="text-right">
                        <span className="font-medium">{formatValue(analysis.statistics.min)}</span>
                        <div className={`text-xs ${getHealthStatus(analysis.statistics.min).color}`}>
                          {getHealthStatus(analysis.statistics.min).status}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Maximum:</span>
                      <div className="text-right">
                        <span className="font-medium">{formatValue(analysis.statistics.max)}</span>
                        <div className={`text-xs ${getHealthStatus(analysis.statistics.max).color}`}>
                          {getHealthStatus(analysis.statistics.max).status}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Std Deviation:</span>
                      <span className="font-medium">{formatValue(analysis.statistics.std)}</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Variability:</span>
                      <span className={`font-medium text-sm ${
                        analysis.statistics.std < 0.1 ? 'text-green-600' :
                        analysis.statistics.std < 0.2 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {analysis.statistics.std < 0.1 ? 'Low' : 
                         analysis.statistics.std < 0.2 ? 'Medium' : 'High'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* NDVI Reference Guide */}
                <div className="pt-3 border-t">
                  <h4 className="font-medium text-sm mb-2">NDVI Reference Scale</h4>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div className="flex justify-between">
                      <span>• -1 to 0:</span>
                      <span>Water/Snow</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• 0 to 0.2:</span>
                      <span>Bare soil/Rock</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• 0.2 to 0.4:</span>
                      <span>Sparse vegetation</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• 0.4 to 0.6:</span>
                      <span>Moderate vegetation</span>
                    </div>
                    <div className="flex justify-between">
                      <span>• 0.6 to 1:</span>
                      <span>Dense vegetation</span>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="pt-3 border-t">
                  <h4 className="font-medium text-sm mb-2">Recommendations</h4>
                  <div className="text-xs text-muted-foreground">
                    {analysis.statistics.avg > 0.6 ? (
                      <p className="text-green-600">Excellent vegetation health. Continue current management practices.</p>
                    ) : analysis.statistics.avg > 0.4 ? (
                      <p className="text-green-500">Good vegetation health. Monitor for optimal growth conditions.</p>
                    ) : analysis.statistics.avg > 0.2 ? (
                      <p className="text-yellow-600">Fair vegetation health. Consider irrigation or fertilization.</p>
                    ) : (
                      <p className="text-red-600">Poor vegetation health. Immediate intervention recommended.</p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Default State */}
      {!analysis && !loading && !error && !processingJob && (
        <div className="bg-muted rounded-lg p-8 text-center">
          <Satellite className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground mb-2">NDVI Satellite Analysis</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Click on the map or enter coordinates to analyze crop health using satellite imagery
          </p>
          <div className="text-xs text-muted-foreground">
            <p>• Real-time satellite data processing</p>
            <p>• Vegetation health assessment</p>
            <p>• Professional analysis reports</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NDVIMap;