import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MapPin, Satellite, Download, RefreshCw } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface NDVIAnalysis {
  image: string;
  statistics: {
    avg: number;
    min: number;
    max: number;
    std: number;
  };
  analysis_id?: number;
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
  height = "h-64"
}) => {
  const [latitude, setLatitude] = useState<string>(initialLat.toString());
  const [longitude, setLongitude] = useState<string>(initialLon.toString());
  const [analysis, setAnalysis] = useState<NDVIAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);

  const API_BASE = import.meta.env.VITE_API_BASE || '/.netlify/functions/api';

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

      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
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
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
        <div className="flex items-end gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={useCurrentLocation}
            disabled={locating}
            className="flex-1"
          >
            <MapPin className="w-4 h-4 mr-2" />
            {locating ? 'Locating...' : 'My Location'}
          </Button>
          <Button
            onClick={getNDVIAnalysis}
            disabled={loading}
            size="sm"
            className="flex-1"
          >
            <Satellite className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Analyze'}
          </Button>
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
        <div className={`${height} bg-muted rounded-lg flex items-center justify-center`}>
          <div className="text-center space-y-2">
            <RefreshCw className="w-8 h-8 text-muted-foreground mx-auto animate-spin" />
            <p className="text-muted-foreground">Analyzing satellite imagery...</p>
            <p className="text-sm text-muted-foreground">This may take a moment</p>
          </div>
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
                  NDVI Analysis
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
                  <div className="rounded-lg overflow-hidden">
                    <img
                      src={`data:image/png;base64,${analysis.image}`}
                      alt="NDVI Analysis"
                      className="w-full h-auto"
                    />
                  </div>
                ) : (
                  <div className={`${height} bg-muted rounded-lg flex items-center justify-center`}>
                    <p className="text-muted-foreground">Analysis completed - image processing...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Statistics */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Health Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Overall Health</p>
                  <p className={`text-lg font-semibold ${getHealthStatus(analysis.statistics.avg).color}`}>
                    {getHealthStatus(analysis.statistics.avg).status}
                  </p>
                  <p className="text-2xl font-bold">{formatValue(analysis.statistics.avg)}</p>
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

                <div className="pt-3 border-t">
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p><strong>NDVI Scale:</strong></p>
                    <p>• -1 to 0: Water/Snow</p>
                    <p>• 0 to 0.2: Bare soil/Rock</p>
                    <p>• 0.2 to 0.4: Sparse vegetation</p>
                    <p>• 0.4 to 0.6: Moderate vegetation</p>
                    <p>• 0.6 to 1: Dense vegetation</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Default State */}
      {!analysis && !loading && !error && (
        <div className={`${height} bg-muted rounded-lg flex items-center justify-center`}>
          <div className="text-center space-y-2">
            <Satellite className="w-12 h-12 text-muted-foreground mx-auto" />
            <p className="text-muted-foreground">NDVI Satellite Analysis</p>
            <p className="text-sm text-muted-foreground">
              Enter coordinates and click "Analyze" to view crop health
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NDVIMap;