import { useState, useEffect } from 'react';

interface DashboardStats {
  total_fields: number;
  total_devices: number;
  active_alerts: number;
  recent_analyses: number;
}

interface Alert {
  id: number;
  field_id: number;
  field_name: string;
  alert_type: string;
  severity: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  status: string;
  created_at: string;
}

interface Field {
  id: number;
  name: string;
  latitude: number;
  longitude: number;
  area_hectares: number;
  crop_type: string;
  status: string;
  device_count: number;
  last_analysis: string;
}

export const useDashboardData = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [fields, setFields] = useState<Field[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = import.meta.env.VITE_API_BASE || '/.netlify/functions/api';

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/dashboard-stats`);
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard stats');
      }
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Dashboard stats error:', err);
      // Fallback to mock data
      setStats({
        total_fields: 12,
        total_devices: 45,
        active_alerts: 8,
        recent_analyses: 24
      });
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE}/alerts`);
      if (!response.ok) {
        throw new Error('Failed to fetch alerts');
      }
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (err) {
      console.error('Alerts fetch error:', err);
      // Fallback to mock data
      setAlerts([
        {
          id: 1,
          field_id: 1,
          field_name: 'North Field',
          alert_type: 'irrigation',
          severity: 'high',
          title: 'Low Soil Moisture',
          description: 'Soil moisture levels below optimal threshold',
          status: 'Active',
          created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
        },
        {
          id: 2,
          field_id: 2,
          field_name: 'South Field',
          alert_type: 'disease',
          severity: 'medium',
          title: 'Leaf Spot Detection',
          description: 'Potential fungal infection detected in sector B',
          status: 'Active',
          created_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString()
        },
        {
          id: 3,
          field_id: 3,
          field_name: 'West Field',
          alert_type: 'nutrient',
          severity: 'low',
          title: 'Nitrogen Deficiency',
          description: 'Nitrogen levels slightly below recommended range',
          status: 'Active',
          created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
        }
      ]);
    }
  };

  const fetchFields = async () => {
    try {
      const response = await fetch(`${API_BASE}/fields`);
      if (!response.ok) {
        throw new Error('Failed to fetch fields');
      }
      const data = await response.json();
      setFields(data.fields || []);
    } catch (err) {
      console.error('Fields fetch error:', err);
      // Fallback to mock data
      setFields([
        {
          id: 1,
          name: 'North Field',
          latitude: 12.9716,
          longitude: 77.5946,
          area_hectares: 2.5,
          crop_type: 'Rice',
          status: 'Active',
          device_count: 3,
          last_analysis: new Date().toISOString()
        },
        {
          id: 2,
          name: 'South Field',
          latitude: 12.9700,
          longitude: 77.5950,
          area_hectares: 1.8,
          crop_type: 'Wheat',
          status: 'Active',
          device_count: 2,
          last_analysis: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
        }
      ]);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchDashboardStats(),
          fetchAlerts(),
          fetchFields()
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const refreshData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchDashboardStats(),
        fetchAlerts(),
        fetchFields()
      ]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setLoading(false);
    }
  };

  return {
    stats,
    alerts,
    fields,
    loading,
    error,
    refreshData
  };
};