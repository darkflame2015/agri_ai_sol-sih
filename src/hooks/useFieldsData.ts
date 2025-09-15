import { useState, useEffect } from 'react';

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
  created_at: string;
}

interface Device {
  id: number;
  field_id: number;
  field_name: string;
  name: string;
  device_type: string;
  device_id: string;
  status: string;
  battery_level: number;
  last_reading: string;
}

export const useFieldsData = () => {
  const [fields, setFields] = useState<Field[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = import.meta.env.VITE_API_BASE || '/.netlify/functions/api';

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
          last_analysis: new Date().toISOString(),
          created_at: new Date().toISOString()
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
          last_analysis: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          created_at: new Date().toISOString()
        },
        {
          id: 3,
          name: 'East Field',
          latitude: 12.9730,
          longitude: 77.5960,
          area_hectares: 3.2,
          crop_type: 'Corn',
          status: 'Active',
          device_count: 4,
          last_analysis: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
          created_at: new Date().toISOString()
        }
      ]);
    }
  };

  const fetchDevices = async () => {
    try {
      const response = await fetch(`${API_BASE}/devices`);
      if (!response.ok) {
        throw new Error('Failed to fetch devices');
      }
      const data = await response.json();
      setDevices(data.devices || []);
    } catch (err) {
      console.error('Devices fetch error:', err);
      // Fallback to mock data
      setDevices([
        {
          id: 1,
          field_id: 1,
          field_name: 'North Field',
          name: 'Soil Sensor 1',
          device_type: 'Soil Moisture',
          device_id: 'SMS001',
          status: 'Active',
          battery_level: 85,
          last_reading: new Date().toISOString()
        },
        {
          id: 2,
          field_id: 1,
          field_name: 'North Field',
          name: 'Weather Station',
          device_type: 'Weather',
          device_id: 'WS001', 
          status: 'Active',
          battery_level: 92,
          last_reading: new Date().toISOString()
        },
        {
          id: 3,
          field_id: 2,
          field_name: 'South Field',
          name: 'pH Sensor',
          device_type: 'Soil pH',
          device_id: 'PH001',
          status: 'Active',
          battery_level: 78,
          last_reading: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
        }
      ]);
    }
  };

  const createField = async (fieldData: Partial<Field>) => {
    try {
      const response = await fetch(`${API_BASE}/fields`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(fieldData),
      });
      
      if (!response.ok) {
        throw new Error('Failed to create field');
      }
      
      const data = await response.json();
      
      // Refresh fields list
      await fetchFields();
      
      return data;
    } catch (err) {
      console.error('Create field error:', err);
      throw err;
    }
  };

  const createDevice = async (deviceData: Partial<Device>) => {
    try {
      const response = await fetch(`${API_BASE}/devices`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(deviceData),
      });
      
      if (!response.ok) {
        throw new Error('Failed to create device');
      }
      
      const data = await response.json();
      
      // Refresh devices list
      await fetchDevices();
      
      return data;
    } catch (err) {
      console.error('Create device error:', err);
      throw err;
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([fetchFields(), fetchDevices()]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const refreshData = async () => {
    setLoading(true);
    try {
      await Promise.all([fetchFields(), fetchDevices()]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setLoading(false);
    }
  };

  return {
    fields,
    devices,
    loading,
    error,
    createField,
    createDevice,
    refreshData
  };
};