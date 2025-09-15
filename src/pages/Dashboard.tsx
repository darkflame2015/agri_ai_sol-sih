import React from 'react';
import { MapPin, Bell, Activity, Clock, TrendingUp, Eye, Plus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useNavigate } from 'react-router-dom';

const KPICard: React.FC<{
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down';
  icon: React.ReactNode;
}> = ({ title, value, change, trend, icon }) => (
  <Card>
    <CardContent className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {change && (
            <p className={`text-xs ${trend === 'up' ? 'text-success' : 'text-destructive'} flex items-center gap-1 mt-1`}>
              <TrendingUp className={`w-3 h-3 ${trend === 'down' ? 'rotate-180' : ''}`} />
              {change}
            </p>
          )}
        </div>
        <div className="h-12 w-12 bg-primary-soft rounded-lg flex items-center justify-center">
          {icon}
        </div>
      </div>
    </CardContent>
  </Card>
);

const AlertItem: React.FC<{
  title: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
  field: string;
  time: string;
}> = ({ title, description, severity, field, time }) => {
  const severityConfig = {
    high: { color: 'destructive', label: 'High' },
    medium: { color: 'warning', label: 'Medium' },
    low: { color: 'secondary', label: 'Low' }
  };

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div className={`w-2 h-2 rounded-full bg-${severityConfig[severity].color} mt-2 flex-shrink-0`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h4 className="font-medium text-sm">{title}</h4>
            <p className="text-xs text-muted-foreground">{description}</p>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="outline" className="text-xs">
                {field}
              </Badge>
              <span className="text-xs text-muted-foreground">{time}</span>
            </div>
          </div>
          <Badge variant={severityConfig[severity].color as any} className="text-xs">
            {severityConfig[severity].label}
          </Badge>
        </div>
      </div>
    </div>
  );
};

const ProcessingJob: React.FC<{
  name: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  progress: number;
  eta?: string;
}> = ({ name, status, progress, eta }) => {
  const statusConfig = {
    queued: { color: 'secondary', label: 'Queued' },
    processing: { color: 'primary', label: 'Processing' },
    complete: { color: 'success', label: 'Complete' },
    failed: { color: 'destructive', label: 'Failed' }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{name}</span>
        <Badge variant={statusConfig[status].color as any} className="text-xs">
          {statusConfig[status].label}
        </Badge>
      </div>
      <Progress value={progress} className="h-2" />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{progress}%</span>
        {eta && <span>ETA: {eta}</span>}
      </div>
    </div>
  );
};

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your fields and track agricultural insights
          </p>
        </div>
        <Button onClick={() => navigate('/fields/new')} className="bg-gradient-primary">
          <Plus className="w-4 h-4 mr-2" />
          Add Field
        </Button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Fields Monitored"
          value="12"
          change="+2 this month"
          trend="up"
          icon={<MapPin className="w-6 h-6 text-primary" />}
        />
        <KPICard
          title="Active Alerts"
          value="5"
          change="-3 since yesterday"
          trend="down"
          icon={<Bell className="w-6 h-6 text-primary" />}
        />
        <KPICard
          title="Avg NDVI (7d)"
          value="0.72"
          change="+0.05 vs last week"
          trend="up"
          icon={<Activity className="w-6 h-6 text-primary" />}
        />
        <KPICard
          title="Last Sync"
          value="2 min ago"
          icon={<Clock className="w-6 h-6 text-primary" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Overview */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Field Overview</CardTitle>
            <Button variant="outline" size="sm" onClick={() => navigate('/fields')}>
              <Eye className="w-4 h-4 mr-2" />
              View All Fields
            </Button>
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <MapPin className="w-12 h-12 text-muted-foreground mx-auto" />
                <p className="text-muted-foreground">Interactive field map</p>
                <p className="text-sm text-muted-foreground">
                  Showing 12 fields with cluster pins
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Alerts */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Alerts</CardTitle>
            <Button variant="outline" size="sm" onClick={() => navigate('/alerts')}>
              View All
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            <AlertItem
              title="Pest Risk Detected"
              description="High aphid probability in northern section"
              severity="high"
              field="Wheat Field A"
              time="2 hours ago"
            />
            <AlertItem
              title="Nutrient Stress"
              description="Nitrogen deficiency indicators"
              severity="medium"
              field="Corn Field B"
              time="6 hours ago"
            />
            <AlertItem
              title="Irrigation Alert"
              description="Soil moisture below threshold"
              severity="medium"
              field="Tomato Field C"
              time="1 day ago"
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processing Queue */}
        <Card>
          <CardHeader>
            <CardTitle>Processing Queue</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ProcessingJob
              name="Sentinel-2 Scene Analysis"
              status="processing"
              progress={65}
              eta="8 min"
            />
            <ProcessingJob
              name="Drone Image Stitching"
              status="queued"
              progress={0}
              eta="15 min"
            />
            <ProcessingJob
              name="NDVI Calculation"
              status="complete"
              progress={100}
            />
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-success rounded-full mt-2" />
                <div>
                  <p className="text-sm font-medium">Field analysis completed</p>
                  <p className="text-xs text-muted-foreground">Wheat Field A - 10 minutes ago</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2" />
                <div>
                  <p className="text-sm font-medium">New imagery uploaded</p>
                  <p className="text-xs text-muted-foreground">Drone capture - 1 hour ago</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-warning rounded-full mt-2" />
                <div>
                  <p className="text-sm font-medium">Alert acknowledged</p>
                  <p className="text-xs text-muted-foreground">Pest risk - 3 hours ago</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};