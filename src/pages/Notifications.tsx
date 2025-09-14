import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Bell, CheckCircle2, AlertCircle, Clock } from 'lucide-react';

export const Notifications: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Notifications</h1>
          <p className="text-muted-foreground">All recent notifications</p>
        </div>
        <Button variant="outline">Mark all as read</Button>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Recent Notifications</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1,2,3,4,5,6].map(i => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg border">
              <div className="flex items-center gap-3">
                <Bell className="w-4 h-4" />
                <div>
                  <p className="text-sm font-medium">{i % 3 === 0 ? 'Alert: Irrigation needed' : i % 2 === 0 ? 'Upload complete' : 'Device offline'}</p>
                  <p className="text-xs text-muted-foreground">{i % 3 === 0 ? 'Field B' : i % 2 === 0 ? 'Field A' : 'Field C'} • {i * 7} min ago</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={i % 3 === 0 ? 'destructive' : i % 2 === 0 ? 'success' : 'warning'} className="text-xs">
                  {i % 3 === 0 ? 'Alert' : i % 2 === 0 ? 'Info' : 'Warning'}
                </Badge>
                {i % 3 === 0 ? <AlertCircle className="w-4 h-4 text-destructive" /> : i % 2 === 0 ? <CheckCircle2 className="w-4 h-4 text-success" /> : <Clock className="w-4 h-4 text-warning" />}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default Notifications;
