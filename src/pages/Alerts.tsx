import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Bell, Timer, Activity, MapPin } from 'lucide-react';

export const Alerts: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Alerts</h1>
          <p className="text-muted-foreground">View and manage alerts</p>
        </div>
        <Button variant="outline">Mark all as read</Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="space-y-2 md:col-span-2">
            <Label htmlFor="search">Search</Label>
            <Input id="search" placeholder="Search alerts" />
          </div>
          <div className="space-y-2">
            <Label>Severity</Label>
            <Select defaultValue="any">
              <SelectTrigger>
                <SelectValue placeholder="Any" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Field</Label>
            <Select defaultValue="all">
              <SelectTrigger>
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="A">Field A</SelectItem>
                <SelectItem value="B">Field B</SelectItem>
                <SelectItem value="C">Field C</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Alerts table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Alert</TableHead>
                <TableHead>Field</TableHead>
                <TableHead>Severity</TableHead>
                <TableHead>Time</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[1,2,3,4,5,6].map(i => (
                <TableRow key={i}>
                  <TableCell className="font-medium flex items-center gap-2">
                    <Bell className="w-4 h-4" />
                    Irrigation needed in zone {i}
                  </TableCell>
                  <TableCell>Field {String.fromCharCode(64 + ((i % 3) + 1))}</TableCell>
                  <TableCell>
                    <Badge variant={i % 3 === 0 ? 'destructive' : i % 2 === 0 ? 'warning' : 'secondary'}>
                      {i % 3 === 0 ? 'High' : i % 2 === 0 ? 'Medium' : 'Low'}
                    </Badge>
                  </TableCell>
                  <TableCell> {i * 7} min ago</TableCell>
                  <TableCell className="text-right">
                    <div className="inline-flex items-center gap-2">
                      <Button size="sm" variant="outline">View</Button>
                      <Button size="sm" variant="secondary">Snooze</Button>
                      <Button size="sm">Resolve</Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Stats widgets */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6 flex items-center gap-3">
            <Activity className="w-5 h-5 text-primary" />
            <div>
              <p className="text-xs text-muted-foreground">Open alerts</p>
              <p className="text-2xl font-bold">18</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 flex items-center gap-3">
            <Timer className="w-5 h-5 text-warning" />
            <div>
              <p className="text-xs text-muted-foreground">Avg. response time</p>
              <p className="text-2xl font-bold">12m</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 flex items-center gap-3">
            <MapPin className="w-5 h-5 text-success" />
            <div>
              <p className="text-xs text-muted-foreground">Fields affected</p>
              <p className="text-2xl font-bold">4</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Alerts;
