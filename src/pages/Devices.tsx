import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Plus, Power, Wifi, AlertTriangle, MapPin, Settings2 } from 'lucide-react';
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogClose } from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import { createDevice } from '@/lib/api';

export const Devices: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [deviceName, setDeviceName] = useState('');
  const [field, setField] = useState('');
  const { toast } = useToast();

  const handleAddDevice = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createDevice({ name: deviceName, field });
      setOpen(false);
      setDeviceName('');
      setField('');
      toast({ title: 'Device added', description: 'Your new device has been created.' });
    } catch (err: any) {
      toast({ title: 'Failed to add device', description: err?.message || 'Unknown error', variant: 'destructive' });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Devices</h1>
          <p className="text-muted-foreground">Manage IoT sensors</p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-primary">
              <Plus className="w-4 h-4 mr-2" />
              Add Device
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Device</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleAddDevice} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="deviceName">Device Name</Label>
                <Input id="deviceName" value={deviceName} onChange={e => setDeviceName(e.target.value)} required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="field">Field</Label>
                <Input id="field" value={field} onChange={e => setField(e.target.value)} required />
              </div>
              <div className="flex justify-end gap-2">
                <DialogClose asChild>
                  <Button variant="outline" type="button">Cancel</Button>
                </DialogClose>
                <Button type="submit">Add Device</Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="search">Search</Label>
            <Input id="search" placeholder="Device name or ID" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="field">Field</Label>
            <Input id="field" placeholder="Any" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="status">Status</Label>
            <Input id="status" placeholder="Any" />
          </div>
        </CardContent>
      </Card>

      {/* Devices table */}
      <Card>
        <CardHeader>
          <CardTitle>Registered Devices</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Field</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Connectivity</TableHead>
                <TableHead>Battery</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[1,2,3,4,5].map((i) => (
                <TableRow key={i}>
                  <TableCell className="font-medium">Soil Sensor #{i}</TableCell>
                  <TableCell>Field {String.fromCharCode(64 + i)}</TableCell>
                  <TableCell>
                    <Badge variant={i % 4 === 0 ? 'destructive' : i % 3 === 0 ? 'warning' : i % 2 === 0 ? 'secondary' : 'success'}>
                      {i % 4 === 0 ? 'Offline' : i % 3 === 0 ? 'Warning' : i % 2 === 0 ? 'Idle' : 'Active'}
                    </Badge>
                  </TableCell>
                  <TableCell className="flex items-center gap-2">
                    <Wifi className={`w-4 h-4 ${i % 4 === 0 ? 'text-destructive' : 'text-primary'}`} />
                    <span className="text-sm text-muted-foreground">{i % 4 === 0 ? 'No signal' : 'Good'}</span>
                  </TableCell>
                  <TableCell>{100 - i * 12}%</TableCell>
                  <TableCell className="text-right">
                    <div className="inline-flex items-center gap-2">
                      <Button size="sm" variant="outline">
                        <MapPin className="w-4 h-4 mr-1" /> Locate
                      </Button>
                      <Button size="sm" variant="outline">
                        <Settings2 className="w-4 h-4 mr-1" /> Configure
                      </Button>
                      <Button size="sm" variant={i % 4 === 0 ? 'secondary' : 'destructive'}>
                        <Power className="w-4 h-4 mr-1" /> {i % 4 === 0 ? 'Wake' : 'Disable'}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Alerts widgets */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Device Alerts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {[1,2,3].map(i => (
              <div key={i} className="flex items-center justify-between p-3 rounded-lg border">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-4 h-4 text-warning" />
                  <div>
                    <p className="text-sm font-medium">Low battery on Soil Sensor #{i}</p>
                    <p className="text-xs text-muted-foreground">Field {String.fromCharCode(64 + i)} • 12 min ago</p>
                  </div>
                </div>
                <Badge variant="warning">Warning</Badge>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Bulk Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Enable maintenance mode</span>
              <Switch />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Auto-restart offline devices</span>
              <Switch />
            </div>
            <div className="flex justify-end">
              <Button>Apply</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Devices;
