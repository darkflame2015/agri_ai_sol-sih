import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Upload, File, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

export const Uploads: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Uploads</h1>
          <p className="text-muted-foreground">Upload and process imagery</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload form */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>New Upload</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="files">Select files</Label>
              <div className="flex items-center gap-3">
                <Input id="files" type="file" multiple />
                <Button>
                  <Upload className="w-4 h-4 mr-2" />
                  Start upload
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">Supported: GeoTIFF, JPG, PNG. Max 2GB per file.</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="field">Target field</Label>
                <Input id="field" placeholder="Field A" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="notes">Notes</Label>
                <Input id="notes" placeholder="Optional notes" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Processing queue */}
        <Card>
          <CardHeader>
            <CardTitle>Processing Queue</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {[{name:'NDVI map - Field A', status:'processing', progress:42, eta:'3m'}, {name:'RGB mosaic - Field B', status:'queued', progress:0}, {name:'Moisture map - Field C', status:'complete', progress:100}, {name:'Thermal map - Field D', status:'failed', progress:0}].map((job, idx) => (
              <div key={idx} className="space-y-2 p-3 rounded-lg border">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{job.name}</span>
                  <Badge variant={job.status === 'processing' ? 'warning' : job.status === 'queued' ? 'secondary' : job.status === 'complete' ? 'success' : 'destructive'} className="text-xs capitalize">
                    {job.status}
                  </Badge>
                </div>
                {job.status === 'processing' && <Progress value={job.progress} className="h-2" />}
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{job.status === 'processing' ? `${job.progress}%` : job.status === 'complete' ? '100%' : '-'}</span>
                  <span>{job.eta ? `ETA: ${job.eta}` : job.status === 'complete' ? 'Done' : job.status === 'failed' ? 'Error' : 'Waiting'}</span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Recent uploads */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Uploads</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1,2,3,4].map(i => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg border">
              <div className="flex items-center gap-3">
                <File className="w-4 h-4" />
                <div>
                  <p className="text-sm font-medium">Field_A_{i}.tif</p>
                  <p className="text-xs text-muted-foreground">GeoTIFF • 1.2GB • Field A</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  <span>2h ago</span>
                </div>
                <Badge variant={i % 4 === 0 ? 'destructive' : i % 3 === 0 ? 'success' : i % 2 === 0 ? 'secondary' : 'outline'} className="text-xs">
                  {i % 4 === 0 ? 'Failed' : i % 3 === 0 ? 'Processed' : i % 2 === 0 ? 'Queued' : 'Uploaded'}
                </Badge>
                <div className="text-muted-foreground">
                  {i % 4 === 0 ? <AlertCircle className="w-4 h-4" /> : i % 3 === 0 ? <CheckCircle2 className="w-4 h-4" /> : <Clock className="w-4 h-4" />}
                </div>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default Uploads;
