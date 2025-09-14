import React, { useState } from 'react';
import { Plus, Search, Filter, MapPin, Edit, Trash2, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogClose } from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import { createField } from '@/lib/api';
import { useNavigate } from 'react-router-dom';

interface Field {
  id: string;
  name: string;
  cropType: string;
  area: number;
  lastProcessed: string;
  status: 'healthy' | 'warning' | 'alert';
}

const mockFields: Field[] = [
  {
    id: '1',
    name: 'North Wheat Field',
    cropType: 'Wheat',
    area: 25.5,
    lastProcessed: '2 hours ago',
    status: 'healthy'
  },
  {
    id: '2',
    name: 'South Corn Field',
    cropType: 'Corn',
    area: 18.2,
    lastProcessed: '6 hours ago',
    status: 'warning'
  },
  {
    id: '3',
    name: 'East Soybean Field',
    cropType: 'Soybeans',
    area: 32.1,
    lastProcessed: '1 day ago',
    status: 'alert'
  },
  {
    id: '4',
    name: 'West Tomato Field',
    cropType: 'Tomatoes',
    area: 12.8,
    lastProcessed: '4 hours ago',
    status: 'healthy'
  }
];

const EmptyState: React.FC = () => (
  <Card className="text-center py-12">
    <CardContent>
      <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
        <MapPin className="w-8 h-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-semibold mb-2">No fields found</h3>
      <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
        Get started by adding your first field to begin monitoring your crops with AI-powered insights.
      </p>
      <Button className="bg-gradient-primary">
        <Plus className="w-4 h-4 mr-2" />
        Add Your First Field
      </Button>
    </CardContent>
  </Card>
);

export const Fields: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [cropFilter, setCropFilter] = useState<string>('all');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [fieldName, setFieldName] = useState('');
  const [cropType, setCropType] = useState('');
  const [area, setArea] = useState('');
  const { toast } = useToast();
  const navigate = useNavigate();

  const filteredFields = mockFields.filter(field => {
    const matchesSearch = field.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         field.cropType.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || field.status === statusFilter;
    const matchesCrop = cropFilter === 'all' || field.cropType === cropFilter;
    
    return matchesSearch && matchesStatus && matchesCrop;
  });

  const getStatusBadge = (status: Field['status']) => {
    const config = {
      healthy: { variant: 'success' as const, label: 'Healthy' },
      warning: { variant: 'warning' as const, label: 'Warning' },
      alert: { variant: 'destructive' as const, label: 'Alert' }
    };
    
    return (
      <Badge variant={config[status].variant}>
        {config[status].label}
      </Badge>
    );
  };

  const uniqueCropTypes = Array.from(new Set(mockFields.map(field => field.cropType)));

  const handleAddField = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createField({ name: fieldName, cropType, area: Number(area) });
      setDialogOpen(false);
      setFieldName('');
      setCropType('');
      setArea('');
      toast({ title: 'Field added', description: 'Your new field has been created.' });
    } catch (err: any) {
      toast({ title: 'Failed to add field', description: err?.message || 'Unknown error', variant: 'destructive' });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Fields</h1>
          <p className="text-muted-foreground">
            Manage and monitor your agricultural fields
          </p>
        </div>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-primary">
              <Plus className="w-4 h-4 mr-2" />
              Add Field
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Field</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleAddField} className="space-y-4">
              <div className="space-y-2">
                <Input placeholder="Field Name" value={fieldName} onChange={e => setFieldName(e.target.value)} required />
              </div>
              <div className="space-y-2">
                <Input placeholder="Crop Type" value={cropType} onChange={e => setCropType(e.target.value)} required />
              </div>
              <div className="space-y-2">
                <Input placeholder="Area (ha)" value={area} onChange={e => setArea(e.target.value)} required type="number" min="0" />
              </div>
              <div className="flex justify-end gap-2">
                <DialogClose asChild>
                  <Button variant="outline" type="button">Cancel</Button>
                </DialogClose>
                <Button type="submit">Add Field</Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search fields..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full sm:w-48">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="healthy">Healthy</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="alert">Alert</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={cropFilter} onValueChange={setCropFilter}>
              <SelectTrigger className="w-full sm:w-48">
                <SelectValue placeholder="Filter by crop" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Crops</SelectItem>
                {uniqueCropTypes.map(crop => (
                  <SelectItem key={crop} value={crop}>{crop}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Fields Table */}
      {filteredFields.length === 0 && searchQuery === '' && statusFilter === 'all' && cropFilter === 'all' ? (
        <EmptyState />
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>
              {filteredFields.length} Field{filteredFields.length !== 1 ? 's' : ''}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {filteredFields.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-muted-foreground">No fields match your current filters.</p>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setSearchQuery('');
                    setStatusFilter('all');
                    setCropFilter('all');
                  }}
                  className="mt-4"
                >
                  Clear Filters
                </Button>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Field Name</TableHead>
                      <TableHead>Crop Type</TableHead>
                      <TableHead>Area (ha)</TableHead>
                      <TableHead>Last Processed</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredFields.map((field) => (
                      <TableRow key={field.id} className="hover:bg-muted/50">
                        <TableCell>
                          <div className="font-medium">{field.name}</div>
                        </TableCell>
                        <TableCell>{field.cropType}</TableCell>
                        <TableCell>{field.area}</TableCell>
                        <TableCell className="text-muted-foreground">
                          {field.lastProcessed}
                        </TableCell>
                        <TableCell>
                          {getStatusBadge(field.status)}
                        </TableCell>
                        <TableCell className="text-right">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="sm">
                                Actions
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => navigate(`/fields/${field.id}`)}>
                                <Eye className="w-4 h-4 mr-2" />
                                View Details
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => navigate(`/fields/${field.id}/edit`)}>
                                <Edit className="w-4 h-4 mr-2" />
                                Edit Field
                              </DropdownMenuItem>
                              <DropdownMenuItem className="text-destructive">
                                <Trash2 className="w-4 h-4 mr-2" />
                                Delete Field
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};