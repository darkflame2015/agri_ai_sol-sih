import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 4001;

app.use(cors());
app.use(express.json());

// In-memory stores
const devices = [];
const fields = [];

// Create device
app.post('/devices', (req, res) => {
  const { name, field } = req.body;
  if (!name || !field) {
    return res.status(400).json({ error: 'Name and field are required.' });
  }
  const device = { id: devices.length + 1, name, field };
  devices.push(device);
  res.status(201).json(device);
});

// List devices
app.get('/devices', (req, res) => {
  res.json(devices);
});

// Create field
app.post('/fields', (req, res) => {
  const { name, cropType, area } = req.body;
  if (!name || !cropType || !area) {
    return res.status(400).json({ error: 'Name, cropType, and area are required.' });
  }
  const field = { id: fields.length + 1, name, cropType, area };
  fields.push(field);
  res.status(201).json(field);
});

// List fields
app.get('/fields', (req, res) => {
  res.json(fields);
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
