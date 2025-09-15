export default async function handler(req, res) {
  if (!global.__DEVICES__) global.__DEVICES__ = [];
  const devices = global.__DEVICES__;

  if (req.method === 'GET') {
    return res.status(200).json(devices);
  }

  if (req.method === 'POST') {
    try {
      const { name, field } = req.body || {};
      if (!name || !field) return res.status(400).json({ error: 'Name and field are required.' });
      const device = { id: devices.length + 1, name, field };
      devices.push(device);
      return res.status(201).json(device);
    } catch (e) {
      return res.status(500).json({ error: 'Server error' });
    }
  }

  res.setHeader('Allow', ['GET', 'POST']);
  return res.status(405).end('Method Not Allowed');
}
