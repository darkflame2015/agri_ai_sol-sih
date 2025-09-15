export default async function handler(req, res) {
  if (!global.__FIELDS__) global.__FIELDS__ = [];
  const fields = global.__FIELDS__;

  if (req.method === 'GET') {
    return res.status(200).json(fields);
  }

  if (req.method === 'POST') {
    try {
      const { name, cropType, area } = req.body || {};
      if (!name || !cropType || !area) return res.status(400).json({ error: 'Name, cropType, and area are required.' });
      const field = { id: fields.length + 1, name, cropType, area };
      fields.push(field);
      return res.status(201).json(field);
    } catch (e) {
      return res.status(500).json({ error: 'Server error' });
    }
  }

  res.setHeader('Allow', ['GET', 'POST']);
  return res.status(405).end('Method Not Allowed');
}
