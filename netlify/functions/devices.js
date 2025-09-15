exports.handler = async (event, context) => {
  // Enable CORS
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Content-Type': 'application/json'
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  // In-memory store (reset on each cold start)
  if (!global.__DEVICES__) global.__DEVICES__ = [];
  const devices = global.__DEVICES__;

  if (event.httpMethod === 'GET') {
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(devices)
    };
  }

  if (event.httpMethod === 'POST') {
    try {
      const { name, field } = JSON.parse(event.body || '{}');
      if (!name || !field) {
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ error: 'Name and field are required.' })
        };
      }
      const device = { id: devices.length + 1, name, field };
      devices.push(device);
      return {
        statusCode: 201,
        headers,
        body: JSON.stringify(device)
      };
    } catch (e) {
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Server error' })
      };
    }
  }

  return {
    statusCode: 405,
    headers,
    body: JSON.stringify({ error: 'Method not allowed' })
  };
};