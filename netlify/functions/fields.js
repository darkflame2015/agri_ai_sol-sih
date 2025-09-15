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
  if (!global.__FIELDS__) global.__FIELDS__ = [];
  const fields = global.__FIELDS__;

  if (event.httpMethod === 'GET') {
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(fields)
    };
  }

  if (event.httpMethod === 'POST') {
    try {
      const { name, cropType, area } = JSON.parse(event.body || '{}');
      if (!name || !cropType || !area) {
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ error: 'Name, cropType, and area are required.' })
        };
      }
      const field = { id: fields.length + 1, name, cropType, area };
      fields.push(field);
      return {
        statusCode: 201,
        headers,
        body: JSON.stringify(field)
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