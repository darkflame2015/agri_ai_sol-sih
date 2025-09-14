import { useState } from "react";
import axios from "axios";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from "recharts";

function App() {
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [mode, setMode] = useState("NDVI");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const fetchIndex = async () => {
    try {
      setError("");
      const res = await axios.post("http://127.0.0.1:5000/api/get-index", {
        lat: parseFloat(lat),
        lon: parseFloat(lon),
        mode,
      });
      const raw = res.data.values;

      // convert 2D array -> chart points
      const points = raw.map((row, i) => ({ x: i, y: row[0] }));
      setData(points);
    } catch (err) {
      setError(err.response?.data?.error || "Request failed");
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h2>Sentinel Spectral Index Viewer</h2>
      
      <div style={{ marginBottom: "1rem" }}>
        <input
          type="text"
          placeholder="Latitude"
          value={lat}
          onChange={(e) => setLat(e.target.value)}
          style={{ marginRight: "0.5rem" }}
        />
        <input
          type="text"
          placeholder="Longitude"
          value={lon}
          onChange={(e) => setLon(e.target.value)}
          style={{ marginRight: "0.5rem" }}
        />
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="NDVI">NDVI</option>
          <option value="NDWI">NDWI</option>
          <option value="SAVI">SAVI</option>
        </select>
        <button onClick={fetchIndex} style={{ marginLeft: "1rem" }}>
          Fetch
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {data && (
        <LineChart width={600} height={300} data={data}>
          <Line type="monotone" dataKey="y" stroke="#8884d8" />
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="x" />
          <YAxis />
          <Tooltip />
        </LineChart>
      )}
    </div>
  );
}

export default App;
