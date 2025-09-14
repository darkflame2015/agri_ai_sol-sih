import { useState } from "react";
import "./index.css";

function App() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [image, setImage] = useState(null);
  const [error, setError] = useState(null);

  const getNDVI = async () => {
    setError(null);
    setImage(null);
    try {
      const res = await fetch("http://127.0.0.1:5000/api/ndvi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latitude, longitude }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setImage("data:image/png;base64," + data.image);
      }
    } catch (err) {
      setError("Request failed: " + err.message);
    }
  };

  return (
    <div className="app">
      <h1>Crop NDVI Viewer (React)</h1>
      <div className="form">
        <input
          type="text"
          placeholder="Latitude e.g., 12.9716"
          value={latitude}
          onChange={(e) => setLatitude(e.target.value)}
        />
        <input
          type="text"
          placeholder="Longitude e.g., 77.5946"
          value={longitude}
          onChange={(e) => setLongitude(e.target.value)}
        />
        <button onClick={getNDVI}>Get NDVI</button>
      </div>

      {error && <p className="error">{error}</p>}
      {image && <img src={image} alt="NDVI" />}
    </div>
  );
}

export default App;
