import { useState } from "react";
import "./index.css";

function App() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [image, setImage] = useState(null);
  const [error, setError] = useState(null);
  const [locating, setLocating] = useState(false);

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

  const handleUseMyLocation = () => {
    if (!("geolocation" in navigator)) {
      setError("Geolocation is not supported by your browser.");
      return;
    }
    setError(null);
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude: lat, longitude: lon } = pos.coords;
        setLatitude(lat.toFixed(6));
        setLongitude(lon.toFixed(6));
        setLocating(false);
      },
      (err) => {
        let msg = "Unable to retrieve your location.";
        if (err.code === 1) msg = "Permission denied. Please allow location access.";
        else if (err.code === 2) msg = "Position unavailable. Try again.";
        else if (err.code === 3) msg = "Location request timed out.";
        setError(msg);
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
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
        <button onClick={handleUseMyLocation} disabled={locating}>
          {locating ? "Locating..." : "Use My Location"}
        </button>
        <button onClick={getNDVI}>Get NDVI</button>
      </div>

      {error && <p className="error">{error}</p>}
      {image && <img src={image} alt="NDVI" />}
    </div>
  );
}

export default App;
