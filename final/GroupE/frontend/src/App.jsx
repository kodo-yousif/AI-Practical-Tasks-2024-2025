import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8080/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log(response);
      setResults(response.data);
    } catch (err) {
      setError(err.response ? err.response.data.error : "An error occurred.");
      console.log(err);
    } finally {
      setLoading(false);
    }
  };

  //   console.log(results)
  return (
    <div className="App">
      <header className="App-header">
        <h1>Model Training Dashboard</h1>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading}>
          Upload and Train
        </button>
        {loading && <p>Loading...</p>}
        {error && <p className="error">Error: {error}</p>}
        {results && (
          <div className="results">
            <h2>Model Results</h2>
            {Object.keys(results).map((model) => (
              <div key={model} className="result-block">
                <h3>{model}</h3>
                <pre>{JSON.stringify(results[model], null, 2)}</pre>
              </div>
            ))}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
