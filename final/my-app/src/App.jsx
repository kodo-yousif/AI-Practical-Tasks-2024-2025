// frontend/src/App.jsx
import React, { useState } from 'react';
import { Home, Info, Linkedin, Github } from 'lucide-react';

const ConfusionMatrix = ({ matrix, labels }) => {
  if (!matrix) return null;

  const getColor = (value) => {
    const maxValue = Math.max(...matrix.flat());
    const intensity = Math.floor((value / maxValue) * 255);
    return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
  };

  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
      <div className="flex flex-col items-center">
        <div className="flex mb-2">
          <div className="w-32"></div>
          <div className="flex">
            <div className="w-24 text-center">Predicted<br />Not Diabetic</div>
            <div className="w-24 text-center">Predicted<br />Diabetic</div>
          </div>
        </div>

        <div className="flex">
          <div className="flex flex-col justify-center mr-2">
            <div className="h-24 flex items-center">
              <div className="w-32 text-right pr-2">Actual<br />Not Diabetic</div>
            </div>
            <div className="h-24 flex items-center">
              <div className="w-32 text-right pr-2">Actual<br />Diabetic</div>
            </div>
          </div>

          <div className="flex flex-col">
            {matrix.map((row, i) => (
              <div key={i} className="flex">
                {row.map((value, j) => (
                  <div
                    key={j}
                    className="w-24 h-24 flex items-center justify-center text-lg font-semibold border"
                    style={{ backgroundColor: getColor(value) }}
                  >
                    {value}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [trainingResults, setTrainingResults] = useState(null);
  const [bestModel, setBestModel] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  });
  const [predictionResult, setPredictionResult] = useState(null);

  const handleTrainModels = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/train", {
        method: "POST"
      });
      if (!response.ok) throw new Error('Training failed');
      const data = await response.json();
      setTrainingResults(data.results);
      setBestModel(data.best_model);
      setSelectedModel(null);
      setConfusionMatrix(null);
    } catch (err) {
      setError('Failed to train models: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSelect = async (modelName) => {
    setSelectedModel(modelName);
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://127.0.0.1:8000/confusion-matrix/${modelName}`);
      if (!response.ok) throw new Error('Failed to fetch confusion matrix');
      const data = await response.json();
      setConfusionMatrix(data);
    } catch (err) {
      setError('Failed to fetch confusion matrix: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    // Validate input
    for (const [key, value] of Object.entries(formData)) {
      if (value === '') {
        alert(`Please enter a value for ${key}`);
        return;
      }
      if (isNaN(value) || Number(value) < 0) {
        alert(`Please enter a valid number for ${key}`);
        return;
      }
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(Object.fromEntries(
          Object.entries(formData).map(([key, value]) => [key, Number(value)])
        ))
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }
      
      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setError('Failed to make prediction: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };
  // Continuing from previous frontend code...

  const AboutPage = () => (
    <div className="max-w-4xl mx-auto p-8">
      <h2 className="text-3xl font-bold text-cyan-600 mb-6">About Our Project</h2>
      <div className="bg-white rounded-lg shadow-lg p-6">
        <p className="text-gray-700 mb-4">
          The Diabetes Prediction System is an advanced machine learning application developed by Group F. 
          Our system utilizes multiple ML models including Neural Networks, SVM, KNN, and Naive Bayes to 
          predict diabetes risk based on various health parameters.
        </p>
        <p className="text-gray-700 mb-4">
          The application compares different models and automatically selects the best performing one 
          for making predictions, ensuring high accuracy and reliability in risk assessment.
        </p>
        <h3 className="text-xl font-semibold text-cyan-600 mt-6 mb-4">Our Team - Group F</h3>
        <div className="space-y-4">
          <h4 className="text-lg font-semibold text-cyan-600">Meet the Team:</h4>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-center">
              <span>Bawar Zring</span>
              <div className="ml-2 flex space-x-2">
                <a href="https://www.linkedin.com/in/bawar-zring-52b545241/" className="text-blue-600 hover:text-blue-800" target="_blank" rel="noopener noreferrer">
                  <Linkedin size={18} />
                </a>
                <a href="https://github.com/Bawar-Zring" className="text-gray-600 hover:text-gray-800" target="_blank" rel="noopener noreferrer">
                  <Github size={18} />
                </a>
              </div>
            </li>
            <li>Halgord Muhammad</li>
            <li>Muhamad Kamal</li>
            <li>Tabarak Xalid</li>
            <li>Ruvan Bestun</li>
          </ul>
        </div>
      </div>
    </div>
  );

return (
<div className="min-h-screen flex flex-col bg-gray-50">
  {/* Header */}
  <header className="bg-white shadow-md">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
      <div className="flex justify-between items-center">
        <div className="text-xl font-bold text-cyan-600">Group F</div>
        <div className="text-2xl font-bold text-cyan-600 [text-shadow:_0_0px_30px_#76B5F0]">
          Diabetes Risk Prediction System
        </div>
        <nav className="flex space-x-4">
          <button 
            onClick={() => setCurrentPage('home')}
            className={`flex items-center space-x-1 px-3 py-2 rounded-md ${
              currentPage === 'home' 
                ? 'bg-cyan-100 text-cyan-600' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Home size={18} />
            <span>Home</span>
          </button>
          <button 
            onClick={() => setCurrentPage('about')}
            className={`flex items-center space-x-1 px-3 py-2 rounded-md ${
              currentPage === 'about' 
                ? 'bg-cyan-100 text-cyan-600' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Info size={18} />
            <span>MyTeam</span>
          </button>
        </nav>
      </div>
    </div>
  </header>

  {/* Main Content */}
  <main className="flex-grow">
    {currentPage === 'about' ? (
      <AboutPage />
    ) : (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <button 
            className={`bg-cyan-600 text-white px-6 py-3 rounded-lg shadow-md hover:bg-cyan-700 transition-colors ${
              isLoading ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            onClick={handleTrainModels}
            disabled={isLoading}
          >
            {isLoading ? 'Training Models...' : 'Train/Evaluate All Models'}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-lg mb-8">
            {error}
          </div>
        )}

        {/* Model Results */}
        {trainingResults && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-bold text-cyan-600 mb-6">Model Performance</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {trainingResults.map((res, index) => (
                <div 
                  key={index}
                  onClick={() => handleModelSelect(res.model)}
                  className={`p-4 rounded-lg cursor-pointer transition-all
                    ${res.model === bestModel 
                      ? 'ring-2 ring-cyan-500 bg-cyan-50' 
                      : 'bg-white hover:bg-gray-50'
                    }
                    ${selectedModel === res.model 
                      ? 'ring-2 ring-cyan-500 shadow-lg' 
                      : 'shadow-md'
                    }`}
                >
                  <h3 className="text-lg font-semibold mb-2">{res.model}</h3>
                  <div className="space-y-1">
                    <div className="text-2xl font-bold text-cyan-600">
                      {(res.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      F1: {(res.f1_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      Precision: {(res.precision * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      Recall: {(res.recall * 100).toFixed(1)}%
                    </div>
                  </div>
                  {res.model === bestModel && (
                    <div className="mt-2 text-cyan-600 font-semibold">Best Model</div>
                  )}
                </div>
              ))}
            </div>

            {/* Confusion Matrix */}
            {selectedModel && confusionMatrix && (
              <div className="mt-8">
                <h3 className="text-xl font-bold text-cyan-600 mb-4">
                  Confusion Matrix for {selectedModel}
                </h3>
                <ConfusionMatrix 
                  matrix={confusionMatrix.confusion_matrix} 
                  labels={confusionMatrix.labels}
                />
              </div>
            )}
          </div>
        )}

        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-cyan-600 mb-6">Enter Patient Data</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.keys(formData).map((key) => (
              <div key={key} className="flex flex-col">
                <label className="text-sm font-medium text-gray-700 mb-1">
                  {key}:
                </label>
                <input
                  type="number"
                  step="any"
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  className="border border-gray-300 rounded-md px-3 py-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder={`Enter ${key}`}
                />
              </div>
            ))}
          </div>

          <div className="mt-6 text-center">
            <button 
              className={`bg-cyan-600 text-white px-6 py-3 rounded-lg shadow-md hover:bg-cyan-700 transition-colors ${
                isLoading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              onClick={handlePredict}
              disabled={isLoading}
            >
              {isLoading ? 'Predicting...' : 'Predict'}
            </button>
          </div>

          {/* Prediction Results */}
          {predictionResult && (
            <div className="mt-6 bg-cyan-50 border border-cyan-200 rounded-lg p-6">
              <h3 className="text-xl font-bold text-cyan-600 mb-4">Prediction Result</h3>
              <div className="space-y-2">
                <p className="text-lg">
                  <span className="font-medium">Prediction: </span>
                  <span className={predictionResult.prediction === 1 ? "text-red-600" : "text-green-600"}>
                    {predictionResult.prediction === 1 ? "Diabetic" : "Not Diabetic"}
                  </span>
                </p>
                <p className="text-lg">
                  <span className="font-medium">Confidence: </span>
                  {(predictionResult.confidence * 100).toFixed(2)}%
                </p>
                <p className="text-lg">
                  <span className="font-medium">Model Used: </span>
                  {predictionResult.model_type}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    )}
  </main>

  {/* Footer */}
  <footer className="bg-gray-800 text-white py-6">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center">
        <p>© {new Date().getFullYear()} Group F - All rights reserved</p>
      </div>
    </div>
  </footer>
</div>
);
};

export default App;