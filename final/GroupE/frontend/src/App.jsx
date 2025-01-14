import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";

export default function App() {
  const [validationMethod, setValidationMethod] = useState("holdout");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [modelStatus, setModelStatus] = useState({});

  const defaultValues = {
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50,
    modelType: "kNN",
  };

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues,
  });

  const parseError = async (response) => {
    try {
      const data = await response.json();
      return data.detail || "An error occurred";
    } catch (e) {
      return await response.text();
    }
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8080/startup", {
          method: "GET",
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText);
        }

        console.log("Models successfully loaded.");
      } catch (err) {
        console.error("Error loading models:", err);
        setError("Failed to load models. Please try again.");
      }
    };

    loadModels();
  }, []);

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8080/models/status");
        if (response.ok) {
          const status = await response.json();
          setModelStatus(status);
        }
      } catch (err) {
        console.error("Error checking model status:", err);
      }
    };

    checkModelStatus();
  }, [results]);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
  
    try {
      const response = await fetch("http://127.0.0.1:8080/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          validation_method: validationMethod,
        }),
      });
  
      if (!response.ok) {
        const errorMessage = await parseError(response);
        throw new Error(errorMessage);
      }
      
      // Re-check model status after training
      const statusResponse = await fetch("http://127.0.0.1:8080/models/status");
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        setModelStatus(status); // Update model status after training
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "An error occurred during training.");
      console.error("Training error:", err);
    } finally {
      setLoading(false);
    }
  };
  

  const handlePredict = async (formData) => {
    if (!modelStatus.kNN || modelStatus.kNN !== "trained") {
      setError("Model is not trained yet. Please train the model first.");
      return;
    }
  
    setLoading(true);
    setError(null);
    setPrediction(null);
  
    try {
      const { modelType, ...dataFields } = formData;
      const dataRow = Object.keys(defaultValues)
        .filter((key) => key !== "modelType")
        .map((key) => Number(dataFields[key]));
  
      const response = await fetch("http://127.0.0.1:8080/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data_row: dataRow,
          model_type: modelType,
        }),
      });
  
      if (!response.ok) {
        const errorMessage = await parseError(response);
        throw new Error(errorMessage);
      }
  
      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message || "An error occurred during prediction.");
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };
  

  const renderMetricValue = (value) => {
    if (typeof value === "number") {
      return (value * 100).toFixed(2) + "%";
    }
    return value;
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
      <main className="w-full max-w-4xl bg-white rounded-xl shadow-lg p-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-blue-600 mb-4">
            Diabetes Prediction Dashboard
          </h1>
          <p className="text-gray-600">
            Train and evaluate different machine learning models for diabetes prediction
          </p>
        </header>

        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Model Training
          </h2>
          <div className="flex items-center gap-4 mb-4">
            <select
              value={validationMethod}
              onChange={(e) => setValidationMethod(e.target.value)}
              className="p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white"
            >
              <option value="holdout">Holdout Validation</option>
              <option value="3-fold">3-Fold Cross Validation</option>
              <option value="10-fold">10-Fold Cross Validation</option>
              <option value="leave-one-out">Leave One Out</option>
            </select>

            <button
              onClick={handleTrain}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                     transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Training..." : "Train Models"}
            </button>
          </div>

          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {results && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Training Results
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(results).map(([model, metrics]) => (
                  <div key={model} className="bg-gray-50 rounded-lg p-4">
                    <h4 className="text-lg font-medium text-gray-700 mb-2">{model}</h4>
                    <div className="space-y-2">
                      {Object.entries(metrics).map(([metric, value]) => (
                        <div key={metric} className="flex justify-between text-sm">
                          <span className="text-gray-600 capitalize">
                            {metric.replace(/_/g, ' ')}:
                          </span>
                          <span className="font-medium">{renderMetricValue(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Make a Prediction
          </h2>
          <form onSubmit={handleSubmit(handlePredict)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(defaultValues)
                .filter(([key]) => key !== "modelType")
                .map(([key]) => (
                  <div key={key}>
                    <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-1">
                      {key}
                    </label>
                    <input
                      type="number"
                      step="any"
                      id={key}
                      {...register(key, {
                        required: "This field is required",
                        valueAsNumber: true,
                      })}
                      className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none 
                               focus:ring-2 focus:ring-blue-600"
                    />
                    {errors[key] && (
                      <p className="mt-1 text-sm text-red-600">{errors[key].message}</p>
                    )}
                  </div>
                ))}
            </div>

            <div>
              <label htmlFor="modelType" className="block text-sm font-medium text-gray-700 mb-1">
                Model Type
              </label>
              <select
                {...register("modelType", { required: "Model type is required" })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none 
                         focus:ring-2 focus:ring-blue-600 bg-white"
              >
                <option value="kNN" disabled={modelStatus.kNN !== "trained"}>
                  k-Nearest Neighbors (kNN) {modelStatus.kNN !== "trained" ? "(Not Trained)" : ""}
                </option>
                <option value="Bayesian" disabled={modelStatus.Bayesian !== "trained"}>
                  Naive Bayes {modelStatus.Bayesian !== "trained" ? "(Not Trained)" : ""}
                </option>
                <option value="SVM" disabled={modelStatus.SVM !== "trained"}>
                  Support Vector Machine (SVM) {modelStatus.SVM !== "trained" ? "(Not Trained)" : ""}
                </option>
                <option value="Neural" disabled={modelStatus.Neural !== "trained"}>
                  Neural Network {modelStatus.Neural !== "trained" ? "(Not Trained)" : ""}
                </option>
              </select>
              {errors.modelType && (
                <p className="mt-1 text-sm text-red-600">{errors.modelType.message}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 
                       transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
          </form>

          {prediction && (
            <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Prediction Result
              </h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-lg mb-2">
                  Diagnosis:{" "}
                  {prediction.prediction === 1 ? (
                    <span className="font-medium text-red-600">Positive (At Risk)</span>
                  ) : (
                    <span className="font-medium text-green-600">Negative (Not At Risk)</span>
                  )}
                </p>
                {prediction.probability && (
                  <p className="text-gray-700">
                    Confidence: {(prediction.probability[1] * 100).toFixed(2)}%
                  </p>
                )}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}