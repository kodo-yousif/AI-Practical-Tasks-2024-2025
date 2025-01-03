import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";

function App() {
  const [validationMethod, setValidationMethod] = useState("holdout");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

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

  // Load models when the app starts
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
        const errorText = await response.text();
        throw new Error(errorText);
      }
      const data = await response.json();
      console.log(data)
      setResults(data);
    } catch (err) {
      setError(err.message || "An error occurred.");
      console.error("Training error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Extract model type and create data row without it
      const { modelType, ...dataFields } = formData;

      // Convert form values to numbers and create array in correct order
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
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const result = await response.json();
      console.log(result);
      setPrediction(result);
    } catch (err) {
      setError(err.message || "An error occurred.");
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <header className="text-center w-full max-w-4xl">
        <h1 className="text-3xl font-bold text-blue-600 mb-6">
          Model Training Dashboard
        </h1>

        <div className="mb-4">
          <select
            value={validationMethod}
            onChange={(e) => setValidationMethod(e.target.value)}
            className="p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            <option value="holdout">Holdout Validation</option>
            <option value="3-fold">3-Fold Cross Validation</option>
            <option value="10-fold">10-Fold Cross Validation</option>
            <option value="leave-one-out">Leave One Out</option>
          </select>
        </div>

        <button
          onClick={handleTrain}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                   transition duration-300 mb-6 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Training..." : "Train Models"}
        </button>

        {error && <p className="text-red-500 text-lg mb-4">{error}</p>}

        {results && (
          <div className="results bg-white p-6 rounded-lg shadow-md mt-6 w-full">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Training Results
            </h2>
            {Object.entries(results).map(([model, modelResults]) => (
              <div key={model} className="result-block mb-4">
                <h3 className="text-xl font-medium text-gray-700">{model}</h3>
                <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto">
                  {JSON.stringify(modelResults, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}

        <h2 className="text-2xl font-semibold text-gray-800 mt-8 mb-4">
          Make a Prediction
        </h2>
        <form
          onSubmit={handleSubmit(handlePredict)}
          className="w-full bg-white p-6 rounded-lg shadow-md"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
            {Object.entries(defaultValues)
              .filter(([key]) => key !== "modelType")
              .map(([key, value]) => (
                <div key={key} className="flex flex-col">
                  <label
                    htmlFor={key}
                    className="text-gray-700 font-medium mb-2"
                  >
                    {key}:
                  </label>
                  <input
                    type="number"
                    step="any"
                    id={key}
                    {...register(key, {
                      required: "This field is required",
                      valueAsNumber: true,
                    })}
                    className="p-2 border border-gray-300 rounded-lg focus:outline-none 
                             focus:ring-2 focus:ring-blue-600"
                  />
                  {errors[key] && (
                    <span className="text-red-500 text-sm mt-1">
                      {errors[key].message}
                    </span>
                  )}
                </div>
              ))}
          </div>

          <div className="mb-6">
            <label
              htmlFor="modelType"
              className="text-gray-700 font-medium mb-2"
            >
              Model Type:
            </label>
            <select
              {...register("modelType", { required: "Model type is required" })}
              className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none 
                       focus:ring-2 focus:ring-blue-600"
            >
              <option value="kNN">k-Nearest Neighbors (kNN)</option>
              <option value="Bayesian">Naive Bayes</option>
              <option value="SVM">Support Vector Machine (SVM)</option>
              <option value="Neural Network">Neural Network</option>
            </select>
            {errors.modelType && (
              <span className="text-red-500 text-sm mt-1">
                {errors.modelType.message}
              </span>
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
          <div className="prediction bg-white p-6 rounded-lg shadow-md mt-6 w-full">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Prediction Result
            </h2>
            <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto">
              {JSON.stringify(prediction, null, 2)}
            </pre>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
