import { useState } from "react";
import { useForm } from "react-hook-form";

export default function App() {
  const [validationMethod, setValidationMethod] = useState("holdout");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const [trainingResults, setTrainingResults] = useState(false);
  const [predictingResults, setPredictingResults] = useState(false);
  const [bestModel, setBestModel] = useState("");
  const [selectedModel, setSelectedModel] = useState(""); // New state for selected model

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

      const data = await response.json();
      setResults(data[0]);
      setBestModel(data[1]?.name);
      setSelectedModel(data[1]?.name);
    } catch (err) {
      setError(err.message || "An error occurred during training.");
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
          model_type: selectedModel,
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

  const models = [
    { value: "kNN", label: "k-Nearest Neighbors" },
    { value: "Bayesian", label: "Naive Bayes" },
    { value: "SVM", label: "Support Vector Machine" },
    { value: "Neural", label: "Neural Network" },
  ];

  const sortedModels = models
    .map((model) => ({
      ...model,
      isRecommended: model.value === bestModel,
    }))
    .sort((a, b) => b.isRecommended - a.isRecommended);

  return (
    <div className="h-[90vh] m-auto fixed inset-0 flex flex-col items-center justify-center bg-slate-900 px-6 overflow-hidden">
      <main className="w-full  bg-slate-300 rounded-xl shadow-lg p-6 overflow-x-hidden overflow-y-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl text-start font-bold text-blue-950 mb-4">
            Diabetes Prediction
          </h1>
        </header>

        <section className="mb-8 border-2 p-6 rounded-lg border-slate-400">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Model Training
            </h2>

            <button
              className={`${
                results ? "cursor-pointer" : "cursor-default"
              } transition-all ${trainingResults ? "rotate-180" : ""}`}
              onClick={() => {
                setTrainingResults(!trainingResults);
              }}
              disabled={!results}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="42px"
                height="42px"
                viewBox="0 0 24 24"
                fill="none"
              >
                <path
                  d="M12.7071 14.7071C12.3166 15.0976 11.6834 15.0976 11.2929 14.7071L6.29289 9.70711C5.90237 9.31658 5.90237 8.68342 6.29289 8.29289C6.68342 7.90237 7.31658 7.90237 7.70711 8.29289L12 12.5858L16.2929 8.29289C16.6834 7.90237 17.3166 7.90237 17.7071 8.29289C18.0976 8.68342 18.0976 9.31658 17.7071 9.70711L12.7071 14.7071Z"
                  fill="#000000"
                />
              </svg>
            </button>
          </div>

          <div className={`flex items-center gap-4 mb-4`}>
            <select
              value={validationMethod}
              onChange={(e) => setValidationMethod(e.target.value)}
              className="p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white"
            >
              <option value="holdout">Holdout Validation</option>
              <option value="3-fold">3-Fold Cross Validation</option>
              <option value="10-fold">10-Fold Cross Validation</option>
            </select>

            <button
              onClick={handleTrain}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Training..." : "Train Models"}
            </button>
          </div>

          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg
                    className="h-5 w-5 text-red-400"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {results && trainingResults && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 transition-all">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Training Results
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {Object.entries(results).map(([model, metrics]) => (
                  <div key={model} className="bg-gray-50 rounded-lg p-4">
                    <h4 className="text-lg font-medium text-gray-700 mb-2">
                      {model}
                    </h4>
                    <div className="space-y-2">
                      {Object.entries(metrics).map(([metric, value]) => (
                        <div
                          key={metric}
                          className="flex justify-between text-sm"
                        >
                          <span className="text-gray-600 capitalize">
                            {metric.replace(/_/g, " ")}:
                          </span>
                          <span className="font-medium">
                            {renderMetricValue(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="p-6 border-2 rounded-lg border-slate-400">
          <div className="flex items-center justify-between ">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Make a Prediction
            </h2>

            <button
              className={`${
                !predictingResults ? "cursor-pointer" : "cursor-default"
              } transition-all ${predictingResults ? "rotate-180" : ""}`}
              onClick={() => {
                setPredictingResults(!predictingResults);
              }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="42px"
                height="42px"
                viewBox="0 0 24 24"
                fill="none"
              >
                <path
                  d="M12.7071 14.7071C12.3166 15.0976 11.6834 15.0976 11.2929 14.7071L6.29289 9.70711C5.90237 9.31658 5.90237 8.68342 6.29289 8.29289C6.68342 7.90237 7.31658 7.90237 7.70711 8.29289L12 12.5858L16.2929 8.29289C16.6834 7.90237 17.3166 7.90237 17.7071 8.29289C18.0976 8.68342 18.0976 9.31658 17.7071 9.70711L12.7071 14.7071Z"
                  fill="#000000"
                />
              </svg>
            </button>
          </div>

          <form onSubmit={handleSubmit(handlePredict)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {Object.entries(defaultValues)
                .filter(([key]) => key !== "modelType")
                .map(([key]) => (
                  <div key={key}>
                    <label
                      htmlFor={key}
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
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
                      <p className="mt-1 text-sm text-red-600">
                        {errors[key].message}
                      </p>
                    )}
                  </div>
                ))}
            </div>

            <div className="flex justify-between items-end">
              <div>
                <label
                  htmlFor="modelType"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Model Type
                </label>
                <select
                  onChange={(e) => setSelectedModel(e.target.value)} 
                  value={selectedModel}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white"
                >
                  {sortedModels.map((model) => (
                    <option
                      key={model.value}
                      value={model.value}
                      className={
                        model.isRecommended ? "bg-green-500 text-white" : ""
                      }
                    >
                      {model.label}
                      {model.isRecommended && " (Recommended)"}
                    </option>
                  ))}
                </select>
              </div>

              {errors.modelType && (
                <p className="mt-1 text-sm text-red-600">
                  {errors.modelType.message}
                </p>
              )}

              <button
                type="submit"
                disabled={loading}
                className=" px-12 h-12 font-bold bg-amber-500 text-white rounded-lg hover:bg-green-700 
                       transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Predicting..." : "Predict"}
              </button>
            </div>
          </form>

          {prediction && predictingResults && (
            <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Prediction Result
              </h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-lg mb-2">
                  Diagnosis:{" "}
                  {prediction.prediction === 1 ? (
                    <span className="font-medium text-red-600">
                      Positive (At Risk)
                    </span>
                  ) : (
                    <span className="font-medium text-green-600">
                      Negative (Not At Risk)
                    </span>
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
