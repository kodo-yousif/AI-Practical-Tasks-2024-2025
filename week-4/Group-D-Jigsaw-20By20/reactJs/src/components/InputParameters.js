import React from 'react';

function InputParameters({
  image,
  onImageUpload,
  gridSize,
  setGridSize,
  populationSize,
  setPopulationSize,
  generations,
  setGenerations,
  mutationRate,
  setMutationRate,
  onSubmit,
  loading,
}) {
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onImageUpload(e.target.files[0]);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-xl p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Input Parameters</h2>
      <div className="mb-6">
        <label className="block text-gray-700 font-semibold mb-2">
          Upload Image:
        </label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
        />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-gray-700 font-semibold mb-1">
            Grid Size:
          </label>
          <input
            type="number"
            value={gridSize}
            onChange={(e) => setGridSize(Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          />
        </div>
        <div>
          <label className="block text-gray-700 font-semibold mb-1">
            Population Size:
          </label>
          <input
            type="number"
            value={populationSize}
            onChange={(e) => setPopulationSize(Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          />
        </div>
        <div>
          <label className="block text-gray-700 font-semibold mb-1">
            Generations:
          </label>
          <input
            type="number"
            value={generations}
            onChange={(e) => setGenerations(Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          />
        </div>
        <div>
          <label className="block text-gray-700 font-semibold mb-1">
            Mutation Rate:
          </label>
          <input
            type="number"
            step="0.01"
            value={mutationRate}
            onChange={(e) => setMutationRate(Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm"
          />
        </div>
      </div>
      <button
        onClick={onSubmit}
        className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-bold hover:bg-blue-700 transition-all shadow-md"
      >
        {loading ? 'Solving...' : 'Solve Puzzle'}
      </button>
    </div>
  );
}

export default InputParameters;
