import React from 'react';

function SolutionSection({ solutions, selectedGeneration, onSliderChange }) {
  if (solutions.length === 0) return null;

  const selectedSolution = solutions[selectedGeneration];

  return (
    <div className="bg-white shadow-lg rounded-xl p-8">
      <h2 className="text-xl font-bold text-gray-800 mb-4">
        Select a Generation:
      </h2>
      <input
        type="range"
        min="0"
        max={solutions.length - 1}
        value={selectedGeneration}
        onChange={(e) => onSliderChange(Number(e.target.value))}
        className="w-full mt-4"
      />
      <div className="mt-4 text-center">
        <span className="font-semibold">
          Generation {selectedSolution.generation} - Fitness:{' '}
          {selectedSolution.fitness}
        </span>
      </div>
      <div className="mt-6 text-center">
        <h3 className="text-xl font-bold mb-4 text-gray-800">
          Selected Generation {selectedSolution.generation}
        </h3>
        {selectedSolution.image && (
          <img
            src={`data:image/png;base64,${selectedSolution.image}`}
            alt="Solved Puzzle"
            className="rounded-lg shadow-md mx-auto"
          />
        )}
      </div>
    </div>
  );
}

export default SolutionSection;
