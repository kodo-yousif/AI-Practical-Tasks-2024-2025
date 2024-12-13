/* eslint-disable react/prop-types */
import { useEffect, useState } from "react";
import "./App.css";

const SolutionList = ({ solutions, onSelect, defaultSolution }) => {
  const [selectedGeneration, setSelectedGeneration] = useState(null);

  useEffect(() => {
    // Automatically select the best solution when `defaultSolution` is updated
    if (defaultSolution) {
      const bestSolution = solutions.find(
        (entry) => entry.solution === defaultSolution
      );
      if (bestSolution) {
        setSelectedGeneration(bestSolution.generation);
        onSelect(bestSolution.solution);
      }
    }
  }, [defaultSolution, solutions, onSelect]);

  const handleGenerationChange = (e) => {
    const selected = solutions.find(
      (entry) => entry.generation === Number(e.target.value)
    );
    if (selected) {
      setSelectedGeneration(selected.generation);
      onSelect(selected.solution); // Update the selected solution in App
    }
  };

  return (
    <div className="solution-list">
      <h2 className="subtitle">{selectedGeneration && "Saved Solutions"}</h2>
      {solutions.length > 0 ? (
        <div className="select-container">
          <label htmlFor="generation-select">Choose a Generation</label>
          <select
            id="generation-select"
            onChange={handleGenerationChange}
            value={selectedGeneration || ""}
          >
            <option value="" disabled>
              Select a generation
            </option>
            {solutions.map((entry, index) => (
              <option key={index} value={entry.generation}>
                Generation {entry.generation} - Fitness: {entry.fitness}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <p>No solutions available...</p>
      )}
    </div>
  );
};

export default SolutionList;