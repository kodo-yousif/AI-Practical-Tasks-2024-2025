import React from 'react';

function History({ history, onHistoryClick }) {
  if (history.length === 0) return null;

  return (
    <div className="bg-white shadow-lg rounded-xl p-8 h-96">
      <h2 className="text-xl font-bold text-gray-800 mb-4">History</h2>
      <div className="overflow-y-auto h-80">
        {history.map((entry, historyIndex) => (
          <div key={historyIndex} className="mb-6">
            <h3 className="font-semibold text-gray-800 mb-2">
              Configuration from {new Date(entry.timestamp).toLocaleString()}
            </h3>
            <ul className="space-y-2">
              {Array.isArray(entry.solutions) &&
                entry.solutions.map((solution, generationIndex) => (
                  <li
                    key={generationIndex}
                    className="p-2 bg-gray-100 rounded-lg shadow-md cursor-pointer hover:bg-gray-200"
                    onClick={() => onHistoryClick(historyIndex, generationIndex)}
                  >
                    <strong>Gen {generationIndex}:</strong> Fitness{' '}
                    {solution.fitness}
                    {solution.image && (
                      <img
                        src={`data:image/png;base64,${solution.image}`}
                        alt={`Gen ${generationIndex}`}
                        className="rounded-lg shadow-md mt-2 mx-auto max-h-20"
                      />
                    )}
                  </li>
                ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default History;
