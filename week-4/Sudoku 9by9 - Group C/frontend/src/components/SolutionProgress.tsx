import React from 'react';
import { GAResponse } from '../types/sudoku';
import { SolutionGraph } from './SolutionGraph';
import { EmptyState } from './EmptyState';

interface SolutionProgressProps {
  history: GAResponse[];
}

export const SolutionProgress: React.FC<SolutionProgressProps> = ({ history }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors h-ma">
      <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
        Solution Progress
      </h2>
      {history.length > 0 ? (
        <SolutionGraph history={history} />
      ) : (
        <div className="h-64">
          <EmptyState
            title="No Solution Data Yet"
            message="Start the GA solver to see the solution progress graph"
          />
        </div>
      )}
    </div>
  );
};