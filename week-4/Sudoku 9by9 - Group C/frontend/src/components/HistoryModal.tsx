import React from 'react';
import { X } from 'lucide-react';
import { GAResponse } from '../types/sudoku';

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: GAResponse[];
}

export const HistoryModal: React.FC<HistoryModalProps> = ({
  isOpen,
  onClose,
  history,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">Solution Progress</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full text-gray-500 dark:text-gray-400"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="space-y-4">
          {history.map((entry, index) => (
            <div
              key={index}
              className="border dark:border-gray-700 rounded-lg p-4 space-y-2"
            >
              <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                <span>Generation: {entry.generation}</span>
                <span>Fitness: {entry.fitness}</span>
              </div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Status: {entry.is_complete ? 'Complete' : 'In Progress'}
              </p>
              {entry.message && (
                <p className="text-sm text-gray-700 dark:text-gray-300">{entry.message}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};