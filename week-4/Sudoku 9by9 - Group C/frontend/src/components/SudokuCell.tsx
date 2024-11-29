import React from 'react';
import { clsx } from 'clsx';

interface SudokuCellProps {
  value: number;
  isOriginal: boolean;
  isSelected: boolean;
  onClick: () => void;
}

export const SudokuCell: React.FC<SudokuCellProps> = ({
  value,
  isOriginal,
  isSelected,
  onClick,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={isOriginal}
      className={clsx(
        'w-12 h-12 text-center border text-lg font-medium',
        'focus:outline-none transition-colors duration-200',
        isSelected && !isOriginal && 'ring-2 ring-blue-500',
        isOriginal 
          ? 'bg-gray-100 dark:bg-gray-700 font-bold cursor-not-allowed text-gray-900 dark:text-gray-100' 
          : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-200',
        'border-gray-200 dark:border-gray-600'
      )}
    >
      {value || ''}
    </button>
  );
};