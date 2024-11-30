import React from 'react';
import clsx from 'clsx';

interface SudokuCellProps {
  value: number;
  isOriginal: boolean;
  isSelected: boolean;
  onClick: () => void;
  rowIndex: number;
  colIndex: number;
}

export const SudokuCell: React.FC<SudokuCellProps> = ({
  value,
  isOriginal,
  isSelected,
  onClick,
  rowIndex,
  colIndex,
}) => {
  const isTopBorder = rowIndex % 3 === 0;
  const isLeftBorder = colIndex % 3 === 0;
  const isBottomBorder = rowIndex === 8;
  const isRightBorder = colIndex === 8;

  return (
    <button
      onClick={onClick}
      disabled={isOriginal}
      className={clsx(
        'w-12 h-12 text-center border text-lg font-medium',
        'focus:outline-none transition-colors duration-200',
        isSelected && !isOriginal && 'bg-blue-200 dark:bg-blue-800',
        isOriginal
          ? 'bg-gray-100 dark:bg-gray-700 font-bold cursor-not-allowed text-gray-900 dark:text-gray-100'
          : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-200',
        'border-gray-300 dark:border-gray-100',
        isTopBorder && 'border-t-4',
        isLeftBorder && 'border-l-4',
        isBottomBorder && 'border-b-4',
        isRightBorder && 'border-r-4'
      )}
    >
      {value || ''}
    </button>
  );
};
