import React from 'react';
import { SudokuBoard } from './SudokuBoard';
import { NumberButtons } from './NumberButtons';
import type { SudokuBoard as BoardType } from '../types/sudoku';

interface SudokuContainerProps {
  board: BoardType;
  originalBoard: BoardType;
  selectedCell: { row: number; col: number } | null;
  selectedNumber: number | null;
  onCellClick: (row: number, col: number) => void;
  onNumberSelect: (number: number) => void;
}

export const SudokuContainer: React.FC<SudokuContainerProps> = ({
  board,
  originalBoard,
  selectedCell,
  selectedNumber,
  onCellClick,
  onNumberSelect,
}) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 justify-center items-center flex transition-colors h-full">
      <div className="flex flex-col items-center">
        <SudokuBoard
          board={board}
          originalBoard={originalBoard}
          selectedCell={selectedCell}
          onCellClick={onCellClick}
        />
        <NumberButtons
          onNumberSelect={onNumberSelect}
          selectedNumber={selectedNumber}
        />
      </div>
    </div>
  );
};