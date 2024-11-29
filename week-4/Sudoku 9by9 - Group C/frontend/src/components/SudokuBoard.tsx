import React from 'react';
import { SudokuCell } from './SudokuCell';
import { SudokuBoard as BoardType } from '../types/sudoku';

interface SudokuBoardProps {
  board: BoardType;
  originalBoard: BoardType;
  selectedCell: { row: number; col: number } | null;
  onCellClick: (row: number, col: number) => void;
}

export const SudokuBoard: React.FC<SudokuBoardProps> = ({
  board,
  originalBoard,
  selectedCell,
  onCellClick,
}) => {
  return (
    <div className="grid grid-cols-9 gap-0 bg-gray-300 p-0.5">
      {board.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <SudokuCell
            key={`${rowIndex}-${colIndex}`}
            value={cell}
            isOriginal={originalBoard[rowIndex][colIndex] !== 0}
            isSelected={
              selectedCell?.row === rowIndex && selectedCell?.col === colIndex
            }
            onClick={() => onCellClick(rowIndex, colIndex)}
            rowIndex={rowIndex}
            colIndex={colIndex}
          />
        ))
      )}
    </div>
  );
};
