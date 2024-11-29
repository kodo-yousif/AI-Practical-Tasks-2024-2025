import React, { useState } from 'react';
import { SudokuContainer } from './components/SudokuContainer';
import { GAControls } from './components/GAControls';
import { SolutionProgress } from './components/SolutionProgress';
import { BestSolution } from './components/BestSolution';
import { ThemeToggle } from './components/ThemeToggle';
import { HistoryList } from './components/HistoryList';
import { ThemeProvider } from './contexts/ThemeContext';
import { startGASolver } from './services/api';
import type { SudokuBoard as BoardType, GAResponse } from './types/sudoku';

function App() {
  const [board, setBoard] = useState<BoardType>([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  ]);

  const [originalBoard, setOriginalBoard] = useState<BoardType>(board);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<GAResponse[]>([]);
  const [graphData, setGraphData] = useState<GAResponse[]>([]);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [selectedNumber, setSelectedNumber] = useState<number | null>(null);

  const [settings, setSettings] = useState({
    populationSize: 1000,
    generations: 1000,
    mutationRate: 0.1,
    maxNoImprovement: 100,
  });

  const handleCellClick = (row: number, col: number) => {
    if (originalBoard[row][col] !== 0) return;

    if (selectedCell?.row === row && selectedCell?.col === col) {
      setSelectedCell(null);
    } else {
      setSelectedCell({ row, col });
      if (selectedNumber !== null) {
        handleNumberSelect(selectedNumber, row, col);
      }
    }
  };

  const handleNumberSelect = (number: number, row?: number, col?: number) => {
    if (number == -1) {
      setBoard(originalBoard);
      setSelectedCell(null);
      setSelectedNumber(null);
      return;
    }
    if (row !== undefined && col !== undefined) {
      const newBoard = board.map((r, i) =>
        i === row ? r.map((c, j) => (j === col ? number : c)) : r
      );
      setBoard(newBoard);
      setSelectedCell(null);
      setSelectedNumber(null);
    } else if (selectedCell) {
      const newBoard = board.map((r, i) =>
        i === selectedCell.row
          ? r.map((c, j) => (j === selectedCell.col ? number : c))
          : r
      );
      setBoard(newBoard);
      setSelectedCell(null);
      setSelectedNumber(null);
    } else {
      setSelectedNumber(number);
    }
  };

  const handleStart = async () => {
    setIsLoading(true);
    try {
      const response = await startGASolver({
        board,
        population_size: settings.populationSize,
        generations: settings.generations,
        mutation_rate: settings.mutationRate,
        max_no_improvement: settings.maxNoImprovement,
      });

      setGraphData(response.graph_data);
      setHistory(response.history.slice(0, -1));
    } catch (error) {
      console.error('Error solving Sudoku:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSettingsChange = (setting: string, value: number) => {
    setSettings((prev) => ({ ...prev, [setting]: value }));
  };

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 py-8 transition-colors">
        <ThemeToggle />
        <div className="mx-auto px-4">
          <h1 className="text-3xl font-bold text-center mb-8 text-gray-900 dark:text-white">
            Sudoku Solver
          </h1>

          <div className = "grid grid-cols-12 gap-4">
            <div className = "grid grid-cols-1 col-span-6 space-y-4">
              <div className = "col-span-1 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors">
                <GAControls
                    onStart = {handleStart}
                    isLoading = {isLoading}
                    populationSize = {settings.populationSize}
                    generations = {settings.generations}
                    mutationRate = {settings.mutationRate}
                    maxNoImprovement = {settings.maxNoImprovement}
                    onSettingsChange = {handleSettingsChange}
                />
              </div>
              <div className={"col-span-1"}>
                <SolutionProgress history={graphData} />
              </div>
            </div>

            <div className = "col-span-6">
              <SudokuContainer
                  board = {board}
                  originalBoard = {originalBoard}
                  selectedCell = {selectedCell}
                  selectedNumber = {selectedNumber}
                  onCellClick = {handleCellClick}
                  onNumberSelect = {handleNumberSelect}
              />
            </div>

            <div className = "col-span-12 grid grid-cols-8 gap-4">
              <div className = "col-span-4">
                <HistoryList history = {history}/>
              </div>
              <div className = "col-span-4">
                <BestSolution history = {history}/>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;