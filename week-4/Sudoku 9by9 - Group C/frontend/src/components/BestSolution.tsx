import React from 'react';
import {Trophy} from 'lucide-react';
import {SudokuBoard} from './SudokuBoard';
import {GAResponse} from '../types/sudoku';
import {EmptyState} from './EmptyState';

interface BestSolutionProps {
    history: GAResponse[];
}

export const BestSolution: React.FC<BestSolutionProps> = ({history}) => {
    const bestSolution = history.length > 0
        ? history.reduce((best, current) =>
            current.fitness > best.fitness ? current : best
        )
        : null;

    return (
        <div className = "bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors h-[500px]">
            <div className = "flex items-center space-x-2 mb-4">
                <Trophy className = "w-5 h-5 text-yellow-500"/>
                <h2 className = "text-xl font-semibold text-gray-900 dark:text-white">
                    Best Solution
                </h2>
                {bestSolution && (
                    <span className = "text-sm text-gray-500 dark:text-gray-400">
            (Fitness: {bestSolution.fitness})
          </span>
                )}
            </div>

            {bestSolution ? (
                <div className = "flex justify-center">
                    <div className = "transform scale-75 origin-center -my-8">
                        <SudokuBoard
                            board = {bestSolution.board}
                            originalBoard = {bestSolution.board}
                            selectedCell = {null}
                            onCellClick = {() => {
                            }}
                        />
                    </div>
                </div>
            ) : (
                <div className = "h-full justify-center">
                    <EmptyState
                        title = "No Solution Yet"
                        message = "Start the GA solver to see the best solution"
                        icon = {<Trophy className = "w-12 h-12 text-gray-400 dark:text-gray-500"/>}
                    />
                </div>
            )}
        </div>
    );
}