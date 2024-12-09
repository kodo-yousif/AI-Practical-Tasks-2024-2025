import React, {useState} from 'react';
import {GAResponse} from '../types/sudoku';
import {CheckCircle, History, XCircle} from 'lucide-react';
import {SudokuBoard} from './SudokuBoard';
import {EmptyState} from "./EmptyState.tsx";

interface HistoryListProps {
    history: GAResponse[];
}

export const HistoryList: React.FC<HistoryListProps> = ({history}) => {
    const [selectedEntry, setSelectedEntry] = useState<GAResponse | null>(null);
    console.log(history);
    return (<div className = "bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden h-[500px]">
            <div className = "p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className = "text-lg font-semibold text-gray-900 dark:text-white">Solution History</h3>
            </div>
            <div className = "grid grid-cols-1 h-[calc(100%-60px)]">
                {selectedEntry ? (<div className = "p-4 h-full">
                        <div className = {"flex flex-row justify-between"}>
                            <button
                                onClick = {() => setSelectedEntry(null)}
                                className = "mb-4 text-sm text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300"
                            >
                                ← Back to History
                            </button>
                            <div className = "text-center mt-2">
                                <p className = "text-sm text-gray-600 dark:text-gray-400">
                                    Generation {selectedEntry.generation} Fitness: {selectedEntry.fitness.toFixed(4)}
                                </p>
                            </div>
                        </div>
                        <div className = "flex justify-center">
                            <div className = "transform scale-75 origin-center -my-10">
                                <SudokuBoard
                                    board = {selectedEntry.board}
                                    originalBoard = {selectedEntry.board}
                                    selectedCell = {null}
                                    onCellClick = {() => {
                                    }}
                                />
                            </div>
                        </div>
                    </div>) : (<div className = "overflow-y-auto p-4 space-y-3">
                        {history.length > 0 ? (history.map((entry, index) => (<button
                                    key = {index}
                                    onClick = {() => setSelectedEntry(entry)}
                                    className = "w-full text-left border dark:border-gray-700 rounded-lg p-3 space-y-2 bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                                >
                                    <div className = "flex justify-between items-center">
                    <span className = "text-sm font-medium text-gray-900 dark:text-white">
                      Generation {entry.generation}
                    </span>
                                        {entry.is_complete ? (<CheckCircle className = "w-5 h-5 text-green-500"/>) : (
                                            <XCircle className = "w-5 h-5 text-red-500"/>)}
                                    </div>
                                    <div className = "text-sm text-gray-600 dark:text-gray-400">
                                        Fitness: {entry.fitness}
                                    </div>
                                    {entry.message && (
                                        <p className = "text-xs text-gray-500 dark:text-gray-400">{entry.message}</p>)}
                                </button>))) : (<div className = "h-full justify-center">
                                <EmptyState
                                    title = "No solution history yet."
                                    message = "Start the GA solver to see the progress."
                                    icon = {<History className = "w-12 h-12 text-gray-400 dark:text-gray-500"/>}
                                />
                            </div>)}
                    </div>)}
            </div>
        </div>);
};