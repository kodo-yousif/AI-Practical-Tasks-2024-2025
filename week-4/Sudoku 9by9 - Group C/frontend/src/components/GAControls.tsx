import React from 'react';
import {Settings, Play} from 'lucide-react';

interface GAControlsProps {
    onStart: () => void;
    isLoading: boolean;
    populationSize: number;
    generations: number;
    mutationRate: number;
    maxNoImprovement: number;
    onSettingsChange: (setting: string, value: number) => void;
}

export const GAControls: React.FC<GAControlsProps> = ({
                                                          onStart,
                                                          isLoading,
                                                          populationSize,
                                                          generations,
                                                          mutationRate,
                                                          maxNoImprovement,
                                                          onSettingsChange,
                                                      }) => {
    return (
        <div className = "space-y-4">
            <div className = "flex items-center space-x-2">
                <Settings className = "w-5 h-5 text-blue-500"/>
                <h3 className = "text-lg font-semibold text-gray-900 dark:text-white">GA Settings</h3>
            </div>

            <div className = "grid grid-cols-8 gap-4">
                <div className = {"col-span-4 grid grid-cols-1 space-y-4"}>
                    <div className = "grid grid-cols-2 col-span-1 items-center">
                        <div className = {"col-span-1"}>
                            <label className = "text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Population Size
                            </label>
                        </div>
                        <div className = {"col-span-1"}>
                            <input
                                type = "number"
                                value = {populationSize}
                                onChange = {(e) => onSettingsChange('populationSize', parseInt(e.target.value))}
                                className = "w-full rounded-sm border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm focus:border-blue-500 focus:ring-blue-500 pl-2"
                            />
                        </div>
                    </div>

                    <div className = "grid grid-cols-2 col-span-1 items-center">
                        <div className = {"col-span-1"}>
                            <label className = "text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Generations
                            </label>
                        </div>
                        <div className = {"col-span-1"}>
                            <input
                                type = "number"
                                value = {generations}
                                onChange = {(e) => onSettingsChange('generations', parseInt(e.target.value))}
                                className = "w-full rounded-sm border-gray-400 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm focus:border-blue-500 focus:ring-blue-500 pl-2"
                            />
                        </div>
                    </div>
                </div>

                <div className = {"col-span-4 grid grid-cols-1 space-y-4"}>
                    <div className = "grid grid-cols-2 col-span-1 items-center">
                        <div className = {"col-span-1"}>
                            <label className = "text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Mutation Rate
                            </label>
                        </div>
                        <div className = {"col-span-1"}>
                            <input
                                type = "number"
                                value = {mutationRate}
                                step = "0.01"
                                onChange = {(e) => onSettingsChange('mutationRate', parseFloat(e.target.value))}
                                className = "w-full rounded-sm border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm focus:border-blue-500 focus:ring-blue-500 pl-2"
                            />
                        </div>
                    </div>

                    <div className = "grid grid-cols-2 col-span-1 items-center">
                        <div className = {"col-span-1"}>
                            <label className = "text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Max No Improvement
                            </label>
                        </div>
                        <div className = {"col-span-1"}>
                            <input
                                type = "number"
                                value = {maxNoImprovement}
                                onChange = {(e) => onSettingsChange('maxNoImprovement', parseInt(e.target.value))}
                                className = "w-full rounded-sm border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm focus:border-blue-500 focus:ring-blue-500 pl-2"
                            />
                        </div>
                    </div>
                </div>
            </div>

            <button
                onClick = {onStart}
                disabled = {isLoading}
                className = "w-full bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 disabled:bg-blue-300 flex items-center justify-center space-x-2"
            >
                <Play className = "w-4 h-4"/>
                <span>{isLoading ? 'Solving...' : 'Start GA Solver'}</span>
            </button>
        </div>
    );
};