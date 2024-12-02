import React from 'react';
import {clsx} from 'clsx';

interface NumberButtonsProps {
    onNumberSelect: (number: number) => void;
    selectedNumber: number | null;
}

export const NumberButtons: React.FC<NumberButtonsProps> = ({
                                                                onNumberSelect, selectedNumber,
                                                            }) => {
    const numbers = Array.from({length: 9}, (_, i) => i + 1);

    return (
        <div>
            <div className = "flex gap-2.5 justify-between mt-2 w-fit">
                {numbers.map((number) => (<button
                    key = {number}
                    onClick = {() => onNumberSelect(number)}
                    className = {clsx('w-10 h-10 rounded-md font-semibold transition-colors', selectedNumber === number ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600')}
                >
                    {number}
                </button>))}
            </div>

            <div className = "flex gap-3 justify-between mt-2 w-full">
                <button
                    key = {'0'}
                    onClick = {() => onNumberSelect(0)}
                    className = {clsx('w-full h-10 rounded-md font-semibold transition-colors', selectedNumber === 0 ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600')}
                >
                    {'Delete'}
                </button>

                <button
                    key = {'clear'}
                    onClick = {() => onNumberSelect(-1)}
                    className = {clsx('w-full h-10 rounded-md font-semibold transition-colors', selectedNumber === -1 ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600')}
                >
                    {'Clear'}
                </button>
            </div>
        </div>);
};