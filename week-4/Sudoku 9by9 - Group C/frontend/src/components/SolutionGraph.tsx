import React, {useMemo} from 'react';
import {LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot} from 'recharts';
import {GAResponse} from '../types/sudoku';
import {useTheme} from '../contexts/ThemeContext';

interface SolutionGraphProps {
    history: GAResponse[];
}

export const SolutionGraph: React.FC<SolutionGraphProps> = ({history}) => {
    const {theme} = useTheme();
    const isDark = theme === 'dark';

    const data = history.map((entry, index) => ({
        x: index,
        y: entry,
    }));

    const bestSolution = useMemo(() => {
        return history.reduce((best, current, index) =>
                current > best.value ? {value: current, generation: index} : best
            , {value: history[0], generation: 0});
    }, [history]);

    console.log(bestSolution)
    console.log(data)
    return (
        <div className = "w-full h-64">
            <ResponsiveContainer>
                <LineChart data = {data} margin = {{top: 5, right: 20, bottom: 35, left: 10}}>
                    <CartesianGrid
                        strokeDasharray = "3 3"
                        stroke = {isDark ? '#374151' : '#E5E7EB'}
                    />
                    <Tooltip
                        contentStyle = {{
                            backgroundColor: isDark ? '#1F2937' : '#FFFFFF',
                            border: 'none',
                            borderRadius: '0.375rem',
                            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                        }}
                        labelStyle = {{color: isDark ? '#9CA3AF' : '#4B5563'}}
                        itemStyle = {{color: isDark ? '#E5E7EB' : '#1F2937'}}
                    />
                    <Line
                        type = "monotone"
                        dataKey = "y"
                        stroke = "#3b82f6"
                        strokeWidth = {2}
                        dot = {false}
                    />
                    {bestSolution && (
                        <ReferenceDot
                            x = {bestSolution.generation}
                            y = {bestSolution.value}
                            r = {4}
                            fill = "green"
                            stroke = "none"
                        />
                    )}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};