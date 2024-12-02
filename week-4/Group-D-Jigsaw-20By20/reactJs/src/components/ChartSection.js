import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

function ChartSection({ solutions }) {
  if (solutions.length === 0) return null;

  const chartData = {
    labels: solutions.map((_, index) => `Gen ${index}`),
    datasets: [
      {
        label: 'Fitness Evolution',
        data: solutions.map((solution) => solution.fitness),
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        fill: true,
      },
    ],
  };

  return (
    <div className="bg-white shadow-lg rounded-xl p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Fitness Evolution
      </h2>
      <Line data={chartData} />
    </div>
  );
}

export default ChartSection;
