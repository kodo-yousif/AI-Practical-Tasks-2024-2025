import React from 'react';
import { LineChart } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  message: string;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ 
  title, 
  message, 
  icon = <LineChart className="w-12 h-12 text-gray-400 dark:text-gray-500" />
}) => {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-6">
      {icon}
      <h3 className="mt-4 text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{message}</p>
    </div>
  );
};