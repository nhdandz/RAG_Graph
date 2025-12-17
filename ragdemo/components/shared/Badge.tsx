import React from 'react';

export interface BadgeProps {
  text: string;
  color?: 'green' | 'red' | 'gold' | 'blue' | 'gray';
  icon?: React.ReactNode;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  text,
  color = 'green',
  icon,
  className = ''
}) => {
  const colorClasses = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 border-green-300 dark:border-green-700',
    red: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 border-red-300 dark:border-red-700',
    gold: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 border-amber-300 dark:border-amber-700',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 border-blue-300 dark:border-blue-700',
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200 border-gray-300 dark:border-gray-700'
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full border ${colorClasses[color]} ${className}`}>
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <span>{text}</span>
    </span>
  );
};
