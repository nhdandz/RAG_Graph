import React from 'react';

export interface CardProps {
  title?: string;
  children: React.ReactNode;
  variant?: 'default' | 'military' | 'gold';
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  children,
  variant = 'default',
  className = ''
}) => {
  const variantClasses = {
    default: 'bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700',
    military: 'bg-white dark:bg-zinc-800 border-2 border-military-green-600',
    gold: 'bg-white dark:bg-zinc-800 gold-border'
  };

  return (
    <div className={`rounded-lg shadow-md p-6 transition-smooth hover:shadow-lg ${variantClasses[variant]} ${className}`}>
      {title && (
        <h3 className="text-xl font-bold mb-4 text-military-green-900 dark:text-military-green-400">
          {title}
        </h3>
      )}
      <div>{children}</div>
    </div>
  );
};
