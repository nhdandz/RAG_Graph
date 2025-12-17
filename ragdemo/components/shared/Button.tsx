import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  icon,
  children,
  className = '',
  disabled,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center gap-2 font-semibold rounded-md transition-smooth military-button-hover disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none';

  const variantClasses = {
    primary: 'bg-military-green-600 hover:bg-military-green-900 text-white shadow-md',
    secondary: 'bg-accent-blue hover:bg-blue-600 text-white shadow-md',
    danger: 'bg-vietnam-red-600 hover:bg-vietnam-red-700 text-white shadow-md',
    outline: 'border-2 border-military-green-600 text-military-green-600 hover:bg-military-green-600 hover:text-white'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
    </button>
  );
};
