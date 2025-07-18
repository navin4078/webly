import React from 'react';
import { Bot, Menu, X, Wifi, WifiOff } from 'lucide-react';

const Header = ({ 
  isConnected, 
  connectionStatus, 
  showSidebar, 
  onToggleSidebar 
}) => {
  return (
    <header className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-4">
      {/* Left side */}
      <div className="flex items-center gap-3">
        {/* Sidebar toggle */}
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          title={showSidebar ? 'Hide sidebar' : 'Show sidebar'}
        >
          {showSidebar ? (
            <X className="w-5 h-5 text-gray-600" />
          ) : (
            <Menu className="w-5 h-5 text-gray-600" />
          )}
        </button>

        {/* Logo */}
        <div className="flex items-center gap-2">
          <Bot className="w-6 h-6 text-orange-500" />
          <span className="font-semibold text-gray-900">Webly</span>
        </div>
      </div>

      {/* Right side - Connection status */}
      <div className="flex items-center gap-2 text-sm text-gray-600">
        {isConnected ? (
          <Wifi className="w-4 h-4 text-green-500" />
        ) : (
          <WifiOff className="w-4 h-4 text-red-500" />
        )}
        <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </header>
  );
};

export default Header;