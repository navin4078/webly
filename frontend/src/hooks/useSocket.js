import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';
import { toast } from 'react-hot-toast';

export const useSocket = (sessionId) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const [lastActivity, setLastActivity] = useState(Date.now());
  
  // Initialize socket connection
  const initializeSocket = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }
    
    console.log('🔌 Initializing Socket.IO connection...');
    setConnectionStatus('connecting');
    
    const serverUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    socketRef.current = io(serverUrl, {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      maxReconnectionAttempts: 5,
      forceNew: false,
    });
    
    const socket = socketRef.current;
    
    // Connection events
    socket.on('connect', () => {
      console.log('🔌 Connected to FastAPI server via Socket.IO');
      setIsConnected(true);
      setConnectionStatus('connected');
      setLastActivity(Date.now());
      
      // Join session room
      if (sessionId) {
        socket.emit('join_room', { session_id: sessionId });
      }
      
      // Clear any reconnection timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      toast.success('Connected to server', {
        duration: 2000,
        position: 'bottom-right',
      });
    });
    
    socket.on('disconnect', (reason) => {
      console.log('🔌 Disconnected from FastAPI server:', reason);
      setIsConnected(false);
      setConnectionStatus('disconnected');
      
      toast.error('Disconnected from server', {
        duration: 3000,
        position: 'bottom-right',
      });
      
      // Auto-reconnect after a delay if not a manual disconnect
      if (reason !== 'io client disconnect') {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('🔄 Attempting to reconnect...');
          setConnectionStatus('connecting');
          socket.connect();
        }, 2000);
      }
    });
    
    socket.on('connect_error', (error) => {
      console.error('🔌 Socket.IO connection error:', error);
      setIsConnected(false);
      setConnectionStatus('error');
      
      toast.error('Connection error: ' + (error.message || 'Unknown error'), {
        duration: 4000,
        position: 'bottom-right',
      });
    });
    
    socket.on('reconnect', (attemptNumber) => {
      console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
      setIsConnected(true);
      setConnectionStatus('connected');
      setLastActivity(Date.now());
      
      toast.success('Reconnected to server', {
        duration: 2000,
        position: 'bottom-right',
      });
    });
    
    socket.on('reconnect_error', (error) => {
      console.error('🔄 Reconnection failed:', error);
      setConnectionStatus('error');
    });
    
    socket.on('reconnect_failed', () => {
      console.error('🔄 All reconnection attempts failed');
      setConnectionStatus('failed');
      
      toast.error('Unable to reconnect to server', {
        duration: 5000,
        position: 'bottom-right',
      });
    });
    
    // Room events
    socket.on('room_joined', (data) => {
      console.log('✅ Joined room:', data.session_id);
    });
    
    // Keep-alive ping
    socket.on('pong', () => {
      setLastActivity(Date.now());
    });
    
    // Error handling
    socket.on('error', (error) => {
      console.error('🔌 Socket.IO error:', error);
      toast.error('Socket error: ' + (error.message || error), {
        duration: 3000,
        position: 'bottom-right',
      });
    });
    
  }, [sessionId]);
  
  // Disconnect socket
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      console.log('🔌 Manually disconnecting socket...');
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);
  
  // Emit event to server
  const emit = useCallback((event, data) => {
    if (socketRef.current?.connected) {
      console.log(`📡 Emitting ${event}:`, data);
      socketRef.current.emit(event, data);
      return true;
    } else {
      console.warn(`❌ Cannot emit ${event}: Socket not connected`);
      return false;
    }
  }, []);
  
  // Listen for events
  const on = useCallback((event, callback) => {
    if (socketRef.current) {
      console.log(`👂 Listening for ${event}`);
      socketRef.current.on(event, callback);
      
      // Return cleanup function
      return () => {
        if (socketRef.current) {
          socketRef.current.off(event, callback);
        }
      };
    }
    
    return () => {}; // No-op cleanup
  }, []);
  
  // Remove event listener
  const off = useCallback((event, callback) => {
    if (socketRef.current) {
      console.log(`🔇 Removing listener for ${event}`);
      socketRef.current.off(event, callback);
    }
  }, []);
  
  // Send ping to keep connection alive
  const ping = useCallback(() => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('ping');
    }
  }, []);
  
  // Initialize socket on mount
  useEffect(() => {
    initializeSocket();
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [initializeSocket, disconnect]);
  
  // Periodic ping to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      ping();
    }, 30000); // Ping every 30 seconds
    
    return () => clearInterval(pingInterval);
  }, [ping]);
  
  // Monitor connection health
  useEffect(() => {
    const healthCheck = setInterval(() => {
      const timeSinceLastActivity = Date.now() - lastActivity;
      
      // If no activity for 2 minutes and supposed to be connected, try to reconnect
      if (timeSinceLastActivity > 120000 && connectionStatus === 'connected') {
        console.warn('⚠️ No socket activity detected, checking connection...');
        ping();
        
        // If still no response after 10 seconds, force reconnect
        setTimeout(() => {
          if (Date.now() - lastActivity > 130000) {
            console.warn('🔄 Forcing reconnection due to inactivity...');
            disconnect();
            setTimeout(initializeSocket, 1000);
          }
        }, 10000);
      }
    }, 60000); // Check every minute
    
    return () => clearInterval(healthCheck);
  }, [lastActivity, connectionStatus, ping, disconnect, initializeSocket]);
  
  return {
    isConnected,
    connectionStatus,
    socket: socketRef.current,
    emit,
    on,
    off,
    disconnect,
    reconnect: initializeSocket,
    ping,
    lastActivity: new Date(lastActivity),
  };
};

// Custom hook for scraping updates
export const useScrapingUpdates = (sessionId) => {
  const [scrapingStatus, setScrapingStatus] = useState(null);
  const [isScrapingActive, setIsScrapingActive] = useState(false);
  const socket = useSocket(sessionId);
  
  useEffect(() => {
    if (!socket.isConnected) return;
    
    const handleScrapingUpdate = (data) => {
      console.log('📊 Scraping update received:', data);
      setScrapingStatus(data);
      setIsScrapingActive(['starting', 'scraping', 'processing', 'creating_vectors'].includes(data.status));
    };
    
    const handleScrapingCompleted = (data) => {
      console.log('🎉 Scraping completed:', data);
      setIsScrapingActive(false);
      
      toast.success(`Successfully scraped ${data.pages_scraped} pages!`, {
        duration: 4000,
        position: 'bottom-right',
      });
    };
    
    const cleanup1 = socket.on('scraping_update', handleScrapingUpdate);
    const cleanup2 = socket.on('scraping_completed', handleScrapingCompleted);
    
    return () => {
      cleanup1();
      cleanup2();
    };
  }, [socket]);
  
  return {
    scrapingStatus,
    isScrapingActive,
    socket,
  };
};

// Custom hook for typing indicators
export const useTypingIndicator = (sessionId) => {
  const [isTyping, setIsTyping] = useState(false);
  const socket = useSocket(sessionId);
  
  useEffect(() => {
    if (!socket.isConnected) return;
    
    const handleTypingStart = () => {
      console.log('✍️ AI started typing...');
      setIsTyping(true);
    };
    
    const handleTypingEnd = () => {
      console.log('✅ AI finished typing');
      setIsTyping(false);
    };
    
    const cleanup1 = socket.on('typing_start', handleTypingStart);
    const cleanup2 = socket.on('typing_end', handleTypingEnd);
    
    return () => {
      cleanup1();
      cleanup2();
    };
  }, [socket]);
  
  return {
    isTyping,
    socket,
  };
};

export default useSocket;
