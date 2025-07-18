import axios from 'axios';
import { getErrorMessage } from './helpers';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add session ID
api.interceptors.request.use(
  (config) => {
    const sessionId = sessionStorage.getItem('sessionId');
    if (sessionId) {
      config.headers['X-Session-ID'] = sessionId;
    } else {
      // Generate session ID if missing
      const newSessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
      sessionStorage.setItem('sessionId', newSessionId);
      config.headers['X-Session-ID'] = newSessionId;
      console.log('🆔 Generated new session ID:', newSessionId);
    }
    
    console.log('📤 API Request:', {
      url: config.url,
      method: config.method,
      sessionId: config.headers['X-Session-ID']
    });
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log('📥 API Response:', {
      url: response.config?.url,
      status: response.status,
      sessionId: response.config?.headers?.['X-Session-ID']
    });
    return response;
  },
  (error) => {
    const message = getErrorMessage(error);
    
    // Enhanced error logging
    console.error('🚨 API Error:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      message: message,
      data: error.response?.data,
      sessionId: error.config?.headers?.['X-Session-ID']
    });
    
    return Promise.reject(new Error(message));
  }
);

// API endpoints
export const apiEndpoints = {
  // Health check
  health: () => api.get('/api/health'),
  test: () => api.get('/api/test'),
  
  // Scraping
  startScraping: (data) => api.post('/api/scrape', data),
  getScrapingStatus: (sessionId) => api.get(`/api/status/${sessionId}`),
  resetScraping: (sessionId) => api.post(`/api/reset_scraping/${sessionId}`),
  getStats: (sessionId) => api.get(`/api/stats/${sessionId}`),
  
  // Chat
  sendMessage: (data) => api.post('/api/chat', data),
  getHistory: (sessionId, threadId = 'default_conversation') => 
    api.get(`/api/history/${sessionId}?thread_id=${threadId}`),
  clearHistory: (sessionId, threadId = 'default_conversation') => 
    api.post(`/api/clear_history/${sessionId}?thread_id=${threadId}`),
  
  // Streaming chat (returns response for manual handling)
  sendMessageStream: (data) => {
    const sessionId = sessionStorage.getItem('sessionId');
    if (!sessionId) {
      throw new Error('No session ID found');
    }
    
    console.log('🌊 Starting stream request:', {
      sessionId,
      message: data.message?.substring(0, 50) + '...'
    });
    
    return fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': sessionId,
      },
      body: JSON.stringify(data),
    });
  },
};

// Utility functions for API calls
export const apiUtils = {
  // Check if backend is healthy
  async checkHealth() {
    try {
      const response = await apiEndpoints.health();
      return response.data;
    } catch (error) {
      throw new Error(`Backend health check failed: ${getErrorMessage(error)}`);
    }
  },
  
  // Test backend connection
  async testConnection() {
    try {
      const response = await apiEndpoints.test();
      return response.data;
    } catch (error) {
      throw new Error(`Backend connection test failed: ${getErrorMessage(error)}`);
    }
  },
  
  // Start scraping with validation
  async startScraping({ url, crawlDepth, maxPages, maxConcurrent }) {
    try {
      // Basic validation
      if (!url) {
        throw new Error('URL is required');
      }
      
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
      }
      
      const response = await apiEndpoints.startScraping({
        url,
        crawl_depth: crawlDepth,
        max_pages: maxPages,
        max_concurrent: maxConcurrent,
      });
      
      return response.data;
    } catch (error) {
      throw new Error(`Failed to start scraping: ${getErrorMessage(error)}`);
    }
  },
  
  // Send chat message with streaming support
  async sendChatMessage({ message, threadId = 'default_conversation', stream = true }) {
    try {
      if (!message?.trim()) {
        throw new Error('Message is required');
      }
      
      if (stream) {
        // Return the fetch response for streaming
        return apiEndpoints.sendMessageStream({
          message: message.trim(),
          thread_id: threadId,
          stream: true,
        });
      } else {
        // Regular non-streaming response
        const response = await apiEndpoints.sendMessage({
          message: message.trim(),
          thread_id: threadId,
          stream: false,
        });
        
        return response.data;
      }
    } catch (error) {
      throw new Error(`Failed to send message: ${getErrorMessage(error)}`);
    }
  },
  
  // Get conversation history
  async getConversationHistory(sessionId, threadId = 'default_conversation') {
    try {
      const response = await apiEndpoints.getHistory(sessionId, threadId);
      return response.data.history || [];
    } catch (error) {
      console.warn('Failed to get conversation history:', getErrorMessage(error));
      return [];
    }
  },
  
  // Clear conversation history
  async clearConversationHistory(sessionId, threadId = 'default_conversation') {
    try {
      const response = await apiEndpoints.clearHistory(sessionId, threadId);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to clear history: ${getErrorMessage(error)}`);
    }
  },
  
  // Reset scraping session
  async resetScrapingSession(sessionId) {
    try {
      const response = await apiEndpoints.resetScraping(sessionId);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to reset scraping: ${getErrorMessage(error)}`);
    }
  },
  
  // Get scraping statistics
  async getScrapingStats(sessionId) {
    try {
      const response = await apiEndpoints.getStats(sessionId);
      return response.data;
    } catch (error) {
      console.warn('Failed to get scraping stats:', getErrorMessage(error));
      return {};
    }
  },
};

export default api;
