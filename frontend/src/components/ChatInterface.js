import React, { useState, useRef, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Send, RotateCcw, Trash2, ChevronLeft, ChevronRight } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { apiUtils } from '../utils/api';
import ChatMessage from './ChatMessage';

const ChatInterface = ({
  hasScrapedContent,
  isTyping,
  sessionId,
  showSidebar,
  onToggleSidebar,
}) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      scrollToBottom();
    }
  }, [messages, isStreaming, scrollToBottom]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Lightweight streaming formatter - NO citations in body
  const formatStreamingText = useCallback((content) => {
    if (!content) return '';

    // Remove citations completely from streaming text - they'll show at bottom in final message
    let formatted = content.replace(/\[Source:[^\]]*\]/gi, '');
    
    // Simple formatting only - optimized for performance during streaming
    // Bold text - only complete pairs
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>');
    
    // Basic line breaks
    formatted = formatted.replace(/\n\n/g, '</p><p class="mb-3">');
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Wrap in paragraph if needed
    if (formatted && !formatted.startsWith('<')) {
      formatted = `<p class="mb-3">${formatted}</p>`;
    }

    return formatted;
  }, []);

  const addMessage = useCallback((type, content) => {
    const newMessage = {
      id: Date.now().toString(),
      content,
      isUser: type === 'user',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
    return newMessage.id;
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !hasScrapedContent || isProcessing) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    addMessage('user', userMessage);
    setIsProcessing(true);
    setIsStreaming(true);
    setStreamingText('');

    abortControllerRef.current = new AbortController();

    // Debug logging
    console.log('🔍 Debug - Sending chat request:', {
      sessionId: sessionId || 'MISSING',
      hasScrapedContent,
      messageLength: userMessage.length,
      message: userMessage.substring(0, 50) + '...'
    });

    try {
      // Ensure session ID is present
      const finalSessionId = sessionId || `session-${Date.now()}`;
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': finalSessionId,
        },
        body: JSON.stringify({
          message: userMessage,
          thread_id: 'default_conversation',
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      console.log('🔍 Debug - Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('🚨 API Error Response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        throw new Error(`Server error ${response.status}: ${errorText || response.statusText}`);
      }

      let assistantResponse = '';
        let responseSources = [];
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'chunk' && data.chunk) {
                assistantResponse += data.chunk;
                // Throttled updates for better performance
                setStreamingText(assistantResponse);
              } else if (data.type === 'sources' && data.sources) {
                responseSources = data.sources;
              } else if (data.type === 'complete') {
                break;
              } else if (data.type === 'error') {
                throw new Error(data.error || 'Unknown server error');
              }
            } catch (parseError) {
              console.warn('Skipping malformed JSON:', parseError);
              continue;
            }
          }
        }
      }

      setIsStreaming(false);
      setStreamingText('');
      // Remove citations from final message before adding to chat
      const cleanResponse = assistantResponse.replace(/\[Source:[^\]]*\]/gi, '');
      
      // Create message with sources for bottom display
      const messageWithSources = {
        id: Date.now().toString(),
        content: cleanResponse,
        isUser: false,
        timestamp: new Date(),
        sources: responseSources || []
      };
      
      setMessages((prev) => [...prev, messageWithSources]);
      setTimeout(scrollToBottom, 100);
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request aborted');
        return;
      }
      
      console.error('🚨 Chat error details:', {
        error: error.message,
        sessionId,
        hasScrapedContent,
        userMessage: userMessage.substring(0, 50)
      });
      
      setIsStreaming(false);
      setStreamingText('');
      
      let errorMessage = '⚠️ Sorry, something went wrong. Please try again.';
      
      if (error.message.includes('Please scrape a website first')) {
        errorMessage = '⚠️ Please scrape a website first before chatting.';
        toast.error('Please scrape a website first!');
      } else if (error.message.includes('Message is required')) {
        errorMessage = '⚠️ Please enter a message.';
        toast.error('Please enter a message.');
      } else if (error.message.includes('400')) {
        errorMessage = '⚠️ Bad request. Please check if website scraping completed successfully.';
        toast.error('Bad request - ensure website is scraped first.');
      } else {
        toast.error('Unable to process your request. Please try again later.');
      }
      
      addMessage('assistant', errorMessage);
    } finally {
      setIsProcessing(false);
      abortControllerRef.current = null;
      if (inputRef.current && document.activeElement !== inputRef.current) {
        inputRef.current.focus();
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = async () => {
    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      await apiUtils.clearConversationHistory(sessionId);
      setMessages([]);
      setIsStreaming(false);
      setStreamingText('');
      setIsProcessing(false);
      toast.success('Conversation cleared');
    } catch (error) {
      toast.error('Failed to clear conversation');
    }
  };

  if (messages.length === 0) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <button
              onClick={onToggleSidebar}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label={showSidebar ? 'Hide scraping panel' : 'Show scraping panel'}
            >
              {showSidebar ? (
                <ChevronLeft className="w-5 h-5 text-gray-600" />
              ) : (
                <ChevronRight className="w-5 h-5 text-gray-600" />
              )}
            </button>
            <h2 className="font-semibold text-gray-900">Chat</h2>
          </div>
          <button
            onClick={clearChat}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Clear conversation"
          >
            <Trash2 className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-md">
            <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-orange-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <span className="text-2xl">🤖</span>
            </div>
            <h3 className="text-2xl font-semibold text-gray-900 mb-4">
              Welcome to Webly
            </h3>
            <p className="text-gray-600 mb-6">
              {hasScrapedContent
                ? "I've analyzed the website content. Ask me anything about it!"
                : 'Please scrape a website first to start chatting.'}
            </p>

            {!hasScrapedContent && (
              <button
                onClick={onToggleSidebar}
                className="px-6 py-3 bg-orange-500 text-white rounded-xl hover:bg-orange-600 transition-colors font-medium"
                aria-label="Open scraping panel"
              >
                Open Scraping Panel
              </button>
            )}

            {hasScrapedContent && (
              <div className="bg-orange-50 p-4 rounded-xl border border-orange-200">
                <p className="text-sm font-medium text-orange-800 mb-2">💡 Try asking:</p>
                <ul className="text-sm text-orange-700 space-y-1 text-left">
                  <li>• "What is this website about?"</li>
                  <li>• "Summarize the main content"</li>
                  <li>• "What are the key points?"</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        {hasScrapedContent && (
          <div className="border-t border-gray-200 p-4">
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-3">
                <input
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Message Webly..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent text-[15px]"
                  disabled={isProcessing}
                  aria-label="Message input"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isProcessing}
                  className="px-4 py-3 bg-orange-500 text-white rounded-xl hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                  aria-label="Send message"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-2 text-center">
                Press Enter to send
              </p>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleSidebar}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label={showSidebar ? 'Hide scraping panel' : 'Show scraping panel'}
          >
            {showSidebar ? (
              <ChevronLeft className="w-5 h-5 text-gray-600" />
            ) : (
              <ChevronRight className="w-5 h-5 text-gray-600" />
            )}
          </button>
          <h2 className="font-semibold text-gray-900">Chat</h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={clearChat}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Clear conversation"
          >
            <Trash2 className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto py-8 px-4">
          {messages.map((message) => (
            <div key={message.id} className="mb-8">
              <ChatMessage message={message} />
            </div>
          ))}

          {/* CLEAN streaming with real-time formatting */}
          {isStreaming && (
            <div className="mb-8">
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-orange-600 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-white text-sm">🤖</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="mb-2">
                    <span className="font-semibold text-gray-900">Webly</span>
                  </div>
                  <div className="prose prose-gray max-w-none">
                    <div
                      className="text-gray-800 text-[15px] leading-7"
                      dangerouslySetInnerHTML={{ __html: formatStreamingText(streamingText) }}
                      style={{
                        wordBreak: 'break-word',
                        overflowWrap: 'break-word',
                        minHeight: '24px',
                      }}
                    />
                    <span className="inline-block w-2 h-5 bg-orange-500 ml-1 animate-pulse" />
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-gray-200 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-3">
            <input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Webly..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent text-[15px]"
              disabled={!hasScrapedContent || isProcessing}
              aria-label="Message input"
            />
            <button
              onClick={handleSendMessage}
              disabled={!hasScrapedContent || isProcessing || !inputValue.trim()}
              className="px-4 py-3 bg-orange-500 text-white rounded-xl hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              aria-label="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send
          </p>
        </div>
      </div>
    </div>
  );
};

ChatInterface.propTypes = {
  hasScrapedContent: PropTypes.bool.isRequired,
  isTyping: PropTypes.bool.isRequired,
  sessionId: PropTypes.string.isRequired,
  showSidebar: PropTypes.bool.isRequired,
  onToggleSidebar: PropTypes.func.isRequired,
};

export default ChatInterface;