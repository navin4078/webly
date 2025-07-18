import React, { useState, useMemo } from 'react';
import { Copy, Check, User } from 'lucide-react';
import { toast } from 'react-hot-toast';

const ChatMessage = ({ message }) => {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      toast.success('Copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Failed to copy');
    }
  };

  const formatContent = (content) => {
    // Remove all citations and URLs completely - no processing into superscript
    let formatted = content
      .replace(/\[Source:[^\]]+\]/gi, '') // Remove [Source: ...] tags
      .replace(/Source:[^\]\(\n]+/gi, '') // Remove Source: ... references
      .replace(/https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g, '') // Remove URLs completely
      .replace(/\(\s*(?=https?)/gi, '') // Remove opening parentheses before URLs
      .replace(/ +/g, ' ') // Multiple spaces to single space
      .replace(/\n +/g, '\n') // Remove spaces at start of lines
      .trim();
    
    const formattedMarkdown = formatMarkdown(formatted);
    
    // NO citations processing - return clean content only
    return { formattedContent: formattedMarkdown, citations: [] };
  };

  const formatMarkdown = (content) => {
    // Format markdown-style content
    let formatted = content;
    
    // Handle headers
    formatted = formatted.replace(/^### (.*$)/gim, '<h3 class="text-lg font-semibold text-gray-900 mt-6 mb-3">$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gim, '<h2 class="text-xl font-semibold text-gray-900 mt-6 mb-3">$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold text-gray-900 mt-6 mb-4">$1</h1>');
    
    // Handle bold text
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>');
    
    // Handle bullet points
    formatted = formatted.replace(/^[•·-]\s(.+)$/gm, '<li class="ml-4 mb-1">$1</li>');
    formatted = formatted.replace(/^(\d+)\.\s(.+)$/gm, '<li class="ml-4 mb-1 list-decimal">$2</li>');
    
    // Wrap consecutive list items in ul tags
    formatted = formatted.replace(/(<li[^>]*>.*?<\/li>\s*)+/gs, '<ul class="list-disc ml-4 mb-4 space-y-1">$&</ul>');
    
    // Handle line breaks and paragraphs
    formatted = formatted.replace(/\n\n/g, '</p><p class="mb-4">');
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Wrap in paragraph tags if not already wrapped
    if (!formatted.startsWith('<h') && !formatted.startsWith('<ul') && !formatted.startsWith('<li') && !formatted.startsWith('<p')) {
      formatted = `<p class="mb-4">${formatted}</p>`;
    }
    
    return formatted;
  };

  const { formattedContent, citations } = useMemo(() => {
    // If message has sources directly, use them; otherwise extract from content
    if (message.sources && message.sources.length > 0) {
      // Use provided sources and clean content
      const cleanContent = message.content
        .replace(/\[Source:[^\]]+\]/gi, '') // Remove [Source: ...] tags
        .replace(/Source:[^\]\(\n]+/gi, '') // Remove Source: ... references
        .replace(/https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g, '') // Remove URLs completely
        .replace(/\(\s*(?=https?)/gi, '') // Remove opening parentheses before URLs
        .replace(/ +/g, ' ') // Multiple spaces to single space
        .replace(/\n +/g, '\n') // Remove spaces at start of lines
        .trim();
      
      const formatted = formatMarkdown(cleanContent);
      const sources = message.sources.map((source, index) => ({
        url: source.url,
        number: index + 1,
        id: `citation-${index + 1}`,
        title: source.title || source.url
      }));
      
      return { formattedContent: formatted, citations: sources };
    } else {
      // Fall back to old extraction method
      return formatContent(message.content);
    }
  }, [message.content, message.sources]);

  if (message.isUser) {
    return (
      <div className="flex gap-4 justify-end">
        <div className="flex-1 max-w-2xl">
          <div className="bg-gray-100 rounded-2xl px-4 py-3">
            <div 
              className="text-gray-900 text-[15px] leading-7"
              style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
            >
              {message.content}
            </div>
          </div>
          <div className="flex items-center justify-end gap-2 mt-2 px-2">
            <span className="text-xs text-gray-500">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        </div>
        <div className="w-8 h-8 bg-gray-500 rounded-full flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-white" />
        </div>
      </div>
    );
  }

  return (
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
            dangerouslySetInnerHTML={{ __html: formattedContent }}
            style={{ 
              wordBreak: 'break-word',
              overflowWrap: 'break-word'
            }}
          />
          
          {citations.length > 0 && (
            <div className="mt-6 pt-4 border-t border-gray-200">
              <p className="text-sm font-medium text-gray-700 mb-3">Sources:</p>
              <div className="space-y-2">
                {citations.map((citation) => (
                  <div key={citation.id} className="flex items-start gap-3 text-sm">
                    <span className="text-orange-500 font-medium flex-shrink-0">[{citation.number}]</span>
                    <a 
                      href={citation.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 hover:underline break-all"
                    >
                      {citation.title}
                    </a>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-2 mt-3">
          <span className="text-xs text-gray-500">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          <button
            onClick={copyToClipboard}
            className="text-gray-400 hover:text-gray-600 p-1 rounded transition-colors"
            title="Copy message"
          >
            {copied ? (
              <Check className="w-3 h-3 text-green-500" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;