import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Settings, 
  Link, 
  Globe, 
  Layers, 
  FileText, 
  Zap, 
  Search, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  RefreshCw,
  Info
} from 'lucide-react';
import { cn, isValidUrl } from '../utils/helpers';
import { apiUtils } from '../utils/api';
import { toast } from 'react-hot-toast';

const ScrapingPanel = ({ 
  onScrapingStart, 
  scrapingStatus, 
  isScrapingActive, 
  onReset 
}) => {
  const [formData, setFormData] = useState({
    url: '',
    scrapeMode: 'multi',
    crawlDepth: 2,
    maxPages: 10,
    maxConcurrent: 3
  });
  
  const [errors, setErrors] = useState({});
  const [isValidating, setIsValidating] = useState(false);

  // Handle form input changes
  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  // Validate form
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.url.trim()) {
      newErrors.url = 'URL is required';
    } else if (!isValidUrl(formData.url.trim())) {
      newErrors.url = 'Please enter a valid URL';
    }
    
    if (formData.crawlDepth < 1 || formData.crawlDepth > 5) {
      newErrors.crawlDepth = 'Crawl depth must be between 1 and 5';
    }
    
    if (formData.maxPages < 1 || formData.maxPages > 100) {
      newErrors.maxPages = 'Max pages must be between 1 and 100';
    }
    
    if (formData.maxConcurrent < 1 || formData.maxConcurrent > 10) {
      newErrors.maxConcurrent = 'Concurrent requests must be between 1 and 10';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      toast.error('Please fix the form errors before submitting');
      return;
    }
    
    setIsValidating(true);
    
    try {
      let url = formData.url.trim();
      
      // Add protocol if missing
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
      }
      
      const scrapingParams = {
        url,
        crawlDepth: formData.scrapeMode === 'single' ? 1 : formData.crawlDepth,
        maxPages: formData.scrapeMode === 'single' ? 1 : formData.maxPages,
        maxConcurrent: formData.scrapeMode === 'single' ? 1 : formData.maxConcurrent,
      };
      
      await onScrapingStart(scrapingParams);
      
    } catch (error) {
      console.error('Scraping error:', error);
      toast.error(error.message || 'Failed to start scraping');
    } finally {
      setIsValidating(false);
    }
  };

  // Status display component
  const StatusDisplay = () => {
    if (!scrapingStatus) return null;

    const getStatusIcon = () => {
      switch (scrapingStatus.status) {
        case 'starting':
        case 'scraping':
        case 'processing':
        case 'creating_vectors':
          return <Loader2 className="w-4 h-4 animate-spin" />;
        case 'completed':
          return <CheckCircle className="w-4 h-4" />;
        case 'error':
          return <AlertCircle className="w-4 h-4" />;
        default:
          return <Info className="w-4 h-4" />;
      }
    };

    const getStatusColor = () => {
      switch (scrapingStatus.status) {
        case 'starting':
        case 'scraping':
        case 'processing':
        case 'creating_vectors':
          return 'bg-blue-50 border-blue-200 text-blue-700';
        case 'completed':
          return 'bg-green-50 border-green-200 text-green-700';
        case 'error':
          return 'bg-red-50 border-red-200 text-red-700';
        default:
          return 'bg-gray-50 border-gray-200 text-gray-700';
      }
    };

    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className={cn(
          'mt-4 p-3 rounded-lg border',
          getStatusColor()
        )}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="font-medium text-sm">
              {scrapingStatus.message || scrapingStatus.status}
            </span>
          </div>
          {scrapingStatus.scraped_pages > 0 && (
            <span className="text-xs opacity-75">
              {scrapingStatus.scraped_pages} pages
            </span>
          )}
        </div>
        
        {scrapingStatus.progress !== undefined && (
          <div className="space-y-2">
            <div className="w-full bg-white/50 rounded-full h-1.5 overflow-hidden">
              <motion.div
                className="h-full bg-current rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${scrapingStatus.progress}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            </div>
            <div className="flex justify-between text-xs opacity-75">
              <span>{scrapingStatus.message || 'Processing...'}</span>
              <span>{Math.round(scrapingStatus.progress)}%</span>
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="h-full bg-gray-50 p-4">
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Website Scraper</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* URL Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Website URL
            </label>
            <input
              type="url"
              value={formData.url}
              onChange={(e) => handleInputChange('url', e.target.value)}
              placeholder="https://example.com"
              disabled={isScrapingActive}
              className={cn(
                'w-full px-3 py-2 border rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent',
                errors.url 
                  ? 'border-red-300 bg-red-50' 
                  : 'border-gray-300',
                isScrapingActive && 'opacity-50 cursor-not-allowed'
              )}
            />
            {errors.url && (
              <p className="text-red-600 text-sm mt-1">{errors.url}</p>
            )}
          </div>

          {/* Scraping Mode */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Scraping Mode
            </label>
            <select
              value={formData.scrapeMode}
              onChange={(e) => handleInputChange('scrapeMode', e.target.value)}
              disabled={isScrapingActive}
              className={cn(
                'w-full px-3 py-2 border border-gray-300 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent',
                isScrapingActive && 'opacity-50 cursor-not-allowed'
              )}
            >
              <option value="single">Single Page Only</option>
              <option value="multi">Full Website Crawling</option>
            </select>
          </div>

          {/* Multi-page options */}
          {formData.scrapeMode === 'multi' && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-4"
            >
              {/* Crawl Depth */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Crawl Depth
                </label>
                <select
                  value={formData.crawlDepth}
                  onChange={(e) => handleInputChange('crawlDepth', parseInt(e.target.value))}
                  disabled={isScrapingActive}
                  className={cn(
                    'w-full px-3 py-2 border border-gray-300 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent',
                    isScrapingActive && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <option value={1}>1 - Main page only</option>
                  <option value={2}>2 - + Linked pages</option>
                  <option value={3}>3 - Deep crawl</option>
                  <option value={4}>4 - Very deep</option>
                  <option value={5}>5 - Maximum depth</option>
                </select>
              </div>

              {/* Max Pages */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Pages
                </label>
                <select
                  value={formData.maxPages}
                  onChange={(e) => handleInputChange('maxPages', parseInt(e.target.value))}
                  disabled={isScrapingActive}
                  className={cn(
                    'w-full px-3 py-2 border border-gray-300 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent',
                    isScrapingActive && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <option value={5}>5 pages</option>
                  <option value={10}>10 pages</option>
                  <option value={20}>20 pages</option>
                  <option value={50}>50 pages</option>
                  <option value={100}>100 pages</option>
                </select>
              </div>

              {/* Crawl Speed */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Crawl Speed
                </label>
                <select
                  value={formData.maxConcurrent}
                  onChange={(e) => handleInputChange('maxConcurrent', parseInt(e.target.value))}
                  disabled={isScrapingActive}
                  className={cn(
                    'w-full px-3 py-2 border border-gray-300 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent',
                    isScrapingActive && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <option value={1}>1 - Slow & Polite</option>
                  <option value={3}>3 - Balanced</option>
                  <option value={5}>5 - Fast</option>
                  <option value={10}>10 - Very Fast</option>
                </select>
              </div>
            </motion.div>
          )}

          {/* Submit Button */}
          <div className="flex gap-2">
            <button
              type="submit"
              disabled={isScrapingActive || isValidating}
              className={cn(
                'flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                isScrapingActive || isValidating
                  ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                  : 'bg-orange-500 text-white hover:bg-orange-600'
              )}
            >
              {isScrapingActive ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Scraping...
                </>
              ) : isValidating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Validating...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Start Scraping
                </>
              )}
            </button>

            {/* Reset Button */}
            {(scrapingStatus?.status === 'completed' || scrapingStatus?.status === 'error') && (
              <button
                type="button"
                onClick={onReset}
                className="px-3 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg transition-all"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>

        {/* Status Display */}
        <StatusDisplay />
      </div>
    </div>
  );
};

export default ScrapingPanel;