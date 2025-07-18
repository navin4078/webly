import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Create React root and render app
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Hot module replacement for development
if (module.hot) {
  module.hot.accept();
}
