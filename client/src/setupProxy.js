const { createProxyMiddleware } = require('http-proxy-middleware');
const http = require('http');

// Use environment variable with fallback
const BACKEND_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

module.exports = function(app) {
  const proxyConfig = {
    target: BACKEND_URL,
    changeOrigin: true,
    logLevel: 'debug',
    timeout: 600000,
    proxyTimeout: 601000,
    ws: false,
    secure: false,
    agent: new http.Agent({
      keepAlive: true,
      keepAliveMsecs: 60000,
      maxSockets: 25,
      maxFreeSockets: 5,
      timeout: 600000,
    }),
    onError: (err, req, res) => {
      console.error('Proxy Error:', err);
      
      if (res.headersSent) {
        return;
      }

      if (req.path === '/api/run-scrapers') {
        res.writeHead(202, {
          'Content-Type': 'application/json',
          'Connection': 'keep-alive',
        });
        res.end(JSON.stringify({ 
          status: 'pending',
          message: 'Scraping process started. This may take several minutes. Please refresh the data table in a few minutes to see new results.',
        }));
        return;
      }

      res.writeHead(500, {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
      });
      res.end(JSON.stringify({ 
        success: false,
        error: 'Connection error occurred',
        details: err.message
      }));
    },
    onProxyReq: (proxyReq, req, res) => {
      if (req.path === '/api/run-scrapers') {
        proxyReq.setTimeout(600000);
      }
      
      proxyReq.setHeader('Connection', 'keep-alive');
      proxyReq.setHeader('Keep-Alive', 'timeout=600');
      
      if (req.method === 'POST' && req.body) {
        const bodyData = JSON.stringify(req.body);
        proxyReq.setHeader('Content-Type', 'application/json');
        proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
        proxyReq.write(bodyData);
      }
    },
    onProxyRes: (proxyRes, req, res) => {
      proxyRes.headers['connection'] = 'keep-alive';
      proxyRes.headers['keep-alive'] = 'timeout=600';
      proxyRes.headers['Access-Control-Allow-Origin'] = '*';
      proxyRes.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS';
      proxyRes.headers['Access-Control-Allow-Headers'] = 'Content-Type';
    }
  };

  app.use('/api', createProxyMiddleware(proxyConfig));
  app.use('/static/plots', createProxyMiddleware(proxyConfig));
};