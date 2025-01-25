import React, { useState, useMemo } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { lightTheme, darkTheme } from './theme';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import About from './pages/About';
import './App.css';
import { AnalysisProvider } from './context/AnalysisContext';
import ConnectionStatus from './components/ConnectionStatus';
import FailureAnalysis from './pages/FailureAnalysis';
import DataTable from './pages/DataTable';
import PredictiveAnalysis from './pages/PredictiveAnalysis';
function App() {
  const [mode, setMode] = useState('light');

  const theme = useMemo(() => 
    mode === 'light' ? lightTheme : darkTheme,
    [mode]
  );

  const toggleTheme = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AnalysisProvider>
        <Router>
          <Layout toggleTheme={toggleTheme}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/failure-analysis" element={<FailureAnalysis />} />
              <Route path="/about" element={<About />} />
              <Route path="/data" element={<DataTable />} />
              <Route path="/predictive-analysis" element={<PredictiveAnalysis />} />
            </Routes>
          </Layout>
        </Router>
      </AnalysisProvider>
      <ConnectionStatus />
    </ThemeProvider>
  );
}

export default App;