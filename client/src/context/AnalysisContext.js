import React, { createContext, useContext, useState } from 'react';

const AnalysisContext = createContext();

export function AnalysisProvider({ children }) {
  const [selectedServices, setSelectedServices] = useState([]);
  const [plots, setPlots] = useState({});
  const [startDate, setStartDate] = useState('2023-08-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const resetAnalysis = () => {
    setSelectedServices([]);
    setPlots({});
    setStartDate('2023-08-01');
    setEndDate(new Date().toISOString().split('T')[0]);
    setError(null);
  };

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);
      const payload = {
        startDate,
        endDate,
        selectedServices,
      };

      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP error! status: ${response.status}`);
      }
      
      if (result.success && result.plots) {
        setPlots(result.plots);
      } else {
        throw new Error(result.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const value = {
    selectedServices,
    setSelectedServices,
    plots,
    setPlots,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    loading,
    setLoading,
    error,
    setError,
    handleAnalyze,
    resetAnalysis
  };

  return (
    <AnalysisContext.Provider value={value}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
} 