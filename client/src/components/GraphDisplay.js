import React, { useState, useEffect, forwardRef } from 'react';
import { Box, Paper, CircularProgress, Typography, Chip, useTheme, useMediaQuery, Button, IconButton, Tooltip, LinearProgress, Skeleton, Fade, FormControl, InputLabel, Select, MenuItem, Dialog, DialogTitle, DialogContent, DialogActions, Alert } from '@mui/material';
import { useAnalysis } from '../context/AnalysisContext';
import { 
  SaveAlt as SaveIcon,
  Download as DownloadIcon,
  Analytics as AnalyticsIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import JSZip from 'jszip';
import { analyzePlot } from '../utils/api';
import PlotAnalysis from './PlotAnalysis';

const plotConfigs = {
  figure1: {
    title: 'Weekly Overview',
    description: 'Distribution of incidents across services per day of the week'
  },
  figure2: {
    title: 'Daily Overview',
    description: 'Distribution of incidents across services per hour of day'
  },
  figure3: {
    title: 'Mean Time To Recovery by Service',
    description: 'Mean Time To Recovery cumulative distribution and percentage of incidents'
  },
  figure4: {
    title: 'Mean Time To Recovery by Provider',
    description: 'Mean Time To Recovery cumulative distribution comparison across providers'
  },
  figure5: {
    title: 'Mean Time To Recovery Boxplot',
    description: 'Detailed MTTR distribution with service-level boxplots'
  },
  figure6: {
    title: 'Mean Time Between Failures by Service',
    description: 'Mean Time Between Failures cumulative distribution and percentage of incidents'
  },
  figure7: {
    title: 'Mean Time Between Failures by Provider',
    description: 'Mean Time Between Failures cumulative distribution comparison across providers'
  },
  figure8: {
    title: 'Mean Time Between Failures Boxplot',
    description: 'Detailed MTBF distribution with service-level boxplots'
  },
  figure9: {
    title: 'Resolution Stages',
    description: 'Duration and Distribution of Resolution Stages'
  },
  figure10: {
    title: 'Status Combinations',
    description: 'Concurrent status combinations'
  },
  figure11: {
    title: 'Daily Availability',
    description: 'Daily service availability patterns'
  },
  figure12: {
    title: 'Service Co-occurrence',
    description: 'Co-occurrence matrix of simultaneous service incidents'
  },
  figure13: {
    title: 'Co-occurrence Probability',
    description: 'Probability matrix of service co-occurrences'
  },
  figure14: {
    title: 'Per Service Co-occurrence',
    description: 'Co-occurrence of failures across services for each provide'
  },
  figure15: {
    title: 'Incident Outage Timeline',
    description: 'Timeline visualization of service outages'
  },
  figure16: {
    title: 'Autocorrelations',
    description: 'Temporal autocorrelation analysis of incidents'
  },
  figure17: {
    title: 'Incident Impact Distribution',
    description: 'Distribution of incident impact levels across providers'
  }
};

const formatDate = (dateStr) => {
  try {
    // Ensure dateStr is exactly 8 characters (YYYYMMDD)
    if (dateStr?.length !== 8) {
      console.error('Invalid date string:', dateStr);
      return 'Invalid Date';
    }

    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);
    
    // Validate components
    if (isNaN(year) || isNaN(month) || isNaN(day)) {
      console.error('Invalid date components:', { year, month, day });
      return 'Invalid Date';
    }

    const date = new Date(year, month - 1, day);
    
    // Validate result
    if (isNaN(date.getTime())) {
      console.error('Invalid date result:', date);
      return 'Invalid Date';
    }

    return date.toLocaleDateString('en-US', {
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    });
  } catch (error) {
    console.error('Error formatting date:', error);
    return 'Invalid Date';
  }
};

const formatServiceName = (serviceName) => {
  // Convert service names to display format
  const [provider, service] = serviceName.split(':');
  switch(true) {
    case provider === 'OpenAI' && service === 'DALL-E':
      return 'OpenAI DALL-E';  // Special case for DALL-E
    case provider === 'OpenAI':
      return `OpenAI ${service}`;
    case provider === 'Anthropic':
      return `Anthropic ${service}`;
    case provider === 'Google':
      return `Google ${service}`;
    case provider === 'Character.AI':
      return 'Character.AI';
    case provider === 'StabilityAI': {
      // Handle different StabilityAI services
      switch(service) {
        case 'REST':
        case 'gRPC':
        case 'Assistant':
        case 'Stable Diffusion':
          return `StabilityAI ${service}`;
        default:
          return `StabilityAI ${service}`; // Fallback 
      }
    }
    default:
      return serviceName;
  }
};

const PlotDetails = ({ details }) => {
  if (!details) return null;
  
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      gap: 1, 
      mb: 2,
      p: 1.5,
      backgroundColor: 'background.default',
      borderRadius: 1,
      border: '1px solid',
      borderColor: 'divider',
    }}>
      <Box sx={{ 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: 0.5, 
        alignItems: 'center' 
      }}>
        <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
          Services:
        </Typography>
        {details.services.map(service => (
          <Chip
            key={service}
            label={service}
            size="small"
            sx={{
              backgroundColor: theme => `${theme.palette.primary.main}22`,
              color: 'primary.main',
              fontWeight: 500,
              borderRadius: '6px',
              '& .MuiChip-label': {
                px: 1,
              },
            }}
          />
        ))}
      </Box>
      <Typography variant="body2" color="text.secondary">
        From {details.startDate} to {details.endDate}
      </Typography>
    </Box>
  );
};

// Add new PlotAnalysisDialog component
const PlotAnalysisDialog = ({ open, onClose, plotTitle, analysis, loading, error }) => {
  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          backgroundColor: 'background.paper',
        }
      }}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        pb: 1
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AnalyticsIcon color="primary" />
          <Typography variant="h6">
            Analysis: {plotTitle}
          </Typography>
        </Box>
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : (
          <Typography 
            variant="body2" 
            sx={{ 
                whiteSpace: 'pre-line',
                overflowWrap: 'break-word',
                wordBreak: 'break-word',
                '& strong': {
                  fontWeight: 600,
                  color: 'primary.main',
                },
            }}
          >
            {analysis}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

const GraphDisplay = forwardRef((props, ref) => {
  const { plots, setPlots, loading } = useAnalysis();
  const [imageErrors, setImageErrors] = useState({});
  const [plotDetails, setPlotDetails] = useState({});
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [savingAll, setSavingAll] = useState(false);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [selectedPlotForAnalysis, setSelectedPlotForAnalysis] = useState(null);
  const [selectedPlotForDialog, setSelectedPlotForDialog] = useState(null);
  const [dialogAnalysis, setDialogAnalysis] = useState('');
  const [dialogLoading, setDialogLoading] = useState(false);
  const [dialogError, setDialogError] = useState(null);

  useEffect(() => {
    setImageErrors({});
    extractPlotDetails();
  }, [plots]);

  useEffect(() => {
    if (error) {
      // Show error message but don't clear plots
      setError(error);
    }
  }, [error]);

  React.useImperativeHandle(ref, () => ({
    refreshPlots: (newPlots) => {
      setPlots(newPlots);
      setImageErrors({});
    }
  }));

  const handleImageError = (figureId) => {
    console.error(`Failed to load image: ${plots[figureId]}`);
    setImageErrors(prev => ({
      ...prev,
      [figureId]: true
    }));
  };

  const handleImageLoad = (figureId) => {
    setImageErrors(prev => ({
      ...prev,
      [figureId]: false
    }));
  };

  const extractPlotDetails = () => {
    const details = {};
    Object.entries(plots).forEach(([figureId, plotPath]) => {
      try {
        const filename = plotPath.split('/').pop();
        console.log('Processing filename:', filename);
        
        const [mainPart, servicePart, timestamp] = filename.split('__');
        console.log('Service part:', servicePart);
        
        // Format is: plotType_YYYYMMDD_YYYYMMDD
        const dateMatch = mainPart.match(/.*?_(\d{8})_(\d{8})/);
        if (!dateMatch) {
          console.error('Could not extract dates from:', mainPart);
          return;
        }
        
        const [, startDate, endDate] = dateMatch;

        // Fix service name formatting
        const services = servicePart.split('-')
          .filter(service => service !== 'E')  // DallE formatting
          .map(service => {
            console.log('Processing service:', service); 
            
            // Handle special cases
            if (service.includes('DALL')) {
              return 'OpenAI DALL·E';
            }
            
            // Handle Character.AI (try different possible formats)
            if (service.toLowerCase().includes('character')) {
              return 'Character.AI';
            }
            
            // Handle StabilityAI (try different possible formats)
            if (service.toLowerCase().includes('stability')) {
              const serviceParts = service.split('_');
              const stabilityService = serviceParts.length > 1 ? serviceParts[1] : 'REST';
              return `StabilityAI ${stabilityService}`;
            }
            
            // Handle other services
            return formatServiceName(service.replace('_', ':'));
          });

        console.log('Processed services:', services); // Debug log

        details[figureId] = {
          startDate: formatDate(startDate),
          endDate: formatDate(endDate),
          services: services
        };

        console.log('Extracted details for', figureId, ':', details[figureId]);

      } catch (error) {
        console.error('Error parsing filename:', error);
      }
    });
    setPlotDetails(details);
  };

  const handleSavePlot = async (figureId) => {
    try {
      if (!plots[figureId]) return;

      const response = await fetch(plots[figureId]);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Create a formatted filename with date and service info
      const plotDetails = plotConfigs[figureId];
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `${plotDetails.title.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.png`;
      
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(`Error saving plot ${figureId}:`, error);
      setError(`Failed to save ${plotConfigs[figureId].title}`);
    }
  };

  const handleSaveAll = async () => {
    try {
      setSavingAll(true);
      const plotIds = Object.keys(plots);
      const zip = new JSZip();
      const timestamp = new Date().toISOString().split('T')[0];
      
      // Create plots folder in the zip
      const plotsFolder = zip.folder("plots");
      
      for (let i = 0; i < plotIds.length; i++) {
        const figureId = plotIds[i];
        try {
          // Fetch the image
          const response = await fetch(plots[figureId]);
          const blob = await response.blob();
          
          const plotDetails = plotConfigs[figureId];
          const filename = `${plotDetails.title.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.png`;
          
          plotsFolder.file(filename, blob);
          
        } catch (error) {
          console.error(`Error adding ${figureId} to zip:`, error);
          setError(`Failed to add ${plotConfigs[figureId].title} to zip`);
        }
      }
      
      const content = await zip.generateAsync({ type: "blob" });
      
      // Create download link
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = `llm_analysis_plots_${timestamp}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Error saving all plots:', error);
      setError('Failed to create ZIP file');
    } finally {
      setSavingAll(false);
    }
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setAnalysisError(null);
    
    if (!selectedPlotForAnalysis || !plots[selectedPlotForAnalysis]) {
      setAnalysisError('Please select a plot for analysis');
      setAnalyzing(false);
      return;
    }
    
    try {
      const plotUrl = plots[selectedPlotForAnalysis];
      const result = await analyzePlot(plotUrl);
      if (result.success) {
        setAnalysis(result.analysis);
      } else {
        setAnalysisError(result.error || 'Failed to analyze plot');
      }
    } catch (error) {
      setAnalysisError(error.message || 'Failed to analyze plot. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  // Individual plot analysis
  const handleAnalyzeIndividual = async (figureId) => {
    setSelectedPlotForDialog(figureId);
    setDialogLoading(true);
    setDialogError(null);
    setDialogAnalysis('');

    try {
      const plotUrl = plots[figureId];
      const result = await analyzePlot(plotUrl, figureId);
      if (result.success) {
        setDialogAnalysis(result.analysis);
      } else {
        setDialogError(result.error || 'Failed to analyze plot');
      }
    } catch (error) {
      setDialogError(error.message || 'Failed to analyze plot. Please try again.');
    } finally {
      setDialogLoading(false);
    }
  };

  // Add function to handle dialog close
  const handleCloseDialog = () => {
    setSelectedPlotForDialog(null);
    setDialogAnalysis('');
    setDialogError(null);
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Paper 
          elevation={3}
          sx={{ 
            p: 4, 
            textAlign: 'center',
            backgroundColor: 'background.paper',
            borderRadius: 2
          }}
        >
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <CircularProgress size={60} />
            <Box sx={{ width: '100%', maxWidth: 400 }}>
              <Typography variant="h6" gutterBottom>
                Analyzing Data
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Generating visualizations and processing insights...
              </Typography>
              <LinearProgress 
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  mb: 2
                }} 
              />
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {Object.keys(plotConfigs).map((figureId) => (
                  <Box 
                    key={figureId}
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center',
                      gap: 2
                    }}
                  >
                    <Skeleton 
                      variant="circular" 
                      width={24} 
                      height={24} 
                    />
                    <Typography variant="body2" color="text.secondary">
                      {plotConfigs[figureId].title}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>
          </Box>
        </Paper>
      </Box>
    );
  }

  const allFigures = Object.keys(plotConfigs);

  // Only show plots section if there are plots or we're loading
  if (Object.keys(plots).length === 0 && !loading && !error) {
    return null;  // Return nothing only if no plots AND not loading AND no error
  }

  // Define grid layout configurations
  const getGridConfig = (figureId) => {
    const configs = {
      figure1: { // Monthly Overview
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px', md: '500px' }
      },
      figure2: { // Daily Overview
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px', md: '500px' }
      },
      figure3: { // MTTR Distribution
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 6' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure4: { // MTTR by Provider
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 6' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure5: { // MTTR Boxplot
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px' }
      },
      figure6: { // MTBF Distribution
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 6' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure7: { // MTBF by Provider
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 6' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure8: { // MTBF Boxplot
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px' }
      },
      figure9: { // Resolution Activities
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px' }
      },
      figure10: { // Status Combinations
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px' }
      },
      figure11: { // Daily Availability
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '300px', sm: '400px' }
      },
      figure12: { // Service Co-occurrence
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure13: { // Co-occurrence Probability
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure14: { // Service Incidents
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure15: { // Incident Outage Timeline
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure16: { // Autocorrelations
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      },
      figure17: { // Incident Distribution
        gridColumn: { xs: 'span 12', sm: 'span 12', md: 'span 12' },
        minHeight: { xs: '400px', sm: '500px' }
      }
    };
    return configs[figureId] || {
      gridColumn: { xs: 'span 12', sm: 'span 6', md: 'span 4' },
      minHeight: { xs: '300px', sm: '400px' }
    };
  };

  return (
    <Fade in={Object.keys(plots).length > 0 || loading || error}>
      <Box sx={{ p: 3 }}>
        {Object.keys(plots).length > 0 && (
          <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={handleSaveAll}
              disabled={savingAll}
              sx={{
                borderRadius: 2,
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: 2
                }
              }}
            >
              {savingAll ? 'Saving...' : 'Save All Plots'}
            </Button>
          </Box>
        )}

        <Box sx={{ 
          display: 'grid',
          gridTemplateColumns: {
            xs: 'repeat(12, 1fr)',
            sm: 'repeat(12, 1fr)',
            md: 'repeat(12, 1fr)'
          },
          gap: 3,
        }}>
          {allFigures.map((figureId) => {
            const gridConfig = getGridConfig(figureId);
            
            return (
              <Fade in key={figureId}>
                <Paper
                  elevation={3}
                  sx={{
                    p: 2,
                    backgroundColor: 'background.paper',
                    borderRadius: 2,
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                    gridColumn: gridConfig.gridColumn,
                    minHeight: gridConfig.minHeight,
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 8
                    }
                  }}
                >
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'flex-start'
                  }}>
                    <Box>
                      <Typography variant="h6" fontWeight="bold" color="primary">
                        {plotConfigs[figureId].title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {plotConfigs[figureId].description}
                      </Typography>
                    </Box>
                    {plots[figureId] && (
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Analyze Plot">
                          <IconButton
                            onClick={() => handleAnalyzeIndividual(figureId)}
                            size="small"
                            sx={{
                              transition: 'all 0.2s ease-in-out',
                              '&:hover': {
                                transform: 'translateY(-2px)',
                                color: 'primary.main'
                              }
                            }}
                          >
                            <AnalyticsIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Save Plot">
                          <IconButton 
                            onClick={() => handleSavePlot(figureId)}
                            size="small"
                            sx={{
                              transition: 'all 0.2s ease-in-out',
                              '&:hover': {
                                transform: 'translateY(-2px)',
                                color: 'primary.main'
                              }
                            }}
                          >
                            <SaveIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    )}
                  </Box>

                  <PlotDetails details={plotDetails[figureId]} />

                  <Box 
                    sx={{ 
                      flexGrow: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                      minHeight: isMobile ? '250px' : '300px'
                    }}
                  >
                    {plots[figureId] ? (
                      imageErrors[figureId] ? (
                        <Typography color="error">
                          Failed to load image. Please try refreshing the analysis.
                        </Typography>
                      ) : (
                        <img
                          src={plots[figureId]}
                          alt={`${plotConfigs[figureId].title}`}
                          onError={() => handleImageError(figureId)}
                          onLoad={() => handleImageLoad(figureId)}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            display: 'block',
                            borderRadius: '8px'
                          }}
                        />
                      )
                    ) : (
                      <Box 
                        sx={{ 
                          height: '100%',
                          width: '100%',
                          display: 'flex', 
                          flexDirection: 'column',
                          justifyContent: 'center', 
                          alignItems: 'center',
                          bgcolor: 'background.default',
                          borderRadius: 1,
                          p: 2,
                          opacity: 0.7
                        }}
                      >
                        {error ? (
                          <>
                            <Typography color="error" align="center" gutterBottom>
                              Error generating plot
                            </Typography>
                            <Typography color="text.secondary" align="center" variant="body2">
                              {error}
                            </Typography>
                          </>
                        ) : (
                          <Typography color="text.secondary" align="center">
                            Error with the generation of this plot. Please try refreshing the analysis.
                          </Typography>
                        )}
                      </Box>
                    )}
                  </Box>
                </Paper>
              </Fade>
            );
          })}
        </Box>

        {/* Add PlotAnalysis component */}
        {Object.keys(plots).length > 0 && (
          <PlotAnalysis 
            plots={plots}
            plotConfigs={plotConfigs}
          />
        )}

        <PlotAnalysisDialog
          open={!!selectedPlotForDialog}
          onClose={handleCloseDialog}
          plotTitle={selectedPlotForDialog ? plotConfigs[selectedPlotForDialog].title : ''}
          analysis={dialogAnalysis}
          loading={dialogLoading}
          error={dialogError}
        />
      </Box>
    </Fade>
  );
});

GraphDisplay.displayName = 'GraphDisplay';

export default GraphDisplay; 