import React, { useState, useEffect, useRef } from 'react';
import { 
  Typography, 
  Paper,
  Chip,
  Box,
  TextField,
  Button,
  Slide,
  AppBar,
  Toolbar,
  IconButton,
  Collapse,
  useTheme,
  useMediaQuery,
  alpha,
  ButtonGroup,
  Alert,
  Snackbar,
  Tooltip
} from '@mui/material';
import {
  AnalyticsOutlined as AnalyticsIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
  DateRange as DateRangeIcon,
  Apps as AppsIcon,
  Refresh as RefreshIcon,
  KeyboardArrowUp as KeyboardArrowUpIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import GraphDisplay from '../components/GraphDisplay';
import { useAnalysis } from '../context/AnalysisContext';

function Dashboard() {
  const {
    selectedServices,
    setSelectedServices,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    loading,
    error,
    setError,
    handleAnalyze,
    resetAnalysis
  } = useAnalysis();
  
  const [controlsOpen, setControlsOpen] = useState(true);
  const [scrolled, setScrolled] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const graphDisplayRef = useRef(null);
  const controlsRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!controlsRef.current) return;

      const controlsRect = controlsRef.current.getBoundingClientRect();
      const graphsTop = window.innerHeight * 0.3; // Close when graphs reach 30% of viewport height

      // Close controls when graphs would overlap with them
      if (controlsRect.bottom > graphsTop && controlsOpen) {
        setControlsOpen(false);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [controlsOpen]);

  const providers = {
    'OpenAI': ['API', 'ChatGPT', 'DALL-E', 'Playground'],
    'Anthropic': ['API', 'Claude', 'Console'],
    'Character.AI': ['Character.AI'],
    'Stability AI': ['REST API', 'gRPC API', 'Stable Assistant'],
    // 'Google': ['Gemini', 'Gemini API', 'Bard']
  };

  const handleServiceToggle = (provider, service) => {
    let serviceId;
    switch(true) {
      case provider === 'Anthropic' && service === 'API':
        serviceId = 'Anthropic:API';
        break;
      case provider === 'Anthropic' && service === 'Claude':
        serviceId = 'Anthropic:Claude';
        break;
      case provider === 'Anthropic' && service === 'Console':
        serviceId = 'Anthropic:Console';
        break;
      case provider === 'Character.AI' && service === 'Character.AI':
        serviceId = 'Character.AI:Character.AI';
        break;
      case provider === 'Stability AI' && service === 'REST API':
        serviceId = 'StabilityAI:REST';
        break;
      case provider === 'Stability AI' && service === 'gRPC API':
        serviceId = 'StabilityAI:gRPC';
        break;
      case provider === 'Stability AI' && service === 'Stable Assistant':
        serviceId = 'StabilityAI:Assistant';
        break;
      case provider === 'Google':
        serviceId = `Google:${service}`;
        break;
      default:
        serviceId = `OpenAI:${service}`;
    }

    setSelectedServices(prev => 
      prev.includes(serviceId)
        ? prev.filter(s => s !== serviceId)
        : [...prev, serviceId]
    );
  };

  const isServiceSelected = (provider, service) => {
    let serviceId;
    switch(true) {
      case provider === 'Anthropic' && service === 'API':
        serviceId = 'Anthropic:API';
        break;
      case provider === 'Anthropic' && service === 'Claude':
        serviceId = 'Anthropic:Claude';
        break;
      case provider === 'Anthropic' && service === 'Console':
        serviceId = 'Anthropic:Console';
        break;
      case provider === 'Character.AI' && service === 'Character.AI':
        serviceId = 'Character.AI:Character.AI';
        break;
      case provider === 'Stability AI' && service === 'REST API':
        serviceId = 'StabilityAI:REST';
        break;
      case provider === 'Stability AI' && service === 'gRPC API':
        serviceId = 'StabilityAI:gRPC';
        break;
      case provider === 'Stability AI' && service === 'Stable Assistant':
        serviceId = 'StabilityAI:Assistant';
        break;
      case provider === 'Google':
        serviceId = `Google:${service}`;
        break;
      default:
        serviceId = `OpenAI:${service}`;
    }
    return selectedServices.includes(serviceId);
  };

  const isDateRangeValid = () => {
    return startDate && endDate && new Date(startDate) <= new Date(endDate);
  };

  const isFormValid = () => {
    return selectedServices.length > 0 && isDateRangeValid();
  };

  const handleToolbarClick = () => {
    if (scrolled) {
      setControlsOpen(!controlsOpen);
    }
  };

  const handleSelectAll = (provider) => {
    const providerServices = providers[provider];
    const serviceIds = providerServices.map(service => {
      switch(true) {
        case provider === 'Anthropic' && service === 'API':
          return 'Anthropic:API';
        case provider === 'Anthropic' && service === 'Claude':
          return 'Anthropic:Claude';
        case provider === 'Anthropic' && service === 'Console':
          return 'Anthropic:Console';
        case provider === 'Character.AI' && service === 'Character.AI':
          return 'Character.AI:Character.AI';
        case provider === 'Stability AI' && service === 'REST API':
          return 'StabilityAI:REST';
        case provider === 'Stability AI' && service === 'gRPC API':
          return 'StabilityAI:gRPC';
        case provider === 'Stability AI' && service === 'Stable Assistant':
          return 'StabilityAI:Assistant';
        case provider === 'Google':
          return `Google:${service}`;
        default:
          return `OpenAI:${service}`;
      }
    });

    setSelectedServices(prev => {
      const currentProviderServices = serviceIds.filter(id => prev.includes(id));
      if (currentProviderServices.length === serviceIds.length) {
        // If all services of this provider are selected, deselect them
        return prev.filter(id => !serviceIds.includes(id));
      } else {
        // Otherwise, select all services of this provider
        const otherServices = prev.filter(id => !serviceIds.includes(id));
        return [...otherServices, ...serviceIds];
      }
    });
  };

  const handleReset = () => {
    resetAnalysis();
    setControlsOpen(true);
  };

  return (
    <Box sx={{ 
      height: '100%', 
      mt: { xs: -2, sm: -3 },
      pt: { xs: 2, sm: 3 },
      position: 'relative',
      '&::before': {
        content: '""',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: theme => `${theme.palette.background.gradient}`,
        zIndex: -1,
      }
    }}>
      <Box
        ref={controlsRef}
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: theme => theme.zIndex.drawer + 3,
          mb: 2,
        }}
      >
        <Paper
          elevation={scrolled ? 2 : 0}
          sx={{
            position: 'relative',
            borderRadius: '0 0 16px 16px',
            overflow: 'visible',
            transition: 'all 0.3s ease-in-out',
          }}
        >
          <Box
            onClick={() => setControlsOpen(!controlsOpen)}
            sx={{
              position: 'absolute',
              bottom: -24,
              left: '50%',
              transform: 'translateX(-50%)',
              width: 48,
              height: 24,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              bgcolor: 'background.paper',
              borderRadius: '0 0 24px 24px',
              border: 1,
              borderTop: 0,
              borderColor: 'divider',
              transition: 'all 0.3s ease-in-out',
              '&:hover': {
                bgcolor: 'action.hover',
                transform: 'translateX(-50%) translateY(2px)',
              },
              zIndex: 1,
            }}
          >
            <motion.div
              animate={{ rotate: controlsOpen ? 0 : 180 }}
              transition={{ duration: 0.3 }}
            >
              <KeyboardArrowUpIcon 
                fontSize="small" 
                sx={{ 
                  color: 'text.secondary',
                  transition: 'color 0.3s ease-in-out',
                  '&:hover': {
                    color: 'primary.main',
                  }
                }}
              />
            </motion.div>
          </Box>

          <Box sx={{ overflow: 'hidden' }}>
            <Collapse in={controlsOpen}>
              <AppBar 
                position="static" 
                elevation={0}
                sx={{ 
                  borderBottom: 1, 
                  borderColor: 'divider',
                  backgroundColor: theme => alpha(theme.palette.background.paper, 0.7),
                  backdropFilter: 'blur(10px)',
                  transition: 'all 0.3s ease-in-out',
                }}
              >
                <Toolbar sx={{ 
                  flexDirection: { xs: 'column', sm: 'row' }, 
                  gap: { xs: 1, sm: 2 },
                  py: { xs: 1, sm: 0 },
                  minHeight: 48,
                }}>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    width: '100%',
                    gap: { xs: 1, sm: 2 }
                  }}>
                    <Typography variant="h6" sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 1,
                      color: 'text.primary',
                      fontSize: { xs: '1rem', sm: scrolled ? '1rem' : '1.25rem' },
                      transition: 'all 0.3s ease-in-out',
                    }}>
                      Analysis Controls
                      {scrolled && (
                        <motion.div
                          animate={{ rotate: controlsOpen ? 180 : 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <KeyboardArrowDownIcon />
                        </motion.div>
                      )}
                    </Typography>

                    <Box sx={{ 
                      display: 'flex', 
                      gap: { xs: 1, sm: 2 },
                      ml: { xs: 0, sm: 'auto' }
                    }}>
                      <Tooltip title="Reset Analysis">
                        <IconButton 
                          onClick={handleReset}
                          disabled={loading}
                          size={isMobile ? "small" : "medium"}
                          sx={{
                            transition: 'all 0.2s ease-in-out',
                            '&:hover': {
                              transform: 'rotate(180deg)',
                            }
                          }}
                        >
                          <RefreshIcon />
                        </IconButton>
                      </Tooltip>
                      <motion.div
                        whileHover={{ scale: isFormValid() ? 1.02 : 1 }}
                        whileTap={{ scale: isFormValid() ? 0.98 : 1 }}
                      >
                        <Button
                          variant="contained"
                          color="primary"
                          startIcon={<AnalyticsIcon />}
                          onClick={handleAnalyze}
                          disabled={!isFormValid() || loading}
                          size={isMobile ? "small" : "medium"}
                          sx={{ 
                            minWidth: { xs: 'auto', sm: 150 },
                            height: 'fit-content',
                            alignSelf: 'center',
                            background: theme => isFormValid()
                              ? `linear-gradient(135deg, 
                                  ${theme.palette.primary.main} 0%, 
                                  ${theme.palette.secondary.main} 100%)`
                              : theme.palette.action.disabledBackground,
                            color: theme => isFormValid()
                              ? '#ffffff'
                              : theme.palette.text.disabled,
                            backdropFilter: 'blur(10px)',
                            border: '1px solid',
                            borderColor: theme => isFormValid()
                              ? 'rgba(255,255,255,0.2)'
                              : 'divider',
                            transition: 'all 0.3s ease-in-out',
                            '&:hover': isFormValid() ? {
                              background: theme => `linear-gradient(135deg, 
                                ${theme.palette.primary.dark} 0%, 
                                ${theme.palette.secondary.dark} 100%)`,
                              transform: 'translateY(-2px)',
                              boxShadow: theme => `0 8px 24px ${alpha(theme.palette.primary.main, 0.25)}`,
                            } : {},
                          }}
                        >
                          {loading ? 'Analyzing...' : 'Analyze'}
                        </Button>
                      </motion.div>
                    </Box>
                  </Box>
                </Toolbar>

                <Collapse in={controlsOpen}>
                  <Box sx={{ p: 2 }}>
                    <Box sx={{ 
                      display: 'flex', 
                      gap: 2, 
                      mb: 2,
                      flexWrap: 'wrap',
                      alignItems: 'flex-start'
                    }}>
                      <Paper 
                        elevation={0} 
                        sx={{ 
                          p: 2, 
                          border: 1, 
                          borderColor: 'divider',
                          minWidth: 280,
                          backdropFilter: 'blur(10px)',
                          backgroundColor: theme => alpha(theme.palette.background.paper, 0.7),
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <DateRangeIcon color="primary" sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Analysis Period</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <TextField
                            label="Start Date"
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            InputLabelProps={{ shrink: true }}
                            size="small"
                            fullWidth
                          />
                          <TextField
                            label="End Date"
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            InputLabelProps={{ shrink: true }}
                            size="small"
                            fullWidth
                          />
                        </Box>
                      </Paper>

                      <Paper 
                        elevation={0} 
                        sx={{ 
                          p: 2, 
                          border: 1, 
                          borderColor: 'divider',
                          flexGrow: 1,
                          backdropFilter: 'blur(10px)',
                          backgroundColor: theme => alpha(theme.palette.background.paper, 0.7),
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <AppsIcon color="primary" sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Select Services</Typography>
                        </Box>
                        <Box sx={{ p: 2 }}>
                          {Object.entries(providers).map(([provider, services]) => (
                            <Box key={provider}>
                              <Box sx={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                justifyContent: 'space-between',
                                mb: 1 
                              }}>
                                <Typography 
                                  variant="caption" 
                                  sx={{ 
                                    color: 'text.secondary',
                                    fontWeight: 500
                                  }}
                                >
                                  {provider}
                                </Typography>
                                <Button
                                  size="small"
                                  variant="text"
                                  onClick={() => handleSelectAll(provider)}
                                  sx={{
                                    minWidth: 'auto',
                                    fontSize: '0.75rem',
                                    color: 'text.secondary',
                                    '&:hover': {
                                      color: 'primary.main',
                                      backgroundColor: 'transparent',
                                    }
                                  }}
                                >
                                  {services.every(service => 
                                    isServiceSelected(provider, service)) ? 'Deselect All' : 'Select All'}
                                </Button>
                              </Box>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {services.map((service) => (
                                  <motion.div
                                    key={`${provider}:${service}`}
                                    whileHover={{ y: -2 }}
                                    whileTap={{ scale: 0.95 }}
                                  >
                                    <Chip
                                      label={service}
                                      onClick={() => handleServiceToggle(provider, service)}
                                      color={isServiceSelected(provider, service) ? "primary" : "default"}
                                      variant={isServiceSelected(provider, service) ? "filled" : "outlined"}
                                      size="small"
                                      sx={{
                                        backdropFilter: 'blur(8px)',
                                        backgroundColor: theme => isServiceSelected(provider, service) 
                                          ? alpha(theme.palette.primary.main, 0.9)
                                          : alpha(theme.palette.background.paper, 0.5),
                                        border: '1px solid',
                                        borderColor: 'divider',
                                      }}
                                    />
                                  </motion.div>
                                ))}
                              </Box>
                            </Box>
                          ))}
                        </Box>
                      </Paper>
                    </Box>
                  </Box>
                </Collapse>
              </AppBar>
            </Collapse>
          </Box>
        </Paper>
      </Box>

      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setError(null)} 
          severity="error" 
          variant="filled"
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>

      <Box sx={{ 
        mt: 2,
        position: 'relative',
        zIndex: theme => theme.zIndex.drawer + 2
      }}>
        <GraphDisplay 
          ref={graphDisplayRef} 
          loading={loading} 
          selectedServices={selectedServices} 
          startDate={startDate} 
          endDate={endDate} 
        />
      </Box>
    </Box>
  );
}

export default Dashboard;