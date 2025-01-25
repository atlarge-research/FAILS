import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Collapse,
  Alert,
  Divider,
  Card,
  CardContent,
  IconButton,
  Grid,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Lightbulb as LightbulbIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { analyzePlot } from '../utils/api';

const PlotAnalysis = ({ plots, plotConfigs, startDate, endDate, services }) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState([]);
  const [error, setError] = useState(null);
  const [showAnalysis, setShowAnalysis] = useState(true);

  const handleAnalyzeAll = async () => {
    setAnalyzing(true);
    setError(null);
    
    try {
      // First analyze all plots individually with their specific prompts
      const individualAnalyses = await Promise.all(
        Object.entries(plots).map(async ([figureId, plotUrl]) => {
          const result = await analyzePlot(plotUrl, figureId, startDate, endDate, services);
          return {
            id: figureId,
            title: plotConfigs[figureId].title,
            analysis: result.success ? result.analysis : null,
            success: result.success
          };
        })
      );

      // Filter successful analyses
      const successfulAnalyses = individualAnalyses
        .filter(a => a.success && a.analysis)
        .map(a => ({ 
          title: a.title, 
          analysis: a.analysis 
        }));

      // Get the summarized analysis with context
      const summaryResult = await fetch('/api/summarize-analyses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          analyses: successfulAnalyses,
          startDate,
          endDate,
          services
        })
      });

      const summary = await summaryResult.json();

      if (summary.success) {
        setAnalysis([{
          id: 'summary',
          title: 'Comprehensive Analysis of All Plots',
          analysis: summary.analysis,
          success: true
        }]);
      } else {
        throw new Error(summary.error);
      }
      setShowAnalysis(true);
    } catch (error) {
      setError(error.message || 'Failed to analyze plots. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const formatMarkdown = (text) => {
    return text
      .replace(/##\s*(.*?)\n/g, '<span class="markdown-h2">$1</span><br/>')
      .replace(/###\s*(.*?)\n/g, '<span class="markdown-h3">$1</span><br/>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br />');
  };

  return (
    <Paper
      elevation={3}
      sx={{
        mt: 4,
        p: 3,
        borderRadius: 2,
        backgroundColor: 'background.paper',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <AnalyticsIcon sx={{ mr: 2, color: 'primary.main' }} />
        <Typography variant="h6" component="h2">
          Comprehensive AI Analysis
        </Typography>
      </Box>

      <Divider sx={{ mb: 3 }} />

      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
        <Button
          variant="contained"
          onClick={handleAnalyzeAll}
          disabled={analyzing}
          startIcon={analyzing ? <CircularProgress size={20} /> : <LightbulbIcon />}
          sx={{
            minWidth: 250,
            borderRadius: 2,
            py: 1.5,
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: 3,
            },
          }}
        >
          {analyzing ? 'Analyzing All Plots...' : 'Analyze All Plots'}
        </Button>
      </Box>

      <Collapse in={!!error}>
        <Alert
          severity="error"
          action={
            <IconButton
              aria-label="close"
              color="inherit"
              size="small"
              onClick={() => setError(null)}
            >
              <CloseIcon fontSize="inherit" />
            </IconButton>
          }
          sx={{ mb: 2, borderRadius: 2 }}
        >
          {error}
        </Alert>
      </Collapse>

      <Collapse in={analysis.length > 0 && showAnalysis}>
        <Card 
          variant="outlined"
          sx={{ 
            borderRadius: 2,
            backgroundColor: 'background.default',
            position: 'relative'
          }}
        >
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <LightbulbIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6" component="h3" color="primary">
                  Analysis Results
                </Typography>
              </Box>
              <IconButton
                size="small"
                onClick={() => setShowAnalysis(false)}
                sx={{ mt: -1, mr: -1 }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>

            <Grid container spacing={2}>
              {analysis.map((item) => (
                <Grid item xs={12} key={item.id}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      p: 2,
                      backgroundColor: 'background.paper',
                      '&:hover': {
                        boxShadow: 1,
                      },
                    }}
                  >
                    <Typography 
                      variant="subtitle1" 
                      color="primary" 
                      sx={{ 
                        fontWeight: 'bold',
                        mb: 1,
                        display: 'flex',
                        alignItems: 'center',
                      }}
                    >
                      {item.title}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        whiteSpace: 'pre-line',
                        '& strong': {
                          fontWeight: 600,
                          color: 'primary.main',
                        },
                        '& .markdown-h3': {
                          display: 'block',
                          fontSize: '1.1rem',
                          fontWeight: 600,
                          color: 'text.primary',
                          mt: 2,
                          mb: 1,
                        },
                        '& p': {
                          mb: 1.5,
                        },
                        '& ul, & ol': {
                          pl: 2,
                          mb: 1.5,
                        },
                      }}
                      dangerouslySetInnerHTML={{ 
                        __html: formatMarkdown(item.analysis)
                      }}
                    />
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      </Collapse>
    </Paper>
  );
};

export default PlotAnalysis; 