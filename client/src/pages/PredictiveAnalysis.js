import React from 'react';
import { Box, Typography } from '@mui/material';

const PredictiveAnalysis = () => {
    return (
        <Box sx={{
            p: 3,
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
        }}>
            <Typography variant="h4" gutterBottom>
                FOR FUTURE IMPLEMENTATION
            </Typography>
            <Typography variant="body1">
                Predictive analysis of how long a service will be down based for, would require real-time data.
            </Typography>
        </Box>
    );
};

export default PredictiveAnalysis;