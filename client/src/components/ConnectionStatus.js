import React, { useState, useEffect } from 'react';

const ConnectionStatus = () => {
    const [isConnected, setIsConnected] = useState(true);

    useEffect(() => {
        const checkConnection = async () => {
            try {
                const response = await fetch('/api/health');
                setIsConnected(response.ok);
            } catch (error) {
                setIsConnected(false);
            }
        };

        // Check immediately and then every 30 seconds
        checkConnection();
        const interval = setInterval(checkConnection, 60000);

        return () => clearInterval(interval);
    }, []);

    if (isConnected) return null; // Don't show anything when connected

    return (
        <div style={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            backgroundColor: '#ff4444',
            color: 'white',
            padding: '10px 20px',
            borderRadius: '5px',
            zIndex: 9999,
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            pointerEvents: 'none'
        }}>
            Backend Server Not Connected
        </div>
    );
};

export default ConnectionStatus; 