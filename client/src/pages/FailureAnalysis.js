import React, { useState } from 'react';
import { 
    TextField, 
    Button, 
    Typography, 
    CircularProgress,
    Box,
    Card,
    CardContent,
    Divider,
    List,
    ListItem,
    Paper,
    Avatar
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import PersonIcon from '@mui/icons-material/Person';
import PsychologyIcon from '@mui/icons-material/Psychology';

const FailureAnalysis = () => {
    const [query, setQuery] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [loading, setLoading] = useState(false);
    const theme = useTheme();

    const handleSubmit = async () => {
        if (!query.trim()) return;
        
        setLoading(true);
        const userMessage = query;
        setQuery('');
        
        // Add user message to chat
        setChatHistory(prev => [...prev, {
            role: 'user',
            content: userMessage
        }]);

        try {
            const response = await fetch('/api/analyze-failures', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: userMessage,
                    history: chatHistory 
                }),
            });
            
            const data = await response.json();
            
            // Add assistant response to chat
            setChatHistory(prev => [...prev, {
                role: 'assistant',
                content: data.success ? data.analysis : `Error: ${data.error}`
            }]);
            
        } catch (error) {
            setChatHistory(prev => [...prev, {
                role: 'assistant',
                content: 'Error analyzing failures. Please try again.'
            }]);
        }
        
        setLoading(false);
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
        }
    };

    return (
        <Box sx={{ p: 3, height: '100vh', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h4" gutterBottom>
                LLM Service Incident Analysis Chat
            </Typography>
            
            {/* Initial instructions card */}
            {chatHistory.length === 0 && (
                <Card sx={{ mb: 4 }}>
                    <CardContent>
                        <Typography variant="body1" gutterBottom>
                            Ask questions about failure patterns, service reliability, and incident trends.
                            Examples:
                        </Typography>
                        <Typography variant="body2" color="text.secondary" component="div" sx={{ mb: 2 }}>
                            • "What are the most common types of failures?"<br/>
                            • "Which services had the longest downtime?"<br/>
                            • "Analyze the pattern of authentication failures"<br/>
                            • "Tell me more about the impact levels of incidents"
                        </Typography>
                    </CardContent>
                </Card>
            )}
            
            {/* Chat messages */}
            <Box sx={{ 
                flex: 1, 
                overflow: 'auto', 
                mb: 2,
                bgcolor: theme.palette.background.default,
                borderRadius: 1
            }}>
                <List>
                    {chatHistory.map((message, index) => (
                        <ListItem 
                            key={index}
                            sx={{ 
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                                padding: 2
                            }}
                        >
                            <Box sx={{ 
                                display: 'flex',
                                alignItems: 'flex-start',
                                maxWidth: '80%'
                            }}>
                                <Avatar 
                                    sx={{ 
                                        mr: 1,
                                        bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main'
                                    }}
                                >
                                    {message.role === 'user' ? <PersonIcon /> : <PsychologyIcon />}
                                </Avatar>
                                <Paper 
                                    elevation={1}
                                    sx={{ 
                                        p: 2,
                                        bgcolor: message.role === 'user' ? 'primary.light' : 'background.paper',
                                        borderRadius: 2
                                    }}
                                >
                                    {message.role === 'user' ? (
                                        <Typography>{message.content}</Typography>
                                    ) : (
                                        <ReactMarkdown>{message.content}</ReactMarkdown>
                                    )}
                                </Paper>
                            </Box>
                        </ListItem>
                    ))}
                    {loading && (
                        <ListItem sx={{ justifyContent: 'center' }}>
                            <CircularProgress size={24} />
                        </ListItem>
                    )}
                </List>
            </Box>
            
            {/* Input area */}
            <Paper
                elevation={3}
                sx={{
                    p: 2,
                    bgcolor: 'background.paper'
                }}
            >
                <TextField
                    fullWidth
                    label="Ask about failure patterns..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    multiline
                    rows={2}
                    sx={{ mb: 1 }}
                />
                <Button 
                    variant="contained" 
                    onClick={handleSubmit}
                    disabled={loading || !query.trim()}
                >
                    Send
                </Button>
            </Paper>
        </Box>
    );
};

export default FailureAnalysis;