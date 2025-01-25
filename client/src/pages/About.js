import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid,
  Card,
  CardContent,
  Avatar,
  Link,
  Divider
} from '@mui/material';
import { motion } from 'framer-motion';
import EmailIcon from '@mui/icons-material/Email';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

const features = [
  {
    title: 'Comprehensive Insights',
    description: 'Gain deep insights into the historical performance and reliability of various LLM services with our detailed analytics.'
  },
  {
    title: 'Multi-service Comparison',
    description: 'Easily compare multiple LLM services to find the best fit for your needs.'
  },
  {
    title: 'User-friendly Interface',
    description: 'Navigate through our intuitive dashboard designed for both beginners and experts.'
  }
];

const team = [
  {
    name: 'Bálint László Szarvas',
    role: 'Developer',
    email: 'b.l.szarvas@student.vu.nl',
    github: 'https://github.com/balintszarvas',
    linkedin: 'https://linkedin.com/in/balintszarvas',
    image: '/team/balint.jpg'
  },
  {
    name: 'Nishanthi Srinivasan',
    role: 'Developer',
    email: 'n.srinivasan@student.vu.nl',
    github: 'https://github.com/nishanthisrinivasan',
    linkedin: 'https://linkedin.com/in/nishanthisrinivasan',
    image: '/team/nishanthi.jpg'
  },
  {
    name: 'Sándor Battaglini-Fischer',
    role: 'Developer',
    email: 's.battaglini-fischer@student.vu.nl',
    github: 'https://github.com/sandor-battaglini-fischer',
    linkedin: 'https://www.linkedin.com/in/s%C3%A1ndor-battaglini-fischer-619b90221/',
    image: '/team/sandor.jpg'
  }
];

const llmServices = [
  {
    name: 'OpenAI',
    statusUrl: 'https://status.openai.com/',
    description: 'Status monitoring for OpenAI services including API, ChatGPT, DALL·E, and Playground.'
  },
  {
    name: 'Anthropic',
    statusUrl: 'https://status.anthropic.com/',
    description: 'Status updates for Anthropic services including API, Claude, and Console.'
  },
  {
    name: 'Character.AI',
    statusUrl: 'https://status.character.ai/',
    description: 'Service status and performance monitoring for Character.AI platform.'
  },
  {
    name: 'Stability AI',
    statusUrl: 'https://status.stability.ai/',
    description: 'Status monitoring for Stability AI services including Stable Diffusion.'
  },
  // {
  //   name: 'Google AI',
  //   statusUrl: 'https://status.cloud.google.com/',
  //   description: 'Status updates for Google AI services including Gemini, Gemini API, and Bard.'
  // }
];

function About() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', py: 4 }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Hero Section */}
        <Box 
          sx={{ 
            position: 'relative',
            mb: 6,
            p: 4,
            borderRadius: 4,
            background: theme => `linear-gradient(135deg, 
              ${theme.palette.primary.main}20, 
              ${theme.palette.secondary.main}20
            )`,
            overflow: 'hidden'
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Typography variant="h3" gutterBottom align="center">
              LLM Analysis Dashboard
            </Typography>
            <Typography variant="h6" align="center" color="text.secondary" sx={{ mb: 4 }}>
              A tool for comprehensive insights into historical incidents and outages of LLM services
            </Typography>
          </motion.div>
        </Box>

        {/* Main Content Grid */}
        <Grid container spacing={4}>
          {/* Left Column - Mission & Features */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4,
                  height: '100%',
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  border: 1,
                  borderColor: 'divider',
                  backgroundImage: theme => 
                    `linear-gradient(145deg, 
                      ${theme.palette.mode === 'dark' ? 
                        'rgba(16,39,68,0.6) 0%, rgba(16,39,68,0.3) 100%' : 
                        'rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.3) 100%'
                    })`,
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Our Mission
                </Typography>
                <Typography variant="body1" paragraph>
                  The LLM Analysis Dashboard aims to empower individuals, developers and organizations by providing comprehensive insights into the performance and reliability of various Language Learning Model services.
                </Typography>
                <Divider sx={{ my: 3 }} />
                <Typography variant="h5" gutterBottom>
                  Key Features
                </Typography>
                <Grid container spacing={2}>
                  {features.map((feature, index) => (
                    <Grid item xs={12} key={index}>
                      <Card 
                        elevation={0}
                        sx={{ 
                          p: 2,
                          bgcolor: 'background.card',
                          borderRadius: 2,
                          border: 1,
                          borderColor: 'divider',
                          transition: 'all 0.3s ease-in-out',
                          '&:hover': {
                            transform: 'translateX(8px)',
                            boxShadow: theme.shadows[4],
                          }
                        }}
                      >
                        <Typography variant="h6" gutterBottom>
                          {feature.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {feature.description}
                        </Typography>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </motion.div>
          </Grid>

          {/* Right Column - Services Status */}
          <Grid item xs={12} md={6}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4,
                  height: '100%',
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                  border: 1,
                  borderColor: 'divider',
                  backgroundImage: theme => 
                    `linear-gradient(145deg, 
                      ${theme.palette.mode === 'dark' ? 
                        'rgba(16,39,68,0.6) 0%, rgba(16,39,68,0.3) 100%' : 
                        'rgba(255,255,255,0.6) 0%, rgba(255,255,255,0.3) 100%'
                    })`,
                }}
              >
                <Typography variant="h4" gutterBottom>
                  Data Sources
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {llmServices.map((service, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 + index * 0.1 }}
                    >
                      <Card 
                        elevation={0}
                        sx={{ 
                          mb: 2,
                          bgcolor: 'background.card',
                          borderRadius: 2,
                          border: 1,
                          borderColor: 'divider',
                          transition: 'all 0.3s ease-in-out',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: theme.shadows[4],
                          }
                        }}
                      >
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {service.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" paragraph>
                            {service.description}
                          </Typography>
                          <Link 
                            href={service.statusUrl} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            sx={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              color: 'primary.main',
                              textDecoration: 'none',
                              '&:hover': {
                                textDecoration: 'underline',
                              }
                            }}
                          >
                            View Status Page
                          </Link>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </Box>
              </Paper>
            </motion.div>
          </Grid>
        </Grid>

        {/* Team Section */}
        <Box sx={{ mt: 6 }}>
          <Typography variant="h4" gutterBottom align="center" sx={{ mb: 4 }}>
            Meet the Team
          </Typography>
          <Grid container spacing={4}>
            {team.map((member, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                >
                  <Card 
                    elevation={0} 
                    sx={{ 
                      border: 1, 
                      borderColor: 'divider',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column'
                    }}
                  >
                    <Box
                      sx={{
                        position: 'relative',
                        paddingTop: '100%', // 1:1 Aspect ratio
                        overflow: 'hidden'
                      }}
                    >
                      {member.image ? (
                        <Box
                          component="img"
                          src={member.image}
                          alt={member.name}
                          sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            transition: 'transform 0.3s ease-in-out',
                            '&:hover': {
                              transform: 'scale(1.05)'
                            }
                          }}
                          onError={(e) => {
                            // Fallback to Avatar if image fails to load
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <Box
                        sx={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          display: member.image ? 'none' : 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: 'action.selected'
                        }}
                      >
                        <Avatar
                          sx={{
                            width: '60%',
                            height: '60%',
                            fontSize: '3rem',
                            bgcolor: 'primary.main'
                          }}
                        >
                          {member.name.split(' ').map(n => n[0]).join('')}
                        </Avatar>
                      </Box>
                    </Box>

                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {member.name}
                      </Typography>
                      <Typography 
                        variant="subtitle2" 
                        color="textSecondary" 
                        gutterBottom
                        sx={{ mb: 2 }}
                      >
                        {member.role}
                      </Typography>
                      <Divider sx={{ my: 2 }} />
                      <Box 
                        sx={{ 
                          display: 'flex', 
                          gap: 2, 
                          justifyContent: 'center',
                          '& a': {
                            transition: 'all 0.2s ease-in-out',
                            '&:hover': {
                              transform: 'translateY(-2px)',
                              color: 'primary.main'
                            }
                          }
                        }}
                      >
                        <Link 
                          href={`mailto:${member.email}`} 
                          color="inherit"
                          title="Email"
                        >
                          <EmailIcon />
                        </Link>
                        <Link 
                          href={member.github} 
                          target="_blank" 
                          color="inherit"
                          title="GitHub"
                        >
                          <GitHubIcon />
                        </Link>
                        <Link 
                          href={member.linkedin} 
                          target="_blank" 
                          color="inherit"
                          title="LinkedIn"
                        >
                          <LinkedInIcon />
                        </Link>
                      </Box>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Footer */}
        <Box sx={{ mt: 6, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Disclaimer: The information provided by the LLM Analysis Dashboard is for general informational purposes only.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            © {new Date().getFullYear()} LLM Analysis Dashboard. All rights reserved.
          </Typography>
        </Box>
      </motion.div>
    </Box>
  );
}

export default About; 