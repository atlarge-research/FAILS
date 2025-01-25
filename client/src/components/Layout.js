import React, { useState, useEffect } from 'react';
import { 
  AppBar, 
  Box, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  Toolbar, 
  Typography,
  useTheme,
  useMediaQuery,
  Fab,
  Slide,
  Divider
} from '@mui/material';
import { 
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Analytics as AnalyticsIcon,
  TableChart as TableChartIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const drawerWidth = 240;

// Navigation items configuration
const navigationItems = [
  { 
    text: 'Dashboard', 
    icon: <DashboardIcon />, 
    path: '/',
    description: 'Overview and key metrics'
  },
  { 
    text: 'Data Table', 
    icon: <TableChartIcon />, 
    path: '/data',
    description: 'View the dataset'
  },
  { 
    text: 'Failure Analysis Chatbot', 
    icon: <AnalyticsIcon />, 
    path: '/failure-analysis',
    description: 'LLM-powered interaction'
  },
  { 
    text: 'Predictive Analysis', 
    icon: <TimelineIcon />, 
    path: '/predictive-analysis',
    description: 'Future incident predictions'
  },
  { 
    text: 'About', 
    icon: <InfoIcon />, 
    path: '/about',
    description: 'Further information'
  }
];

function Layout({ children, toggleTheme }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const NavigationList = () => (
    <List>
      {navigationItems.map((item) => (
        <ListItem
          key={item.path}
          component={Link}
          to={item.path}
          selected={location.pathname === item.path}
          sx={{
            borderRadius: 1,
            mx: 1,
            mb: 0.5,
            color: 'text.primary',
            '&.Mui-selected': {
              bgcolor: `${theme.palette.primary.main}15`,
              color: 'primary.main',
              '& .MuiListItemIcon-root': {
                color: 'primary.main',
              },
            },
            '&:hover': {
              bgcolor: `${theme.palette.primary.main}08`,
            },
          }}
        >
          <ListItemIcon sx={{ 
            minWidth: 40,
            color: location.pathname === item.path ? 'primary.main' : 'inherit'
          }}>
            {item.icon}
          </ListItemIcon>
          <ListItemText 
            primary={item.text}
            secondary={item.description}
            primaryTypographyProps={{
              variant: 'body2',
              fontWeight: location.pathname === item.path ? 600 : 400
            }}
            secondaryTypographyProps={{
              variant: 'caption',
              sx: { opacity: 0.7 }
            }}
          />
        </ListItem>
      ))}
    </List>
  );

  const drawer = (
    <Box sx={{ 
      height: '100%', 
      bgcolor: 'background.default', 
      display: 'flex', 
      flexDirection: 'column'
    }}>
      <Box sx={{ 
        display: 'flex',
        flexDirection: 'column'
      }}>
        <Box sx={{ 
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <Typography variant="h6" sx={{ fontSize: '2.2em', fontWeight: 'bold', color: 'primary.main' }}>
            FAILS
          </Typography>
          <IconButton onClick={toggleTheme} color="inherit" size="small">
            {theme.palette.mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Box>
        
        <Divider />
        
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ lineHeight: 1.4, fontSize: '1.1em' }}>
            <span style={{ color: theme.palette.primary.main, fontWeight: 'bold', fontSize: '1.2em' }}>F</span>ramework for the{' '}
            <span style={{ color: theme.palette.primary.main, fontWeight: 'bold', fontSize: '1.2em' }}>A</span>nalysis of{' '}
            <span style={{ color: theme.palette.primary.main, fontWeight: 'bold', fontSize: '1.2em' }}>I</span>ncidents and Outages of{' '}
            <span style={{ color: theme.palette.primary.main, fontWeight: 'bold', fontSize: '1.2em' }}>L</span>LM{' '}
            <span style={{ color: theme.palette.primary.main, fontWeight: 'bold', fontSize: '1.2em' }}>S</span>ervices
          </Typography>
        </Box>

        <Divider />

        <Box sx={{ overflow: 'auto', flex: 1, py: 2 }}>
          <NavigationList />
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          display: { sm: 'none' },
          bgcolor: scrolled ? 'background.default' : 'transparent',
          boxShadow: scrolled ? 1 : 'none',
          backdropFilter: scrolled ? 'blur(10px)' : 'none',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap sx={{ flex: 1 }}>
            LLM Analysis
          </Typography>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
              border: 'none',
              boxShadow: theme => isMobile ? theme.shadows[8] : 'none',
              bgcolor: 'background.default'
            }
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          bgcolor: 'background.default'
        }}
      >
        <Toolbar sx={{ display: { sm: 'none' } }} />
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </Box>
    </Box>
  );
}

export default Layout; 