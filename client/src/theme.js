import { createTheme, alpha } from '@mui/material/styles';

const generateGradient = (color1, color2, opacity = 0.1) => 
  `linear-gradient(135deg, 
    ${color1 + Math.floor(opacity * 255).toString(16)} 0%, 
    ${color2 + Math.floor(opacity * 255).toString(16)} 100%)`;

const generateGlassmorphism = (mode, opacity = 0.1, blur = 10) => ({
  backgroundColor: mode === 'light' 
    ? 'rgba(255, 255, 255, 0.7)'
    : 'rgba(10, 25, 41, 0.7)',
  backdropFilter: `blur(${blur}px)`,
  borderColor: mode === 'light'
    ? 'rgba(255, 255, 255, 0.3)'
    : 'rgba(255, 255, 255, 0.05)',
  boxShadow: mode === 'light'
    ? '0 8px 32px rgba(0, 0, 0, 0.1)'
    : '0 8px 32px rgba(0, 0, 0, 0.3)',
});

// Modern color palette with gradients
const colors = {
  primary: {
    main: '#3B82F6',
    light: '#60A5FA',
    dark: '#2563EB',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#6366F1',
    light: '#818CF8',
    dark: '#4F46E5',
    contrastText: '#ffffff',
  },
  success: {
    main: '#10B981',
    light: '#34D399',
    dark: '#059669',
  },
  error: {
    main: '#EF4444',
    light: '#F87171',
    dark: '#DC2626',
  },
};

const commonComponents = {
  MuiCard: {
    styleOverrides: {
      root: {
        borderRadius: 16,
        transition: 'all 0.3s ease-in-out',
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        textTransform: 'none',
        fontWeight: 600,
        padding: '8px 24px',
        backdropFilter: 'blur(10px)',
      },
      contained: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
          transform: 'translateY(-2px)',
        },
      },
    },
  },
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        fontWeight: 500,
        backdropFilter: 'blur(8px)',
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: {
        borderRadius: 16,
      },
    },
  },
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 12,
          backdropFilter: 'blur(8px)',
        },
      },
    },
  },
  MuiAppBar: {
    styleOverrides: {
      root: {
        backdropFilter: 'blur(10px)',
        backgroundColor: 'transparent',
      },
    },
  },
  MuiDrawer: {
    styleOverrides: {
      paper: {
        backdropFilter: 'blur(10px)',
        backgroundColor: 'transparent',
      },
    },
  },
};

// Light theme
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: colors.primary,
    secondary: colors.secondary,
    success: colors.success,
    error: colors.error,
    background: {
      default: '#F8F9FC',
      paper: alpha('#FFFFFF', 0.7),
      card: alpha('#FFFFFF', 0.5),
      gradient: generateGradient(colors.primary.main, colors.secondary.main, 0.05),
      gradientStrong: generateGradient(colors.primary.main, colors.secondary.main, 0.1),
    },
    text: {
      primary: '#1A1A1A',
      secondary: '#666666',
    },
    divider: 'rgba(0, 0, 0, 0.08)',
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    ...commonComponents,
    MuiCard: {
      styleOverrides: {
        root: {
          ...commonComponents.MuiCard.styleOverrides.root,
          backgroundColor: alpha('#FFFFFF', 0.7),
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          ...commonComponents.MuiPaper.styleOverrides.root,
          backgroundColor: alpha('#FFFFFF', 0.7),
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.3)',
        },
      },
    },
  },
});

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: colors.primary,
    secondary: colors.secondary,
    success: colors.success,
    error: colors.error,
    background: {
      default: '#0F172A',
      paper: alpha('#1E293B', 0.7),
      card: alpha('#1E293B', 0.5),
      gradient: generateGradient(colors.primary.light, colors.secondary.light, 0.08),
      gradientStrong: generateGradient(colors.primary.light, colors.secondary.light, 0.12),
    },
    text: {
      primary: '#F1F5F9',
      secondary: '#94A3B8',
    },
    divider: 'rgba(148, 163, 184, 0.08)',
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    ...commonComponents,
    MuiCard: {
      styleOverrides: {
        root: {
          ...commonComponents.MuiCard.styleOverrides.root,
          backgroundColor: alpha('#0F2744', 0.7),
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.05)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          ...commonComponents.MuiPaper.styleOverrides.root,
          backgroundColor: alpha('#0F2744', 0.7),
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.05)',
        },
      },
    },
  },
});

export { lightTheme, darkTheme }; 