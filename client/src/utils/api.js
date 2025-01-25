export const handleApiError = (error) => {
    if (!error.response) {
        // Network error or server not running
        return {
            success: false,
            error: "Please use production server. The development server is not running.",
        };
    }
    // Method not allowed error (405)
    if (error.status === 405) {
        return {
            success: false,
            error: "Please use production server. This feature is only available in production.",
        };
    }
    // Handle other types of errors...
    return {
        success: false,
        error: error.message,
    };
};

// Example usage in your API calls:
export const analyzeData = async (data) => {
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        return handleApiError(error);
    }
};

const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const fetchWithRetry = async (url, options, retries = MAX_RETRIES) => {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  } catch (error) {
    if (retries > 0 && (error.message.includes('ECONNRESET') || error.message.includes('Failed to fetch'))) {
      console.log(`Retrying... ${retries} attempts remaining`);
      await sleep(RETRY_DELAY);
      return fetchWithRetry(url, options, retries - 1);
    }
    throw error;
  }
};

export const analyzePlot = async (plotUrl, plotType, startDate, endDate, services) => {
  try {
    // Fetch the image data from the plot URL with retry
    const imageResponse = await fetchWithRetry(plotUrl);
    const blob = await imageResponse.blob();
    
    // Convert blob to base64
    const base64 = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
      reader.readAsDataURL(blob);
    });

    // Send the base64 image to the analysis endpoint with retry
    const response = await fetchWithRetry('/api/analyze-plot', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        image: base64,
        plotType,
        startDate,
        endDate,
        services
      }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to analyze plot');
    }
    
    return data;
  } catch (error) {
    console.error('Error analyzing plot:', error);
    throw error;
  }
};

export const summarizeAnalyses = async (analyses, startDate, endDate, services) => {
  try {
    const response = await fetch('/api/summarize-analyses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        analyses,
        startDate,
        endDate,
        services
      }),
    });

    if (response.status === 405) {
      throw new Error("Please use production server. This feature is only available in production.");
    }

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to summarize analyses');
    }
    
    return data;
  } catch (error) {
    console.error('Error summarizing analyses:', error);
    throw error;
  }
}; 