# FAILS
**A Framework for Automated Collection and Analysis of Incidents on LLM Services**

This repository contains the web application for the FAILS project. It is built using React for the frontend and Flask for the backend.

> Large Language Model (LLM) services have rapidly become essential tools for applications ranging from customer support to content generation, yet their distributed nature makes them prone to failures that impact reliability and uptime. Existing tools for analysing service incidents are either closed-source, lack comparative capabilities, or fail to provide comprehensive insights into failure trends and recovery patterns. To address these gaps, we present FAILS (Framework for Analysis of Incidents and Outages of LLM Services), an open-source system designed to collect, analyse and visualize incident data from leading LLM providers. FAILS enables users to explore temporal trends, assess reliability metrics associated with failure models such as Mean Time to Recovery (MTTR) and Mean Time Between Failures (MTBF), and gain insights into service co-dependencies using a modern LLM-assisted analysis. With a web-based interface and advanced plotting tools, FAILS enables researchers, engineers, and decision-makers to understand and mitigate disruptions due to LLM services.

## Getting it running

### Prerequisites

- Node.js and npm
- Python 3.11 (tested with 3.12 and 3.13, didn't work!)
- OpenAI API key (not system critical but needed for AI plot analysis feature)

### Installation

1. **Install Node.js and npm:**

   If you haven't installed Node.js and npm, download and install them from the [official Node.js website](https://nodejs.org/). This will also install npm, which is the package manager for Node.js.

2. **Install frontend dependencies:**

   Navigate to the `client` directory and install the dependencies:

   ```bash
   cd client
   npm install
   ```

3. **Set up Python virtual environment:**

   Navigate to the `llm_analysis` directory and create a virtual environment:

   ```bash
   cd llm_analysis
   python -m venv venv
   ```

   Activate the virtual environment:

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

4. **Install backend dependencies:**

   With the virtual environment activated, install the dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables:**

   Create a `.env` file in the `server/scripts` directory with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

### Running the Application

1. **Start the backend server:**

   #### Development Mode

   For development with auto-reload:

   In the `server` directory, ensure the virtual environment is activated, then run:

   ```bash
   python app.py
   ```

   This will start the Flask server on `http://localhost:5000`.

   #### Production Mode

   For production deployment using Gunicorn:

   ```bash
   cd server
   chmod +x start.sh stop.sh # Make scripts executable (first time only)
   ./start.sh # Start the server
   ./stop.sh # Stop the server when needed
   ```

   The server will be available at `http://localhost:5000`.

2. **Start the frontend development server:**

   In the `client` directory, run:

   ```bash
   npm start
   ```

   This will start the React development server on `http://localhost:3000`.

## Features

### Dashboard

The main dashboard provides visualization and analysis of LLM service incidents through various plots and metrics.

### Failure Analysis Chat

An interactive chat interface that allows users to analyze incident patterns and get AI-powered insights about service reliability. The chat interface:

- Maintains conversation context for follow-up questions
- Provides markdown-formatted responses
- Supports natural language queries about:
  - Common failure patterns
  - Service reliability trends
  - Impact analysis
  - Recovery time patterns
  - Root cause categorization

Example queries:

- "Sort the service providers by number of incidents in total in the entire dataset and give the timeframe!"
- "Tell me more about the impact levels of incidents"

The analysis is powered by GPT-4o-mini and uses the historical incident data to provide data-backed insights.

### AI Plot Analysis

The application includes an AI-powered plot analysis feature that can analyze visualizations and provide insights. To use this feature:

1. **Setup Requirements:**
   - Ensure you have a valid OpenAI API key
   - Add the API key to your `.env` file as described above
   - Make sure you're running the application in production mode using the start.sh script

2. **Using the Feature:**
   - Generate plots by selecting services and date range
   - Once plots are displayed, find the "AI Plot Analysis" section below the plots
   - Choose either:
     - A single plot to analyze specific visualizations
     - "Analyze All Plots" for a comprehensive summary
   - Click "Analyze Plot" to generate AI insights

3. **Analysis Types:**
   - **Single Plot Analysis**: Provides detailed insights about specific visualizations
   - **All Plots Analysis**: Generates a comprehensive summary of all plots, highlighting key patterns and insights

4. **Troubleshooting:**
   - If you see "Please use production server" message, ensure you're running the server using start.sh
   - Verify your API key is correctly set in the .env file
   - Check the server logs for any API-related errors

## Data Collection and Updates

The application includes scripts to collect and update incident data from various LLM providers. There are two main data collection scripts:

1. **Regular Data Updates** - Collects recent incidents:

   ```bash
   cd server/scripts
   python run_incident_scrapers.py
   ```

   This script:
   - Collects new incidents from OpenAI, Anthropic, Character.AI, and StabilityAI
   - Updates the existing incident database with new data
   - Runs both the StabilityAI.py file and the incident_scraper_oac.py file

2. **Historical Data Collection** - One-time collection of all historical incidents:

   ```bash
   cd server/scripts/data_gen_modules
   python incident_scraper_oac_historical.py
   ```

   This script:
   - Collects all available historical incidents
   - Creates a complete historical database
   - Should be run only once when setting up a new instance

### Troubleshooting Data Collection

If you encounter issues during data collection:

1. **Check the Logs:**
   - View server/logs/incident_scrapers.log for detailed error messages
   - Common issues include network timeouts and parsing errors

2. **Browser Issues:**
   - If you see WebDriver errors, ensure Chrome is properly installed
   - Try running without headless mode for debugging by removing the '--headless=new' option

3. **Data Validation Failures:**
   - Check that the source websites haven't changed their structure
   - Verify network connectivity to all provider status pages

### Learn More

- [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started)
- [React documentation](https://reactjs.org/)
- [Flask documentation](https://flask.palletsprojects.com/)
- [OpenAI API documentation](https://platform.openai.com/docs/api-reference)


---

### Some screenshots of the interface:

<img width="2056" alt="mainpage" src="https://github.com/user-attachments/assets/e31dfd2c-54d6-4a3b-ba23-d1c8fd5fb1bc" />
<img width="2056" alt="datatable" src="https://github.com/user-attachments/assets/57fe0198-43fd-41ae-93f5-53c7fc3788bd" />
<img width="2056" alt="chatbot" src="https://github.com/user-attachments/assets/0d927fd0-bffa-4362-9fd2-9c5f2dc609f8" />
<img width="2056" alt="llmanalysis" src="https://github.com/user-attachments/assets/9ebb9e69-0444-41be-888c-c816642895f6" />

---

Code by [Nishanthi Srinivasan](mailto:n.srinivasan@student.vu.nl), [Bálint László Szarvas](mailto:b.l.szarvas@student.vu.nl) and [Sándor Battaglini-Fischer](mailto:s.battaglini-fischer@student.vu.nl).

Many thanks to [Xiaoyu Chu](mailto:x.chu@vu.nl) and [Prof. Dr. Ir. Alexandru Iosup](mailto:a.iosup@vu.nl) for the support!