# GitTimeCast 📊🔮
Multi-Model Time-Series Forecasting for GitHub Repositories


# 📌 Overview
GitTimeCast is an advanced time-series forecasting platform that predicts trends in GitHub repositories using LSTM (TensorFlow/Keras), Prophet, and StatsModels. It fetches data from GitHub APIs and applies forecasting models to pull requests, commits, branches, contributors, and releases.

🚀 The entire backend and frontend are packaged as Docker images and deployed on Google Cloud Run.

☁️ All model predictions and forecast images are stored in Google Cloud Storage.

# ✨ Features
✅ Multi-Model Forecasting (LSTM, Prophet, StatsModels)

✅ Comprehensive GitHub Repository Analysis

✅ Time-Series Predictions for Key Metrics

✅ Interactive Dashboard with Data Visualizations

✅ Google Cloud Storage for Image Hosting

✅ Dockerized Backend & Frontend Deployment

# 🛠️ Tech Stack
### Frontend (React + MUI)
* React.js (UI Framework)
* Material UI (MUI) (Styled Components)
* Recharts (Data Visualization)
* Fetch API (Data Fetching)
### Backend (Python + Flask + ML Models)
* Flask (REST API)
* TensorFlow/Keras (LSTM Forecasting)
* Facebook Prophet (Prophet Forecasting)
* StatsModels (SARIMA/ARIMA Forecasting)
* Pandas, NumPy, Matplotlib (Data Processing & Visualization)
### Services & Deployment
* GitHub API (Data Extraction)
* Google Cloud Storage (Image Hosting)
* Docker (Containerization)
* Google Cloud Run (Scalable Hosting for Backend & Frontend)



# 📡 API Endpoints
### Endpoint	Description
* /api/github: Fetch repository data
* /api/github_pulls: Predict pull requests trend
* /api/github_commits: Predict commits trend
* /api/github_branches: Predict branch growth
* /api/github_contributors: Predict contributor activity
* /api/github_releases: Predict repository releases

# 📊 How It Works?
* User selects a GitHub repository 🏗️
* GitTimeCast fetches real-time data via GitHub API 📡
* LSTM, Prophet, and StatsModels process & forecast time-series trends 📈
* Results are visualized in an interactive React dashboard 🔍

# Forecast images are stored on Google Cloud Storage ☁️
### 🎯 Model Comparison & Insights
* LSTM: Performs well on large datasets but requires extensive training time.
* Prophet: Best for handling seasonality and trend components.
* StatsModels (ARIMA/SARIMA): Works best for small, structured datasets.

✅ Based on our experiments, Prophet showed better forecasting results for pull requests and releases.



### Side NavBar to choose the Github repository
![SS1](https://github.com/user-attachments/assets/bfe0b902-3fe5-41ed-bab2-90d19de6473b)

### Monthly Issues Created Info
![SS2](https://github.com/user-attachments/assets/e6e79185-913e-4fd9-9a65-c16860e32a28)

### LSTM Forecast
![SS3](https://github.com/user-attachments/assets/d6db5a3b-3c4c-40e0-b0b7-8ea1f5aee811)

### Prophet Forecast
![SS4](https://github.com/user-attachments/assets/714ca9ed-ee12-4ae4-8e91-d67aa360462d)



