import * as React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import AppBar from "@mui/material/AppBar";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import { ListItemButton } from "@mui/material";
import BarCharts from "./BarCharts";
import Loader from "./Loader";

const drawerWidth = 240;

const repositories = [
  { key: "langchain-ai/langchain", value: "Langchain" },
  { key: "langchain-ai/langgraph", value: "Langgraph" },
  { key: "microsoft/autogen", value: "Autogen" },
  { key: "openai/openai-cookbook", value: "OpenAI Cookbook" },
  { key: "elastic/elasticsearch", value: "Elasticsearch" },
  { key: "milvus-io/pymilvus", value: "Pymilvus" },
];

export default function Home() {
  const [loading, setLoading] = useState({
    github: true,
    prophet: true,
    statsmodel: true,
    forecast: {
      pulls: false,
      commits: false,
      branches: false,
      contributors: false,
      releases: false,
    },
  });
  const [repository, setRepository] = useState({
    key: "langchain-ai/langchain",
    value: "Langchain",
  });
  const [githubRepoData, setGithubData] = useState({});
  const [forecastData, setForecastData] = useState({});
  const [prophetRepoData, setProphetRepoData] = useState({});
  const [StatRepoData, setStatRepoData] = useState({});


  const eventHandler = (repo) => {
    setRepository(repo);
  };

  React.useEffect(() => {
    fetchGitHubData();
    fetchProphetData();
    fetchStatData();
    fetchForecast("pulls", "/api/github_pulls");
    fetchForecast("commits", "/api/github_commits");
    fetchForecast("branches", "/api/github_branches");
    fetchForecast("contributors", "/api/github_contributors");
    fetchForecast("releases", "/api/github_releases");
  }, [repository]);

  const fetchGitHubData = () => {
    setLoading((prev) => ({ ...prev, github: true }));
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ repository: repository.key }),
    };

    fetch("/api/github", requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          console.log("GitHub API Result:", result);
          setLoading((prev) => ({ ...prev, github: false }));
          setGithubData(result);
        },
        (error) => {
          console.error("GitHub API Error:", error);
          setLoading((prev) => ({ ...prev, github: false }));
          setGithubData({});
        }
      );
  };

  const fetchProphetData = () => {
    setLoading((prev) => ({ ...prev, prophet: true }));
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ repository: repository.key }),
    };
  
    fetch("/api/github_prophet", requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          console.log("Prophet API Result:", result);
          setLoading((prev) => ({ ...prev, prophet: false }));
          setProphetRepoData(result);
        },
        (error) => {
          console.error("Prophet API Error:", error);
          setLoading((prev) => ({ ...prev, prophet: false }));
          setProphetRepoData({});
        }
      );
  };

  const fetchStatData = () => {
    setLoading((prev) => ({ ...prev, statsmodel: true }));
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ repository: repository.key }),
    };
  
    fetch("/api/github_statsmodel", requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          console.log("Statmodel API Result:", result);
          setLoading((prev) => ({ ...prev, statsmodel: false }));
          setStatRepoData(result);
        },
        (error) => {
          console.error("Prophet API Error:", error);
          setLoading((prev) => ({ ...prev, statsmodel: false }));
          setStatRepoData({});
        }
      );
  };
  

  const fetchForecast = (key, endpoint) => {
    // Set loading to true for the specific key
    setLoading((prevState) => ({
      ...prevState,
      forecast: { ...prevState.forecast, [key]: true },
    }));
  
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ repository: repository.key }),
    };
  
    fetch(endpoint, requestOptions)
      .then((res) => res.json())
      .then(
        (result) => {
          console.log(`${key} Forecast Result:`, result);
  
          // Update forecast data
          setForecastData((prevState) => ({
            ...prevState,
            [key]: {
              lstm: result.forecast_data[`lstm_forecast_${key}_image_url`] || "",
              prophet: result.forecast_data[`prophet_forecast_${key}_image_url`] || "",
              statsmodel: result.forecast_data[`statsmodel_forecast_${key}_image_url`] || "",
            },
          }));
  
          // Set loading to false for the specific key
          setLoading((prevState) => ({
            ...prevState,
            forecast: { ...prevState.forecast, [key]: false },
          }));
        },
        (error) => {
          console.error(`${key} Forecast API Error:`, error);
  
          // Set loading to false for the specific key in case of error
          setLoading((prevState) => ({
            ...prevState,
            forecast: { ...prevState.forecast, [key]: false },
          }));
        }
      );
  };
  
  

  if (Object.values(loading).some((value) => value === true || (typeof value === "object" && Object.values(value).some((subValue) => subValue === true)))) {
    return <Loader />;
  }
  

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Timeseries Forecasting
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: "border-box" },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: "auto" }}>
          <List>
            {repositories.map((repo) => (
              <ListItem
                button
                key={repo.key}
                onClick={() => eventHandler(repo)}
              >
                <ListItemButton selected={repo.value === repository.value}>
                  <ListItemText primary={repo.value} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <BarCharts
          title={`Monthly Created Issues for ${repository.value} in the last 1 year`}
          data={
            githubRepoData?.created_issues_by_month
              ? Object.entries(githubRepoData.created_issues_by_month)
              : []
          }
        />
        <BarCharts
          title={`Monthly Closed Issues for ${repository.value} in the last 1 year`}
          data={
            githubRepoData?.closed_issues_by_month
              ? Object.entries(githubRepoData.closed_issues_by_month)
              : []
          }
        />
        <Divider sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }} />
        <div>
          <Typography variant="h5" gutterBottom>
            Forecast Insights for {repository.value}
          </Typography>
          <Typography variant="body1">
            <strong>Day with Maximum Created Issues:</strong>{" "}
            {githubRepoData?.forecast_data?.max_created_day_of_week}
          </Typography>
          <Typography variant="body1">
            <strong>Day with Maximum Closed Issues:</strong>{" "}
            {githubRepoData?.forecast_data?.max_closed_day_of_week}
          </Typography>
          <Typography variant="body1">
            <strong>Month with Maximum Closed Issues:</strong>{" "}
            {githubRepoData?.forecast_data?.max_closed_month_of_year}
          </Typography>
          <div>
            <Typography component="h4">Forecasted Created Issues</Typography>
            <img
              src={githubRepoData?.forecast_data?.created_issues_forecast_url}
              alt="Forecasted Created Issues"
              loading="lazy"
            />
          </div>
          <Divider sx={{ my: 3 }} />
          <div>
            <Typography component="h4">Forecasted Closed Issues</Typography>
            <img
              src={githubRepoData?.forecast_data?.closed_issues_forecast_url}
              alt="Forecasted Closed Issues"
              loading="lazy"
            />
          </div>
          <Divider sx={{ my: 3 }} />
          <Divider sx={{ my: 3 }} />
          <div>
            <Typography variant="h5" gutterBottom>
              Prophet Forecast Insights for {repository.value}
            </Typography>
            <Typography variant="body1">
              <strong>Day with Maximum Created Issues:</strong>{" "}
              {prophetRepoData?.forecast_data?.max_created_day_of_week}
            </Typography>
            <Typography variant="body1">
              <strong>Day with Maximum Closed Issues:</strong>{" "}
              {prophetRepoData?.forecast_data?.max_closed_day_of_week}
            </Typography>
            <Typography variant="body1">
              <strong>Month with Maximum Closed Issues:</strong>{" "}
              {prophetRepoData?.forecast_data?.max_closed_month_of_year}
            </Typography>
            <div>
              <Typography component="h4">Prophet Forecasted Created Issues</Typography>
              <img
                src={prophetRepoData?.forecast_data?.created_issues_forecast_url}
                alt="Prophet Forecasted Created Issues"
                loading="lazy"
              />
            </div>
            <Divider sx={{ my: 3 }} />
            <div>
              <Typography component="h4">Prophet Forecasted Closed Issues</Typography>
              <img
                src={prophetRepoData?.forecast_data?.closed_issues_forecast_url}
                alt="Prophet Forecasted Closed Issues"
                loading="lazy"
              />
            </div>
          </div>

          <Divider sx={{ my: 3 }} />
          <div>
            <Typography variant="h5" gutterBottom>
              Statmodel Forecast Insights for {repository.value}
            </Typography>
            <Typography variant="body1">
              <strong>Day with Maximum Created Issues:</strong>{" "}
              {StatRepoData?.forecast_data?.max_created_day_of_week}
            </Typography>
            <Typography variant="body1">
              <strong>Day with Maximum Closed Issues:</strong>{" "}
              {StatRepoData?.forecast_data?.max_closed_day_of_week}
            </Typography>
            <Typography variant="body1">
              <strong>Month with Maximum Closed Issues:</strong>{" "}
              {StatRepoData?.forecast_data?.max_closed_month_of_year}
            </Typography>
            <div>
              <Typography component="h4">Statmodel Forecasted Created Issues</Typography>
              <img
                src={StatRepoData?.forecast_data?.created_issues_forecast_url}
                alt="Statmodel Forecasted Created Issues"
                loading="lazy"
              />
            </div>
            <Divider sx={{ my: 3 }} />
            <div>
              <Typography component="h4">Statmodel Forecasted Closed Issues</Typography>
              <img
                src={StatRepoData?.forecast_data?.closed_issues_forecast_url}
                alt="Statmodel Forecasted Closed Issues"
                loading="lazy"
              />
            </div>
          </div>

          <div>
            <Typography variant="h6">Commits Forecast</Typography>
            {loading.forecast.commits ? (
              <Loader /> // Show loader while loading
            ) : (
              <div>
                <Typography variant="body2">LSTM Model:</Typography>
                <img src={forecastData.commits?.lstm} alt="LSTM Commits Forecast" />
                <Typography variant="body2">Prophet Model:</Typography>
                <img src={forecastData.commits?.prophet} alt="Prophet Commits Forecast" />
                <Typography variant="body2">Statsmodel:</Typography>
                <img src={forecastData.commits?.statsmodel} alt="Statsmodel Commits Forecast" />
              </div>
            )}
          </div>

          <div>
            <Typography variant="h6">Branches Forecast</Typography>
            {loading.forecast.branches ? (
              <Loader /> // Show loader while loading
            ) : (
              <div>
                <Typography variant="body2">LSTM Model:</Typography>
                <img src={forecastData.branches?.lstm} alt="LSTM Branches Forecast" />
                <Typography variant="body2">Prophet Model:</Typography>
                <img src={forecastData.branches?.prophet} alt="Prophet Branches Forecast" />
                <Typography variant="body2">Statsmodel:</Typography>
                <img src={forecastData.branches?.statsmodel} alt="Statsmodel Branches Forecast" />
              </div>
            )}
          </div>

          <div>
            <Typography variant="h6">Contributors Forecast</Typography>
            {loading.forecast.contributors ? (
              <Loader /> // Show loader while loading
            ) : (
              <div>
                <Typography variant="body2">LSTM Model:</Typography>
                <img src={forecastData.contributors?.lstm} alt="LSTM Contributors Forecast" />
                <Typography variant="body2">Prophet Model:</Typography>
                <img src={forecastData.contributors?.prophet} alt="Prophet Contributors Forecast" />
                <Typography variant="body2">Statsmodel:</Typography>
                <img src={forecastData.contributors?.statsmodel} alt="Statsmodel Contributors Forecast" />
              </div>
            )}
          </div>

          <div>
            <Typography variant="h6">Releases Forecast</Typography>
            {loading.forecast.releases ? (
              <Loader /> // Show loader while loading
            ) : (
              <div>
                <Typography variant="body2">LSTM Model:</Typography>
                <img src={forecastData.releases?.lstm} alt="LSTM Releases Forecast" />
                <Typography variant="body2">Prophet Model:</Typography>
                <img src={forecastData.releases?.prophet} alt="Prophet Releases Forecast" />
                <Typography variant="body2">Statsmodel:</Typography>
                <img src={forecastData.releases?.statsmodel} alt="Statsmodel Releases Forecast" />
              </div>
            )}
          </div>

          <div>
            <Typography variant="h6">Pull Requests Forecast</Typography>
            {loading.forecast.pulls ? (
              <Loader /> // Show loader while loading
            ) : (
              <div>
                <Typography variant="body2">LSTM Model:</Typography>
                <img src={forecastData.pulls?.lstm} alt="LSTM Pulls Forecast" />
                <Typography variant="body2">Prophet Model:</Typography>
                <img src={forecastData.pulls?.prophet} alt="Prophet Pulls Forecast" />
                <Typography variant="body2">Statsmodel:</Typography>
                <img src={forecastData.pulls?.statsmodel} alt="Statsmodel Pulls Forecast" />
              </div>
            )}
          </div>










          
        </div>
        
      </Box>
    </Box>
  );
}
