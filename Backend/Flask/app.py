# Import all the required packages 
import os
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    repo_name = body['repository']
    token = os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {"Authorization": f'token {token}'}

    # Fetch repository information
    repository_url = GITHUB_URL + "repos/" + repo_name
    repository = requests.get(repository_url, headers=headers).json()

    today = date.today()
    issues_response = []

    # Fetch issues for each month over the past year
    for _ in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        search_query = f"type:issue repo:{repo_name} created:{last_month}..{today}"
        query_url = f"{GITHUB_URL}search/issues?q={search_query}&per_page=100"
        search_issues = requests.get(query_url, headers=headers).json().get("items", [])
        for issue in search_issues:
            issues_response.append({
                "issue_number": issue["number"],
                "created_at": issue["created_at"][:10],
                "closed_at": issue["closed_at"][:10] if issue["closed_at"] else None,
                "state": issue["state"],
                "author": issue["user"]["login"],
            })
        today = last_month

    df = pd.DataFrame(issues_response)

    # Group and format data by month
    created_issues_by_month = df.groupby(df['created_at'].str[:7])['issue_number'].count().to_dict()
    closed_issues_by_month = df[df['closed_at'].notnull()].groupby(df['closed_at'].str[:7])['issue_number'].count().to_dict()

    # Prepare body for LSTM microservice for created issues
    created_body = {
        "issues": issues_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    # Prepare body for LSTM microservice for closed issues
    closed_body = {
        "issues": issues_response,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }
    LSTM_API_URL = "http://localhost:8080/api/forecast"

    # Call LSTM microservice for created issues
    created_response = requests.post(LSTM_API_URL, json=created_body, headers={'content-type': 'application/json'}).json()

    # Call LSTM microservice for closed issues
    closed_response = requests.post(LSTM_API_URL, json=closed_body, headers={'content-type': 'application/json'}).json()

    # Build final response
    json_response = {
        "repository_info": {
            "name": repo_name,
            "stars": repository.get("stargazers_count", 0),
            "forks": repository.get("forks_count", 0),
        },
        "created_issues_by_month": created_issues_by_month,
        "closed_issues_by_month": closed_issues_by_month,
        "forecast_data": {
            "created_issues_forecast_url": created_response["forecast_created_image_url"],
            "closed_issues_forecast_url": closed_response["forecast_closed_image_url"],
            "model_loss_image_url": created_response["model_loss_image_url"],
            "max_created_day_of_week": created_response["max_created_day_of_week"],
            "max_closed_day_of_week": closed_response["max_closed_day_of_week"],
            "max_closed_month_of_year": closed_response["max_closed_month_of_year"],
        }
    }

    return jsonify(json_response)


def fetch_pulls(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).isoformat()
    params = {"state": "all", "per_page": 100, "since": two_years_ago}
    pulls = []
    page = 1
    while True:
        params["page"] = page
        response = requests.get(url, params=params, headers={"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"})
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")
            break
        data = response.json()
        if not data:
            break
        pulls.extend(data)
        page += 1
    return pulls

@app.route('/api/github_pulls', methods=['POST'])
def fetch_and_send_pulls():
    body = request.get_json()
    repository = body.get("repository")

    if not repository:
        return jsonify({"error": "Repository name is required"}), 400

    # Split repository into owner and repo
    try:
        owner, repo = repository.split("/")
    except ValueError:
        return jsonify({"error": "Invalid repository format. Use 'owner/repo'"}), 400

    # Fetch pull request data
    pulls = fetch_pulls(owner, repo)
    if not pulls:
        return jsonify({"error": "No pull request data found"}), 404

    print(f"Fetched {len(pulls)} pull requests for {owner}/{repo}")

    # Initialize forecast data
    forecast_data = {}

    # Send pull request data to LSTM endpoint
    lstm_url = "http://localhost:8080/api/forecast_pulls"
    lstm_response = requests.post(lstm_url, json={"pulls": pulls, "repo": repo})
    
    if lstm_response.status_code == 200:
        forecast_data["lstm_forecast_pulls_image_url"] = lstm_response.json().get("forecast_pulls_image_url")
    else:
        print(f"Error from LSTM endpoint: {lstm_response.text}")
        forecast_data["lstm_forecast_pulls_image_url"] = None

    # Send pull request data to Prophet endpoint
    prophet_url = "http://localhost:8080/api/forecast_pulls_prophet"
    prophet_response = requests.post(prophet_url, json={"pulls": pulls, "repo": repo})
    
    if prophet_response.status_code == 200:
        forecast_data["prophet_forecast_pulls_image_url"] = prophet_response.json().get("forecast_pulls_image_url")
    else:
        print(f"Error from Prophet endpoint: {prophet_response.text}")
        forecast_data["prophet_forecast_pulls_image_url"] = None

    # Send pull request data to Statsmodels endpoint
    statsmodel_url = "http://localhost:8080/api/forecast_pulls_holtwinter"
    statsmodel_response = requests.post(statsmodel_url, json={"pulls": pulls, "repo": repo})
    
    if statsmodel_response.status_code == 200:
        forecast_data["statsmodel_forecast_pulls_image_url"] = statsmodel_response.json().get("forecast_pulls_image_url")
    else:
        print(f"Error from Statsmodels endpoint: {statsmodel_response.text}")
        forecast_data["statsmodel_forecast_pulls_image_url"] = None

    # Construct standardized JSON response
    json_response = {
        "repository_info": {
            "name": repository,
        },
        "forecast_data": forecast_data
    }

    return jsonify(json_response)




# Helper function to fetch commit data
def fetch_commits(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    two_years_ago = (datetime.now() - timedelta(days=2 * 365)).isoformat()
    params = {"since": two_years_ago, "per_page": 100}
    commits = []
    page = 1
    while True:
        params["page"] = page
        response = requests.get(url, params=params, headers={"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"})
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")
            break
        data = response.json()
        if not data:
            break
        for commit in data:
            commits.append({
                "sha": commit["sha"],
                "date": commit["commit"]["author"]["date"]
            })
        page += 1
    return commits

# Flask endpoint for fetching commit data
@app.route('/api/github_commits', methods=['POST'])
def fetch_and_send_commits():
    body = request.get_json()
    repository = body.get("repository")

    if not repository:
        return jsonify({"error": "Repository name is required"}), 400

    # Split repository into owner and repo
    try:
        owner, repo = repository.split("/")
    except ValueError:
        return jsonify({"error": "Invalid repository format. Use 'owner/repo'"}), 400

    # Fetch commit data
    commits = fetch_commits(owner, repo)
    if not commits:
        return jsonify({"error": "No commit data found"}), 404

    print(f"Fetched {len(commits)} commits for {owner}/{repo}")

    # Send commit data to LSTM endpoint
    lstm_url = "http://localhost:8080/api/forecast_commits"
    lstm_response = requests.post(lstm_url, json={"commits": commits, "repo": repo})
    if lstm_response.status_code != 200:
        print(f"Error from LSTM endpoint: {lstm_response.text}")
        return jsonify({"error": "Failed to process commit data at LSTM endpoint"}), 500

    lstm_response_data = lstm_response.json()

    # Send commit data to Prophet endpoint
    prophet_url = "http://localhost:8080/api/forecast_commits_prophet"
    prophet_response = requests.post(prophet_url, json={"commits": commits, "repo": repo})
    if prophet_response.status_code != 200:
        print(f"Error from Prophet endpoint: {prophet_response.text}")
        return jsonify({"error": "Failed to process commit data at Prophet endpoint"}), 500

    prophet_response_data = prophet_response.json()

    # Send commit data to Statsmodel ARIMA endpoint
    statsmodel_url = "http://localhost:8080/api/forecast_commits_arima"
    statsmodel_response = requests.post(statsmodel_url, json={"commits": commits, "repo": repo})
    if statsmodel_response.status_code != 200:
        print(f"Error from Statsmodel ARIMA endpoint: {statsmodel_response.text}")
        return jsonify({"error": "Failed to process commit data at Statsmodel ARIMA endpoint"}), 500

    statsmodel_response_data = statsmodel_response.json()

    # Format the response to include all three models' forecast data
    json_response = {
        "forecast_data": {
            "lstm_forecast_commits_image_url": lstm_response_data.get("forecast_commits_image_url"),
            "prophet_forecast_commits_image_url": prophet_response_data.get("forecast_commits_image_url"),
            "statsmodel_forecast_commits_image_url": statsmodel_response_data.get("forecast_commits_image_url"),
        }
    }

    return jsonify(json_response)


# Helper function to fetch branch data
def fetch_branches(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/branches"
    params = {"per_page": 100}
    branches = []
    page = 1
    while True:
        params["page"] = page
        response = requests.get(url, params=params, headers={"Authorization": f"token {os.environ.get('GITHUB_TOKEN')}"})
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}, {response.text}")
            break
        data = response.json()
        if not data:
            break
        for branch in data:
            branches.append({
                "name": branch["name"],
                "commit_sha": branch["commit"]["sha"]
            })
        page += 1
    return branches

# Flask endpoint for fetching branch data
@app.route('/api/github_branches', methods=['POST'])
def fetch_and_send_branches():
    body = request.get_json()
    repository = body.get("repository")

    if not repository:
        return jsonify({"error": "Repository name is required"}), 400

    # Split repository into owner and repo
    try:
        owner, repo = repository.split("/")
    except ValueError:
        return jsonify({"error": "Invalid repository format. Use 'owner/repo'"}), 400

    # Fetch branch data
    branches = fetch_branches(owner, repo)
    if not branches:
        return jsonify({"error": "No branch data found"}), 404

    print(f"Fetched {len(branches)} branches for {owner}/{repo}")

    # URLs for LSTM, Prophet, and StatsModel ARIMA endpoints
    lstm_url = "http://localhost:8080/api/forecast_branches"
    prophet_url = "http://localhost:8080/api/forecast_branches_prophet"
    arima_url = "http://localhost:8080/api/forecast_branches_arima"

    # Initialize the responses
    lstm_response_data = {}
    prophet_response_data = {}
    arima_response_data = {}

    # Send branch data to LSTM endpoint
    try:
        lstm_response = requests.post(lstm_url, json={"branches": branches, "repo": repo})
        if lstm_response.status_code == 200:
            lstm_response_data = lstm_response.json()
        else:
            print(f"Error from LSTM endpoint: {lstm_response.text}")
    except Exception as e:
        print(f"Error querying LSTM endpoint: {e}")

    # Send branch data to Prophet endpoint
    try:
        prophet_response = requests.post(prophet_url, json={"branches": branches, "repo": repo})
        if prophet_response.status_code == 200:
            prophet_response_data = prophet_response.json()
        else:
            print(f"Error from Prophet endpoint: {prophet_response.text}")
    except Exception as e:
        print(f"Error querying Prophet endpoint: {e}")

    # Send branch data to StatsModel ARIMA endpoint
    try:
        arima_response = requests.post(arima_url, json={"branches": branches, "repo": repo})
        if arima_response.status_code == 200:
            arima_response_data = arima_response.json()
        else:
            print(f"Error from ARIMA endpoint: {arima_response.text}")
    except Exception as e:
        print(f"Error querying ARIMA endpoint: {e}")

    # Format the response to include all three model forecast data
    json_response = {
        "repository_info": {
            "name": repository,
        },
        "forecast_data": {
            "lstm_forecast_branches_image_url": lstm_response_data.get("forecast_branches_image_url"),
            "prophet_forecast_branches_image_url": prophet_response_data.get("forecast_branches_image_url"),
            "arima_forecast_branches_image_url": arima_response_data.get("forecast_branches_image_url"),
        }
    }

    return jsonify(json_response)


@app.route('/api/github_contributors', methods=['POST'])
def fetch_contributors():
    body = request.get_json()
    repository = body.get("repository")

    if not repository:
        return jsonify({"error": "Repository name is required"}), 400

    # Split owner and repo
    try:
        owner, repo = repository.split('/')
    except ValueError:
        print(f"Invalid repository format: {repository}")
        return jsonify({"error": "Invalid repository format. Expected 'owner/repo'."}), 400

    token = os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {"Authorization": f'token {token}'}

    # Fetch contributors from GitHub API for 2 years
    contributors = []
    today = datetime.now()
    two_years_ago = today - timedelta(days=730)

    page = 1
    while True:
        url = f"{GITHUB_URL}repos/{owner}/{repo}/contributors"
        params = {"since": two_years_ago.isoformat(), "page": page, "per_page": 100}
        response = requests.get(url, headers=headers, params=params).json()

        if not response or isinstance(response, dict) and response.get("message"):
            print("Error fetching contributors:", response.get("message"))
            break

        for contrib in response:
            contributors.append({
                "name": contrib.get("login", "unknown"),
                "contributions": contrib.get("contributions", 0)
            })

        if len(response) < 100:  # Stop if less than a full page
            break

        page += 1

    if not contributors:
        return jsonify({"error": "No contributor data found"}), 404

    print(f"Fetched {len(contributors)} contributors for {repository}")

    # Query LSTM endpoint
    lstm_url = "http://localhost:8080/api/forecast_contributors"
    try:
        lstm_response = requests.post(lstm_url, json={"contributors": contributors, "repo": repo})
        lstm_data = lstm_response.json() if lstm_response.status_code == 200 else {"forecast_contributors_image_url": None}
    except Exception as e:
        print("Error querying LSTM endpoint:", e)
        lstm_data = {"forecast_contributors_image_url": None}

    # Query Prophet endpoint
    prophet_url = "http://localhost:8080/api/forecast_contributors_prophet"
    try:
        prophet_response = requests.post(prophet_url, json={"contributors": contributors, "repo": repo})
        prophet_data = prophet_response.json() if prophet_response.status_code == 200 else {"forecast_contributors_image_url": None}
    except Exception as e:
        print("Error querying Prophet endpoint:", e)
        prophet_data = {"forecast_contributors_image_url": None}

    # Query StatsModel endpoint
    statsmodel_url = "http://localhost:8080/api/forecast_contributors_statsmodel"
    try:
        statsmodel_response = requests.post(statsmodel_url, json={"contributors": contributors, "repo": repo})
        statsmodel_data = statsmodel_response.json() if statsmodel_response.status_code == 200 else {"forecast_contributors_image_url": None}
    except Exception as e:
        print("Error querying StatsModel endpoint:", e)
        statsmodel_data = {"forecast_contributors_image_url": None}

    # Return a consistent response format
    json_response = {
        "forecast_data": {
            "lstm_forecast_contributors_image_url": lstm_data.get("forecast_contributors_image_url"),
            "prophet_forecast_contributors_image_url": prophet_data.get("forecast_contributors_image_url"),
            "statsmodel_forecast_contributors_image_url": statsmodel_data.get("forecast_contributors_image_url"),
        }
    }

    return jsonify(json_response)


from datetime import datetime, timedelta
import os
import requests
from flask import Flask, request, jsonify

def fetch_releases(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    headers = {"Authorization": f"token {os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')}"}
    releases = []
    page = 1

    while True:
        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers=headers, params=params)

        # Check for request errors
        if response.status_code != 200:
            print(f"Failed to fetch releases: {response.status_code}, {response.text}")
            break

        data = response.json()
        if not data:
            break

        releases.extend(data)
        page += 1

    # Filter releases from the last 2 years
    two_years_ago = datetime.now() - timedelta(days=730)
    filtered_releases = [
        {
            "name": release.get("name", "unknown"),
            "published_at": release.get("published_at", ""),
        }
        for release in releases if release.get("published_at")
        and datetime.strptime(release["published_at"], "%Y-%m-%dT%H:%M:%SZ") >= two_years_ago
    ]

    print(f"Fetched {len(filtered_releases)} releases from the last 2 years.")
    return filtered_releases

@app.route('/api/github_releases', methods=['POST'])
def fetch_releases_endpoint():
    body = request.get_json()
    repository = body.get("repository")

    if not repository:
        return jsonify({"error": "Repository name is required"}), 400

    # Split owner and repo
    try:
        owner, repo = repository.split('/')
    except ValueError:
        print(f"Invalid repository format: {repository}")
        return jsonify({"error": "Invalid repository format. Expected 'owner/repo'."}), 400

    # Fetch releases for the past 2 years
    releases = fetch_releases(owner, repo)
    if not releases:
        return jsonify({"error": "No release data found."}), 404

    print(f"Fetched {len(releases)} releases for {owner}/{repo}")

    # URLs for forecasting endpoints
    lstm_url = "http://localhost:8080/api/forecast_releases"
    prophet_url = "http://localhost:8080/api/forecast_releases_prophet"
    statsmodel_url = "http://localhost:8080/api/forecast_releases_statsmodel"

    # Initialize response data containers
    lstm_data = {}
    prophet_data = {}
    statsmodel_data = {}

    # Query LSTM endpoint
    try:
        lstm_response = requests.post(lstm_url, json={"releases": releases, "repo": repo})
        if lstm_response.status_code == 200:
            lstm_data = lstm_response.json()
        else:
            print(f"Error from LSTM endpoint: {lstm_response.text}")
    except Exception as e:
        print(f"Error querying LSTM endpoint: {e}")

    # Query Prophet endpoint
    try:
        prophet_response = requests.post(prophet_url, json={"releases": releases, "repo": repo})
        if prophet_response.status_code == 200:
            prophet_data = prophet_response.json()
        else:
            print(f"Error from Prophet endpoint: {prophet_response.text}")
    except Exception as e:
        print(f"Error querying Prophet endpoint: {e}")

    # Query Statsmodel endpoint
    try:
        statsmodel_response = requests.post(statsmodel_url, json={"releases": releases, "repo": repo})
        if statsmodel_response.status_code == 200:
            statsmodel_data = statsmodel_response.json()
        else:
            print(f"Error from Statsmodel endpoint: {statsmodel_response.text}")
    except Exception as e:
        print(f"Error querying Statsmodel endpoint: {e}")

    # Construct the final response
    json_response = {
        "forecast_data": {
            "lstm_forecast_releases_image_url": lstm_data.get("forecast_releases_image_url"),
            "prophet_forecast_releases_image_url": prophet_data.get("forecast_releases_image_url"),
            "statsmodel_forecast_releases_image_url": statsmodel_data.get("forecast_releases_image_url"),
        }
    }

    return jsonify(json_response)




###PROPHET

@app.route('/api/github_prophet', methods=['POST'])
def github_prophet():
    body = request.get_json()
    repo_name = body['repository']
    token = os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {"Authorization": f'token {token}'}

    # Fetch repository information
    repository_url = GITHUB_URL + "repos/" + repo_name
    repository = requests.get(repository_url, headers=headers).json()

    today = date.today()
    issues_response = []

    # Fetch issues for each month over the past year
    for _ in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        search_query = f"type:issue repo:{repo_name} created:{last_month}..{today}"
        query_url = f"{GITHUB_URL}search/issues?q={search_query}&per_page=100"
        search_issues = requests.get(query_url, headers=headers).json().get("items", [])
        for issue in search_issues:
            issues_response.append({
                "issue_number": issue["number"],
                "created_at": issue["created_at"][:10],
                "closed_at": issue["closed_at"][:10] if issue["closed_at"] else None,
                "state": issue["state"],
                "author": issue["user"]["login"],
            })
        today = last_month

    df = pd.DataFrame(issues_response)

    # Group and format data by month
    created_issues_by_month = df.groupby(df['created_at'].str[:7])['issue_number'].count().to_dict()
    closed_issues_by_month = df[df['closed_at'].notnull()].groupby(df['closed_at'].str[:7])['issue_number'].count().to_dict()

    # Prepare body for Prophet microservice for created issues
    created_body = {
        "issues": issues_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    # Prepare body for Prophet microservice for closed issues
    closed_body = {
        "issues": issues_response,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }
    PROPHET_API_URL = "http://localhost:8080/api/forecast_prophet"

    # Call Prophet microservice for created issues
    created_response = requests.post(PROPHET_API_URL, json=created_body, headers={'content-type': 'application/json'}).json()

    # Call Prophet microservice for closed issues
    closed_response = requests.post(PROPHET_API_URL, json=closed_body, headers={'content-type': 'application/json'}).json()

    # Build final response
    json_response = {
        "repository_info": {
            "name": repo_name,
            "stars": repository.get("stargazers_count", 0),
            "forks": repository.get("forks_count", 0),
        },
        "created_issues_by_month": created_issues_by_month,
        "closed_issues_by_month": closed_issues_by_month,
        "forecast_data": {
            "created_issues_forecast_url": created_response["forecast_created_image_url"],
            "closed_issues_forecast_url": closed_response["forecast_closed_image_url"],
            "max_created_day_of_week": created_response["max_created_day_of_week"],
            "max_closed_day_of_week": closed_response["max_closed_day_of_week"],
            "max_closed_month_of_year": closed_response["max_closed_month_of_year"],
        }
    }

    return jsonify(json_response)


###Statsmodel
@app.route('/api/github_statsmodel', methods=['POST'])
def github_statsmodel():
    body = request.get_json()
    repo_name = body['repository']
    token = os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {"Authorization": f'token {token}'}

    # Fetch repository information
    repository_url = GITHUB_URL + "repos/" + repo_name
    repository = requests.get(repository_url, headers=headers).json()

    today = date.today()
    issues_response = []

    # Fetch issues for each month over the past year
    for _ in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        search_query = f"type:issue repo:{repo_name} created:{last_month}..{today}"
        query_url = f"{GITHUB_URL}search/issues?q={search_query}&per_page=100"
        search_issues = requests.get(query_url, headers=headers).json().get("items", [])
        for issue in search_issues:
            issues_response.append({
                "issue_number": issue["number"],
                "created_at": issue["created_at"][:10],
                "closed_at": issue["closed_at"][:10] if issue["closed_at"] else None,
                "state": issue["state"],
                "author": issue["user"]["login"],
            })
        today = last_month

    df = pd.DataFrame(issues_response)

    # Group and format data by month
    created_issues_by_month = df.groupby(df['created_at'].str[:7])['issue_number'].count().to_dict()
    closed_issues_by_month = df[df['closed_at'].notnull()].groupby(df['closed_at'].str[:7])['issue_number'].count().to_dict()

    # Prepare body for Statsmodels microservice for created issues
    created_body = {
        "issues": issues_response,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    # Prepare body for Statsmodels microservice for closed issues
    closed_body = {
        "issues": issues_response,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }
    STATSMODEL_API_URL = "http://localhost:8080/api/forecast_statsmodels"

    # Call Statsmodels microservice for created issues
    created_response = requests.post(STATSMODEL_API_URL, json=created_body, headers={'content-type': 'application/json'}).json()

    # Call Statsmodels microservice for closed issues
    closed_response = requests.post(STATSMODEL_API_URL, json=closed_body, headers={'content-type': 'application/json'}).json()

    # Build final response
    json_response = {
        "repository_info": {
            "name": repo_name,
            "stars": repository.get("stargazers_count", 0),
            "forks": repository.get("forks_count", 0),
        },
        "created_issues_by_month": created_issues_by_month,
        "closed_issues_by_month": closed_issues_by_month,
        "forecast_data": {
            "created_issues_forecast_url": created_response.get("forecast_created_image_url"),
            "closed_issues_forecast_url": closed_response.get("forecast_closed_image_url"),
            "max_created_day_of_week": created_response.get("max_created_day_of_week"),
            "max_closed_day_of_week": closed_response.get("max_closed_day_of_week"),
            "max_closed_month_of_year": closed_response.get("max_closed_month_of_year"),
        }
    }

    return jsonify(json_response)


# Run Flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
