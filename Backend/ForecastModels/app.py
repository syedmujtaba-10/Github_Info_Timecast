# Import required packages
from flask import Flask, jsonify, request
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
from datetime import datetime
from prophet import Prophet


# TensorFlow (Keras & LSTM) related packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Google Cloud Storage package
from google.cloud import storage
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Google Cloud Storage client
client = storage.Client()

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    issue_type = body["type"]
    repo_name = body["repo"]

    # Prepare DataFrame
    data_frame = pd.DataFrame(issues)
    df = data_frame[[issue_type, 'issue_number']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Extract day of week and month
    df['day_of_week'] = df['ds'].dt.day_name()
    df['month_of_year'] = df['ds'].dt.month_name()

    # Insights
    max_created_day = df[df['ds'].notnull()]['day_of_week'].value_counts().idxmax()
    max_closed_day = df[df['ds'].notnull()]['day_of_week'].value_counts().idxmax()
    max_closed_month = df['month_of_year'].value_counts().idxmax()

    # Prepare time series data
    df = df.groupby('ds', as_index=False).sum()
    df['y'] = df['y'].astype('float32')
    data = df.set_index('ds')['y']

    # Resampling to daily frequency and filling missing values
    data = data.asfreq('D').fillna(0)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    # Split into train and test
    look_back = 30
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Helper function to create dataset
    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape inputs
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # Define LSTM model
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1,
        shuffle=False
    )

    # Predict values
    predictions = model.predict(X_test)

    # Generate plots
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    MODEL_LOSS_IMAGE_NAME = f"model_loss_{repo_name}.png"
    FORECAST_CREATED_IMAGE_NAME = f"forecast_created_{repo_name}.png"
    FORECAST_CLOSED_IMAGE_NAME = f"forecast_closed_{repo_name}.png"

    # Model Loss Plot
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Created Issues Forecast Plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(Y_train)), scaler.inverse_transform(Y_train.reshape(-1, 1)), label='Train Data')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), scaler.inverse_transform(Y_test.reshape(-1, 1)), label='Actual')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), scaler.inverse_transform(predictions), label='Predicted')
    plt.title('Created Issues Forecast')
    plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME)

    # Closed Issues Forecast Plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(Y_train)), scaler.inverse_transform(Y_train.reshape(-1, 1)), label='Train Data')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), scaler.inverse_transform(Y_test.reshape(-1, 1)), label='Actual')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), scaler.inverse_transform(predictions), label='Predicted')
    plt.title('Closed Issues Forecast')
    plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME)

    # Upload images to Google Cloud Storage
    bucket = client.get_bucket(BUCKET_NAME)
    for img_name in [MODEL_LOSS_IMAGE_NAME, FORECAST_CREATED_IMAGE_NAME, FORECAST_CLOSED_IMAGE_NAME]:
        blob = bucket.blob(img_name)
        blob.upload_from_filename(LOCAL_IMAGE_PATH + img_name)

    # Return response
    json_response = {
        "model_loss_image_url": f"{BASE_IMAGE_PATH}{MODEL_LOSS_IMAGE_NAME}",
        "forecast_created_image_url": f"{BASE_IMAGE_PATH}{FORECAST_CREATED_IMAGE_NAME}",
        "forecast_closed_image_url": f"{BASE_IMAGE_PATH}{FORECAST_CLOSED_IMAGE_NAME}",
        "max_created_day_of_week": max_created_day,
        "max_closed_day_of_week": max_closed_day,
        "max_closed_month_of_year": max_closed_month,
    }
    return jsonify(json_response)


def preprocess_pulls(pulls):
    df = pd.DataFrame(pulls)
    if "created_at" not in df.columns:
        print("No 'created_at' field found in pull request data")
        return pd.DataFrame()  # Return empty DataFrame if invalid

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["month"] = df["created_at"].dt.month  # Extract month
    return df.groupby("month").size().reset_index(name="count")

# Flask endpoint for forecasting pull requests
@app.route('/api/forecast_pulls', methods=['POST'])
def forecast_pulls():
    body = request.get_json()
    pulls = body.get("pulls", [])
    repo_name = body.get("repo", "unknown_repo")

    if not pulls:
        return jsonify({"error": "No pull request data provided"}), 400

    # Preprocess the pull requests
    grouped_df = preprocess_pulls(pulls)
    if grouped_df.empty:
        return jsonify({"error": "No valid pull request data available"}), 400

    print(f"Processed Pull Request Data for {repo_name}:\n", grouped_df)

    # Prepare the data for LSTM
    X = np.array(grouped_df["month"]).reshape(-1, 1)  # Months (1-12)
    y = grouped_df["count"].values  # Pull request counts

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))  # Scale months to [0, 1]
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Scale pull counts to [0, 1]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape X for LSTM: [samples, time_steps, features]
    X_lstm = np.array([X_scaled[i:i + 1] for i in range(len(X_scaled))])
    y_lstm = y_scaled

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential([
        Input(shape=(1, 1)),  # Define input shape explicitly
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

    # Make predictions for the entire dataset
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Save the forecast plot
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_pulls_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["month"], predictions.flatten(), label="Forecasted Pull Requests", marker="x", linestyle="--", color="green")
    plt.title(f"Forecasted Pull Requests for {repo_name}")
    plt.xlabel("Month (1=January, 12=December)")
    plt.ylabel("Forecasted Number of Pull Requests")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_pulls_image_url": forecast_image_url}), 200


@app.route('/api/forecast_commits', methods=['POST'])
def forecast_commits():
    body = request.get_json()
    commits = body.get("commits", [])
    repo_name = body.get("repo", "unknown_repo")

    if not commits:
        return jsonify({"error": "No commit data provided"}), 400

    # Preprocess commit data
    df = pd.DataFrame(commits)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")  # Group by month
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()  # Convert to timestamp for plotting

    if grouped_df.empty:
        return jsonify({"error": "No valid commit data available"}), 400

    print(f"Processed Commit Data for {repo_name}:\n", grouped_df)

    # Prepare the data for LSTM
    X = np.array(grouped_df.index).reshape(-1, 1)  # Month indices
    y = grouped_df["count"].values  # Commit counts

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))  # Scale months to [0, 1]
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Scale commit counts to [0, 1]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape X for LSTM: [samples, time_steps, features]
    X_lstm = np.array([X_scaled[i:i + 1] for i in range(len(X_scaled))])
    y_lstm = y_scaled

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential([
        Input(shape=(1, 1)),  # Define input shape explicitly
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

    # Make predictions for the entire dataset
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_commits_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["month"], predictions.flatten(), label="Forecasted Commits", marker="x", linestyle="--", color="blue")
    plt.title(f"Forecasted Commits for {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Forecasted Number of Commits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_commits_image_url": forecast_image_url}), 200

@app.route('/api/forecast_branches', methods=['POST'])
def forecast_branches():
    from datetime import timedelta

    body = request.get_json()
    branches = body.get("branches", [])
    repo_name = body.get("repo", "unknown_repo")

    if not branches:
        return jsonify({"error": "No branch data provided"}), 400

    # Preprocess branch data
    df = pd.DataFrame(branches)
    
    # Simulate creation dates for branches (assign dates incrementally)
    start_date = datetime.now() - timedelta(days=len(df))  # Start from `len(df)` days ago
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid branch data available"}), 400

    print(f"Processed Branch Data for {repo_name}:\n", grouped_df)

    # Prepare the data for LSTM
    X = np.array(grouped_df.index).reshape(-1, 1)  # Month indices
    y = grouped_df["count"].values  # Branch counts

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))  # Scale months to [0, 1]
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Scale branch counts to [0, 1]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape X for LSTM: [samples, time_steps, features]
    X_lstm = np.array([X_scaled[i:i + 1] for i in range(len(X_scaled))])
    y_lstm = y_scaled

    # Handle small datasets
    if len(X_lstm) < 2:  # Not enough samples for splitting
        print("Not enough data for train-test split. Using all data for training.")
        X_train, y_train = X_lstm, y_lstm
        X_test, y_test = X_lstm, y_lstm  # Use the same data for testing
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential([
        Input(shape=(1, 1)),  # Define input shape explicitly
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

    # Make predictions for the entire dataset
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_branches_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["month"], predictions.flatten(), label="Forecasted Branch Counts", marker="x", linestyle="--", color="purple")
    plt.title(f"Forecasted Branches for {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Branches")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_branches_image_url": forecast_image_url}), 200

@app.route('/api/forecast_contributors', methods=['POST'])
def forecast_contributors():
    body = request.get_json()
    contributors = body.get("contributors", [])
    repo_name = body.get("repo", "unknown_repo")

    if not contributors:
        return jsonify({"error": "No contributor data provided"}), 400

    # Preprocess contributor data
    df = pd.DataFrame(contributors)

    # Simulate timestamps for contributors
    start_date = datetime.now() - timedelta(days=len(df))
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month")["contributions"].sum().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid contributor data available"}), 400

    print(f"Processed Contributor Data for {repo_name}:\n", grouped_df)

    # Prepare data for LSTM
    X = np.array(grouped_df.index).reshape(-1, 1)  # Month indices
    y = grouped_df["count"].values  # Contribution counts

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))  # Scale months to [0, 1]
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Scale contribution counts to [0, 1]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape X for LSTM: [samples, time_steps, features]
    X_lstm = np.array([X_scaled[i:i + 1] for i in range(len(X_scaled))])
    y_lstm = y_scaled

    # Handle small datasets
    if len(X_lstm) < 2:  # Not enough samples for splitting
        print("Not enough data for train-test split. Using all data for training.")
        X_train, y_train = X_lstm, y_lstm
        X_test, y_test = X_lstm, y_lstm  # Use the same data for testing
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential([
        Input(shape=(1, 1)),  # Define input shape explicitly
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

    # Make predictions for the entire dataset
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_contributors_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["month"], predictions.flatten(), label="Forecasted Contributions", marker="x", linestyle="--", color="purple")
    plt.title(f"Forecasted Contributions for {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Contributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)  # Save in the created directory
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_contributors_image_url": forecast_image_url}), 200

@app.route('/api/forecast_releases', methods=['POST'])
def forecast_releases():
    body = request.get_json()
    releases = body.get("releases", [])
    repo_name = body.get("repo", "unknown_repo")

    if not releases:
        return jsonify({"error": "No release data provided"}), 400

    # Preprocess release data
    df = pd.DataFrame(releases)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["month"] = df["published_at"].dt.to_period("M")
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid release data available"}), 400

    print(f"Processed Release Data for {repo_name}:\n", grouped_df)

    # Prepare data for LSTM
    X = np.array(grouped_df.index).reshape(-1, 1)  # Month indices
    y = grouped_df["count"].values  # Release counts

    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))  # Scale months to [0, 1]
    scaler_y = MinMaxScaler(feature_range=(0, 1))  # Scale release counts to [0, 1]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape X for LSTM: [samples, time_steps, features]
    X_lstm = np.array([X_scaled[i:i + 1] for i in range(len(X_scaled))])
    y_lstm = y_scaled

    # Handle small datasets
    if len(X_lstm) < 2:
        print("Not enough data for train-test split. Using all data for training.")
        X_train, y_train = X_lstm, y_lstm
        X_test, y_test = X_lstm, y_lstm
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential([
        Input(shape=(1, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=0)

    # Make predictions
    predictions_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_releases_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df["month"], predictions.flatten(), label="Forecasted Releases", marker="x", linestyle="--", color="purple")
    plt.title(f"Forecasted Releases for {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Releases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_releases_image_url": forecast_image_url}), 200


####Prophet


@app.route('/api/forecast_prophet', methods=['POST'])
def forecast_prophet():
    body = request.get_json()
    issues = body["issues"]
    repo_name = body["repo"]

    # Prepare DataFrame
    data_frame = pd.DataFrame(issues)
    df = data_frame[["created_at", "issue_number"]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # Remove rows with NaN values in `ds` column
    df = df.dropna(subset=["ds"])

    if df.empty:
        return jsonify({"error": "No valid data available for Prophet forecast."}), 400

    # Extract day of week and month
    df["day_of_week"] = df["ds"].dt.day_name()
    df["month_of_year"] = df["ds"].dt.month_name()

    # Insights
    max_created_day = df["day_of_week"].value_counts().idxmax()
    max_closed_day = df["day_of_week"].value_counts().idxmax()
    max_closed_month = df["month_of_year"].value_counts().idxmax()

    # Prepare data for Prophet
    df = df.groupby("ds", as_index=False).sum()
    df["y"] = df["y"].astype("float32")

    # Initialize Prophet model
    model = Prophet()
    model.fit(df[["ds", "y"]])

    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Forecast next 30 days
    forecast = model.predict(future)

    # Define image paths
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path')
    MODEL_PLOT_IMAGE_NAME = f"prophet_model_plot_{repo_name}.png"
    FORECAST_CREATED_IMAGE_NAME = f"prophet_forecast_created_{repo_name}.png"
    FORECAST_CLOSED_IMAGE_NAME = f"prophet_forecast_closed_{repo_name}.png"

    # Plot the forecast
    fig1 = model.plot(forecast)
    fig1.savefig(LOCAL_IMAGE_PATH + MODEL_PLOT_IMAGE_NAME)
    plt.close(fig1)

    # Plot Created Issues Forecast
    plt.figure(figsize=(10, 4))
    plt.plot(df["ds"], df["y"], label="Actual Data")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.title("Created Issues Forecast")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME)
    plt.close()

    # Closed Issues Forecast - Simulate a similar plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["ds"], df["y"], label="Actual Data")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    plt.title("Closed Issues Forecast")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME)
    plt.close()

    # Upload images to Google Cloud Storage (optional)
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    for img_name in [MODEL_PLOT_IMAGE_NAME, FORECAST_CREATED_IMAGE_NAME, FORECAST_CLOSED_IMAGE_NAME]:
        blob = bucket.blob(img_name)
        blob.upload_from_filename(LOCAL_IMAGE_PATH + img_name)

    # Build response
    json_response = {
        "model_plot_image_url": f"{BASE_IMAGE_PATH}{MODEL_PLOT_IMAGE_NAME}",
        "forecast_created_image_url": f"{BASE_IMAGE_PATH}{FORECAST_CREATED_IMAGE_NAME}",
        "forecast_closed_image_url": f"{BASE_IMAGE_PATH}{FORECAST_CLOSED_IMAGE_NAME}",
        "max_created_day_of_week": max_created_day,
        "max_closed_day_of_week": max_closed_day,
        "max_closed_month_of_year": max_closed_month,
    }
    return jsonify(json_response)


@app.route('/api/forecast_pulls_prophet', methods=['POST'])
def forecast_pulls_prophet():
    def preprocess_pulls(pulls):
        df = pd.DataFrame(pulls)
        if "created_at" not in df.columns:
            print("No 'created_at' field found in pull request data")
            return pd.DataFrame()  # Return empty DataFrame if invalid

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"])
        df["month"] = df["created_at"].dt.to_period("M")  # Extract month
        grouped_df = df.groupby("month").size().reset_index(name="count")
        grouped_df["month"] = grouped_df["month"].dt.to_timestamp()
        return grouped_df

    body = request.get_json()
    pulls = body.get("pulls", [])
    repo_name = body.get("repo", "unknown_repo")

    if not pulls:
        return jsonify({"error": "No pull request data provided"}), 400

    # Preprocess the pull requests
    grouped_df = preprocess_pulls(pulls)
    if grouped_df.empty:
        return jsonify({"error": "No valid pull request data available"}), 400

    print(f"Processed Pull Request Data for {repo_name}:\n", grouped_df)

    # Prepare data for Prophet
    df = grouped_df.rename(columns={"month": "ds", "count": "y"})

    # Initialize Prophet model
    model = Prophet()
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=12, freq="M")  # Forecast next 12 months
    forecast = model.predict(future)

    # Define image paths
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    FORECAST_IMAGE_NAME = f"prophet_forecast_pulls_{repo_name}.png"

    # Forecast Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["ds"], df["y"], label="Actual Pull Requests", marker="o", linestyle="-", color="blue")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecasted Pull Requests", marker="x", linestyle="--", color="green")
    plt.title(f"Forecasted Pull Requests for {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Pull Requests")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    plt.close()

    # Upload images to Google Cloud Storage
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)

        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{LOCAL_IMAGE_PATH}{FORECAST_IMAGE_NAME}"

    # Build JSON response
    json_response = {
        "forecast_pulls_image_url": forecast_image_url
    }

    return jsonify(json_response)

@app.route('/api/forecast_commits_prophet', methods=['POST'])
def forecast_commits_prophet():
    body = request.get_json()
    commits = body.get("commits", [])
    repo_name = body.get("repo", "unknown_repo")

    if not commits:
        return jsonify({"error": "No commit data provided"}), 400

    # Preprocess commit data
    df = pd.DataFrame(commits)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")  # Group by month
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()  # Convert to timestamp for Prophet

    if grouped_df.empty:
        return jsonify({"error": "No valid commit data available"}), 400

    print(f"Processed Commit Data for {repo_name}:\n", grouped_df)

    # Prepare data for Prophet
    prophet_df = grouped_df.rename(columns={"month": "ds", "count": "y"})

    # Initialize and train Prophet model
    from prophet import Prophet
    model = Prophet()
    model.fit(prophet_df)

    # Generate future dates for prediction
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_commits_prophet_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    # Plot forecast
    plt.figure(figsize=(10, 6))
    model.plot(forecast)
    plt.title(f"Forecasted Commits for {repo_name} (Prophet)")
    plt.xlabel("Month")
    plt.ylabel("Forecasted Number of Commits")
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_commits_image_url": forecast_image_url}), 200



@app.route('/api/forecast_branches_prophet', methods=['POST'])
def forecast_branches_prophet():
    from datetime import timedelta

    body = request.get_json()
    branches = body.get("branches", [])
    repo_name = body.get("repo", "unknown_repo")

    if not branches:
        return jsonify({"error": "No branch data provided"}), 400

    # Preprocess branch data
    df = pd.DataFrame(branches)

    # Simulate creation dates for branches (assign dates incrementally)
    start_date = datetime.now() - timedelta(days=len(df))  # Start from `len(df)` days ago
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid branch data available"}), 400

    print(f"Processed Branch Data for {repo_name}:\n", grouped_df)

    # Prepare data for Prophet
    prophet_df = grouped_df.rename(columns={"month": "ds", "count": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    from prophet import Prophet
    model = Prophet()
    model.fit(prophet_df)

    # Make future predictions
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_branches_prophet_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    # Plot forecast results
    fig = model.plot(forecast)
    plt.title(f"Prophet Forecast for Branches - {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Branches")
    fig.savefig(full_local_path)
    plt.close(fig)

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_branches_image_url": forecast_image_url}), 200




@app.route('/api/forecast_contributors_prophet', methods=['POST'])
def forecast_contributors_prophet():
    body = request.get_json()
    contributors = body.get("contributors", [])
    repo_name = body.get("repo", "unknown_repo")

    if not contributors:
        return jsonify({"error": "No contributor data provided"}), 400

    # Preprocess contributor data
    df = pd.DataFrame(contributors)

    # Simulate timestamps for contributors
    start_date = datetime.now() - timedelta(days=len(df))
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month")["contributions"].sum().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid contributor data available"}), 400

    print(f"Processed Contributor Data for {repo_name}:\n", grouped_df)

    # Prepare data for Prophet
    prophet_df = grouped_df.rename(columns={"month": "ds", "count": "y"})

    # Instantiate and fit the Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Generate future dates
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Plot the forecast
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_contributors_prophet_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    fig = model.plot(forecast)
    plt.title(f"Prophet Forecasted Contributions for {repo_name}")
    plt.xlabel("Date")
    plt.ylabel("Number of Contributions")
    plt.tight_layout()
    fig.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_contributors_image_url": forecast_image_url}), 200


@app.route('/api/forecast_releases_prophet', methods=['POST'])
def forecast_releases_prophet():

    body = request.get_json()
    releases = body.get("releases", [])
    repo_name = body.get("repo", "unknown_repo")

    if not releases:
        return jsonify({"error": "No release data provided"}), 400

    try:
        # Preprocess release data
        release_dates = [
            datetime.strptime(release["published_at"], "%Y-%m-%dT%H:%M:%SZ")
            for release in releases if "published_at" in release
        ]
        if len(release_dates) < 2:  # Check for insufficient data
            return jsonify({"error": "Not enough data to forecast releases. At least 2 data points are required."}), 400

        # Create a DataFrame and process the data
        df = pd.DataFrame({"date": release_dates})
        df["month"] = df["date"].dt.to_period("M")
        df["count"] = 1
        monthly_releases = df.groupby("month").size().reset_index(name="count")
        monthly_releases["month"] = monthly_releases["month"].dt.to_timestamp()

        # Fill missing months with zeros
        all_months = pd.date_range(
            start=monthly_releases["month"].min(),
            end=monthly_releases["month"].max(),
            freq="MS"
        )
        filled_df = (
            monthly_releases.set_index("month")
            .reindex(all_months, fill_value=0)
            .reset_index()
            .rename(columns={"index": "ds", "count": "y"})
        )

        print(f"Processed release data for Prophet:\n{filled_df}")

        # Ensure 'ds' and 'y' columns exist
        if "ds" not in filled_df.columns or "y" not in filled_df.columns:
            return jsonify({"error": "Processed data is missing required columns for Prophet ('ds' and 'y')."}), 400

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(filled_df[["ds", "y"]])

        # Create future dataframe for the next 12 months
        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)

        # Extract forecast results
        forecasted_releases = forecast[["ds", "yhat"]].tail(12)

        # Save the forecast plot locally
        LOCAL_IMAGE_PATH = "static/images/"
        os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
        FORECAST_IMAGE_NAME = f"forecast_releases_prophet_{repo_name}.png"
        full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(filled_df["ds"], filled_df["y"], label="Actual Releases", marker="o", linestyle="-", color="blue")
        plt.plot(forecasted_releases["ds"], forecasted_releases["yhat"], label="Forecasted Releases", marker="x", linestyle="--", color="purple")
        plt.title(f"Prophet Forecasted Releases for {repo_name}")
        plt.xlabel("Month")
        plt.ylabel("Number of Releases")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_local_path)
        plt.close()

        # Upload the image to Google Cloud Storage
        BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
        BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(FORECAST_IMAGE_NAME)
            blob.upload_from_filename(full_local_path)

            # Generate the public URL for the image
            forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
        except Exception as e:
            print(f"Error uploading image to Google Cloud Storage: {e}")
            forecast_image_url = f"/{full_local_path}"

        return jsonify({"forecast_releases_image_url": forecast_image_url}), 200

    except Exception as e:
        print(f"Error during processing or forecasting: {e}")
        return jsonify({"error": "Failed to generate forecast."}), 500

###StatsModel

@app.route('/api/forecast_statsmodels', methods=['POST'])
def forecast_statsmodels():
    from pmdarima import auto_arima  # For automatic ARIMA parameter selection

    body = request.get_json()
    issues = body["issues"]
    issue_type = body["type"]
    repo_name = body["repo"]

    # Prepare DataFrame
    data_frame = pd.DataFrame(issues)
    df = data_frame[[issue_type, 'issue_number']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Extract day of week and month
    df['day_of_week'] = df['ds'].dt.day_name()
    df['month_of_year'] = df['ds'].dt.month_name()

    # Insights
    max_created_day = df[df['ds'].notnull()]['day_of_week'].value_counts().idxmax()
    max_closed_day = df[df['ds'].notnull()]['day_of_week'].value_counts().idxmax()
    max_closed_month = df['month_of_year'].value_counts().idxmax()

    # Prepare time series data
    df = df.groupby('ds', as_index=False).sum()
    df['y'] = df['y'].astype('float32')
    data = df.set_index('ds')['y']

    # Resampling to daily frequency and filling missing values
    data = data.asfreq('D').fillna(0)

    # Split into train and test
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # Use auto_arima to find the best ARIMA parameters
    auto_arima_model = auto_arima(
        train, 
        seasonal=False, 
        trace=True, 
        suppress_warnings=True,
        stepwise=True
    )
    order = auto_arima_model.order
    print(f"Best ARIMA order: {order}")

    # Fit ARIMA model with the best parameters
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast values
    forecast = model_fit.forecast(steps=len(test))
    forecast_full = model_fit.forecast(steps=len(test) + 12)  # Include future 12 months for plotting

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_CREATED_IMAGE_NAME = f"forecast_created_statsmodels_{repo_name}.png"
    FORECAST_CLOSED_IMAGE_NAME = f"forecast_closed_statsmodels_{repo_name}.png"

    # Plot Created Issues Forecast
    plt.figure(figsize=(10, 4))
    plt.plot(train.index, train, label='Train Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, forecast, label='Forecasted Data', linestyle='--', color='orange')
    plt.title(f'Created Issues Forecast - Statsmodels ARIMA ({repo_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME)
    plt.close()

    # Closed Issues Forecast Plot (Reuse similar logic for closed issues)
    plt.figure(figsize=(10, 4))
    plt.plot(train.index, train, label='Train Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, forecast, label='Forecasted Data', linestyle='--', color='green')
    plt.title(f'Closed Issues Forecast - Statsmodels ARIMA ({repo_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOCAL_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME)
    plt.close()

    # Upload the images to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)

        created_blob = bucket.blob(FORECAST_CREATED_IMAGE_NAME)
        created_blob.upload_from_filename(LOCAL_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME)

        closed_blob = bucket.blob(FORECAST_CLOSED_IMAGE_NAME)
        closed_blob.upload_from_filename(LOCAL_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME)

        # Generate the public URLs for the images
        forecast_created_image_url = BASE_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME
        forecast_closed_image_url = BASE_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_created_image_url = f"/{LOCAL_IMAGE_PATH + FORECAST_CREATED_IMAGE_NAME}"
        forecast_closed_image_url = f"/{LOCAL_IMAGE_PATH + FORECAST_CLOSED_IMAGE_NAME}"

    # Return JSON response
    json_response = {
        "forecast_created_image_url": forecast_created_image_url,
        "forecast_closed_image_url": forecast_closed_image_url,
        "max_created_day_of_week": max_created_day,
        "max_closed_day_of_week": max_closed_day,
        "max_closed_month_of_year": max_closed_month,
    }

    return jsonify(json_response)


@app.route('/api/forecast_pulls_holtwinter', methods=['POST'])
def forecast_pulls_holtwinter():
    body = request.get_json()
    pulls = body.get("pulls", [])
    repo_name = body.get("repo", "unknown_repo")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    if not pulls:
        return jsonify({"error": "No pull request data provided"}), 400

    # Preprocess pull requests data
    df = pd.DataFrame(pulls)
    if "created_at" not in df.columns:
        return jsonify({"error": "No 'created_at' field found in pull request data"}), 400

    # Parse and clean dates
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    # Aggregate data monthly
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty or len(grouped_df) < 6:  # Ensure at least 6 months of data
        return jsonify({"error": "Not enough data to forecast."}), 400

    # Prepare data for Holt-Winters Exponential Smoothing
    try:
        if len(grouped_df) >= 24:  # Use seasonal model for enough data
            model = ExponentialSmoothing(
                grouped_df["count"], seasonal="add", seasonal_periods=12
            ).fit()
        else:  # Use simple exponential smoothing for less data
            model = SimpleExpSmoothing(grouped_df["count"]).fit()

        # Forecast the next 12 months
        forecast = model.forecast(12)
        forecast_dates = pd.date_range(
            start=grouped_df["month"].iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS"
        )
        forecast_df = pd.DataFrame({"month": forecast_dates, "forecast": forecast})

        # Save the forecast plot locally
        LOCAL_IMAGE_PATH = "static/images/"
        os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
        FORECAST_IMAGE_NAME = f"forecast_pulls_holtwinter_{repo_name}.png"
        full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

        # Plot historical and forecasted data
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_df["month"], grouped_df["count"], label="Historical Data", marker="o", color="blue")
        plt.plot(forecast_df["month"], forecast_df["forecast"], label="Forecasted Data", marker="x", linestyle="--", color="red")
        plt.axvline(forecast_df["month"].iloc[0], color="green", linestyle=":", label="Forecast Start")
        plt.title(f"Pull Requests Forecast using Holt-Winters for {repo_name}")
        plt.xlabel("Month")
        plt.ylabel("Number of Pull Requests")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_local_path)
        plt.close()

        # Upload the image to Google Cloud Storage
        BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
        BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(FORECAST_IMAGE_NAME)
            blob.upload_from_filename(full_local_path)

            # Generate the public URL for the image
            forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
        except Exception as e:
            print(f"Error uploading image to Google Cloud Storage: {e}")
            forecast_image_url = f"/{full_local_path}"

        return jsonify({
            "forecast_pulls_image_url": forecast_image_url
        }), 200

    except Exception as e:
        print(f"Error during Holt-Winters forecasting: {e}")
        return jsonify({"error": f"Failed to forecast using Holt-Winters: {str(e)}"}), 500



@app.route('/api/forecast_commits_arima', methods=['POST'])
def forecast_commits_arima():
    body = request.get_json()
    commits = body.get("commits", [])
    repo_name = body.get("repo", "unknown_repo")

    if not commits:
        return jsonify({"error": "No commit data provided"}), 400

    # Preprocess commit data
    df = pd.DataFrame(commits)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")  # Group by month
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()  # Convert to timestamp

    if grouped_df.empty:
        return jsonify({"error": "No valid commit data available"}), 400

    print(f"Processed Commit Data for {repo_name}:\n", grouped_df)

    # Prepare time series data
    ts = grouped_df.set_index("month")["count"]

    # Ensure data sufficiency
    if len(ts) < 12:  # Require at least 12 data points
        return jsonify({"error": "Not enough data to forecast."}), 400

    try:
        # Fit ARIMA model
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(ts, order=(5, 1, 0))  # ARIMA(p=5, d=1, q=0) as an example
        model_fit = model.fit()

        # Forecast the next 12 months
        forecast = model_fit.get_forecast(steps=12)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
        forecast_df = pd.DataFrame({
            "month": forecast_index,
            "forecast": forecast.predicted_mean
        })

        # Save the forecast plot locally
        LOCAL_IMAGE_PATH = "static/images/"
        os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
        FORECAST_IMAGE_NAME = f"forecast_commits_arima_{repo_name}.png"
        full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

        # Plot historical and forecasted data
        plt.figure(figsize=(10, 6))
        plt.plot(ts.index, ts.values, label="Historical Data", marker="o", color="blue")
        plt.plot(forecast_df["month"], forecast_df["forecast"], label="Forecasted Data", marker="x", linestyle="--", color="red")
        plt.axvline(forecast_df["month"].iloc[0], color="green", linestyle=":", label="Forecast Start")
        plt.title(f"Forecasted Commits for {repo_name} (ARIMA)")
        plt.xlabel("Month")
        plt.ylabel("Number of Commits")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_local_path)
        plt.close()

        # Upload the image to Google Cloud Storage
        BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
        BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(FORECAST_IMAGE_NAME)
            blob.upload_from_filename(full_local_path)

            # Generate the public URL for the image
            forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
        except Exception as e:
            print(f"Error uploading image to Google Cloud Storage: {e}")
            forecast_image_url = f"/{full_local_path}"

        return jsonify({
            "forecast_commits_image_url": forecast_image_url
        }), 200

    except Exception as e:
        print(f"Error during ARIMA forecasting: {e}")
        return jsonify({"error": f"Failed to forecast using ARIMA: {str(e)}"}), 500


@app.route('/api/forecast_branches_arima', methods=['POST'])
def forecast_branches_arima():
    from datetime import timedelta
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np

    body = request.get_json()
    branches = body.get("branches", [])
    repo_name = body.get("repo", "unknown_repo")

    if not branches:
        return jsonify({"error": "No branch data provided"}), 400

    # Preprocess branch data
    df = pd.DataFrame(branches)

    # Simulate creation dates for branches (assign dates incrementally)
    start_date = datetime.now() - timedelta(days=len(df))  # Start from `len(df)` days ago
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month").size().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty:
        return jsonify({"error": "No valid branch data available"}), 400

    print(f"Processed Branch Data for {repo_name}:\n", grouped_df)

    # Prepare data for ARIMA
    ts_data = grouped_df.set_index("month")["count"]
    ts_data = ts_data.asfreq("M").fillna(0)  # Ensure monthly frequency and fill missing values with 0

    # Fit ARIMA model
    try:
        arima_model = ARIMA(ts_data, order=(5, 1, 0))  # Example ARIMA order (p, d, q)
        arima_result = arima_model.fit()
    except Exception as e:
        print(f"ARIMA model fitting failed: {e}")
        return jsonify({"error": "Failed to fit ARIMA model"}), 500

    # Forecast for the next 12 months
    forecast_steps = 12
    forecast = arima_result.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq="MS")

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({"Month": forecast_dates, "Forecasted Branches": forecast})

    # Save the forecast plot locally
    LOCAL_IMAGE_PATH = "static/images/"
    os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
    FORECAST_IMAGE_NAME = f"forecast_branches_arima_{repo_name}.png"
    full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(ts_data.index, ts_data, label="Historical Data", marker="o", color="blue")
    plt.plot(forecast_df["Month"], forecast_df["Forecasted Branches"], label="Forecasted Data", marker="x", linestyle="--", color="red")
    plt.axvline(forecast_df["Month"].iloc[0], color="green", linestyle=":", label="Forecast Start")
    plt.title(f"ARIMA Forecast for Branches - {repo_name}")
    plt.xlabel("Month")
    plt.ylabel("Number of Branches")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_local_path)
    plt.close()

    # Upload the image to Google Cloud Storage
    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(FORECAST_IMAGE_NAME)
        blob.upload_from_filename(full_local_path)

        # Generate the public URL for the image
        forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
    except Exception as e:
        print(f"Error uploading image to Google Cloud Storage: {e}")
        forecast_image_url = f"/{full_local_path}"

    return jsonify({"forecast_branches_image_url": forecast_image_url}), 200


@app.route('/api/forecast_contributors_statsmodel', methods=['POST'])
def forecast_contributors_statsmodel():
    from statsmodels.tsa.arima.model import ARIMA

    body = request.get_json()
    contributors = body.get("contributors", [])
    repo_name = body.get("repo", "unknown_repo")

    if not contributors:
        return jsonify({"error": "No contributor data provided"}), 400

    # Preprocess contributor data
    df = pd.DataFrame(contributors)

    # Simulate timestamps for contributors
    start_date = datetime.now() - timedelta(days=len(df))
    df["created_at"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Group by month
    df["month"] = df["created_at"].dt.to_period("M")
    grouped_df = df.groupby("month")["contributions"].sum().reset_index(name="count")
    grouped_df["month"] = grouped_df["month"].dt.to_timestamp()

    if grouped_df.empty or len(grouped_df) < 6:  # Require at least 6 months of data
        return jsonify({"error": "Not enough data to forecast"}), 400

    print(f"Processed Contributor Data for {repo_name}:\n", grouped_df)

    # Prepare data for ARIMA
    time_series = grouped_df.set_index("month")["count"]

    try:
        # Fit ARIMA model
        model = ARIMA(time_series, order=(1, 1, 1))  # Adjust order as needed
        results = model.fit()

        # Forecast for the next 12 months
        forecast = results.forecast(steps=12)
        forecast_index = pd.date_range(
            start=grouped_df["month"].iloc[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS"
        )
        forecast_df = pd.DataFrame({"Month": forecast_index, "Forecasted Contributions": forecast.values})

        # Save the forecast plot locally
        LOCAL_IMAGE_PATH = "static/images/"
        os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
        FORECAST_IMAGE_NAME = f"forecast_contributors_statsmodel_{repo_name}.png"
        full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_df["month"], grouped_df["count"], label="Historical Data", marker="o", color="blue")
        plt.plot(forecast_df["Month"], forecast_df["Forecasted Contributions"], label="Forecasted Data", marker="x", linestyle="--", color="red")
        plt.axvline(forecast_df["Month"].iloc[0], color="green", linestyle=":", label="Forecast Start")
        plt.title(f"ARIMA Forecast for Contributions - {repo_name}")
        plt.xlabel("Month")
        plt.ylabel("Number of Contributions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_local_path)
        plt.close()

        # Upload the image to Google Cloud Storage
        BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
        BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(FORECAST_IMAGE_NAME)
            blob.upload_from_filename(full_local_path)

            # Generate the public URL for the image
            forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
        except Exception as e:
            print(f"Error uploading image to Google Cloud Storage: {e}")
            forecast_image_url = f"/{full_local_path}"

        return jsonify({"forecast_contributors_image_url": forecast_image_url}), 200

    except Exception as e:
        print(f"ARIMA modeling failed: {e}")
        return jsonify({"error": f"ARIMA modeling failed: {str(e)}"}), 500




@app.route('/api/forecast_releases_statsmodel', methods=['POST'])
def forecast_releases_statsmodel():
    from statsmodels.tsa.arima.model import ARIMA

    body = request.get_json()
    releases = body.get("releases", [])
    repo_name = body.get("repo", "unknown_repo")

    if not releases:
        return jsonify({"error": "No release data provided"}), 400

    try:
        # Prepare release data
        release_dates = [
            datetime.strptime(release["published_at"], "%Y-%m-%dT%H:%M:%SZ")
            for release in releases if "published_at" in release
        ]
        if len(release_dates) < 3:
            return jsonify({"error": "Not enough data to forecast. At least 3 months of data are required."}), 400

        # Create a DataFrame and process the data
        df = pd.DataFrame({"date": release_dates})
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        df["count"] = 1
        monthly_releases = df.groupby("month").size().reset_index(name="count")
        monthly_releases["month"] = monthly_releases["month"].dt.to_timestamp()

        # Ensure all months are represented and fill missing months with zeros
        all_months = pd.date_range(
            start=monthly_releases["month"].min(),
            end=monthly_releases["month"].max(),
            freq="MS"
        )
        df = monthly_releases.set_index("month").reindex(all_months, fill_value=0).reset_index()
        df.columns = ["month", "count"]

        print(f"Processed release data for ARIMA model:\n{df}")

        # Fit ARIMA model on the full two years of data
        model = ARIMA(df["count"], order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast for the next 12 months
        forecast = model_fit.forecast(steps=12)
        forecast_dates = pd.date_range(
            start=df["month"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=12,
            freq="MS"
        )
        forecast_df = pd.DataFrame({"Month": forecast_dates, "Forecasted Releases": forecast})

        # Save the forecast plot locally
        LOCAL_IMAGE_PATH = "static/images/"
        os.makedirs(LOCAL_IMAGE_PATH, exist_ok=True)
        FORECAST_IMAGE_NAME = f"forecast_releases_statsmodel_{repo_name}.png"
        full_local_path = LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(df["month"], df["count"], label="Historical Data", marker="o", color="blue")
        plt.plot(forecast_df["Month"], forecast_df["Forecasted Releases"], label="Forecasted Data", marker="x", linestyle="--", color="red")
        plt.axvline(forecast_df["Month"].iloc[0], color="green", linestyle=":", label="Forecast Start")
        plt.title(f"Releases Forecast for {repo_name}")
        plt.xlabel("Month")
        plt.ylabel("Number of Releases")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(full_local_path)
        plt.close()

        # Upload the image to Google Cloud Storage
        BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'Your_Base_Image_path/')
        BUCKET_NAME = os.environ.get('BUCKET_NAME', 'Your_BUCKET_NAME')

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(FORECAST_IMAGE_NAME)
            blob.upload_from_filename(full_local_path)

            # Generate the public URL for the image
            forecast_image_url = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME
        except Exception as e:
            print(f"Error uploading image to Google Cloud Storage: {e}")
            forecast_image_url = f"/{full_local_path}"

        return jsonify({"forecast_releases_image_url": forecast_image_url}), 200

    except Exception as e:
        print(f"Error during processing or forecasting: {e}")
        return jsonify({"error": "Failed to generate forecast."}), 500





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=False)
