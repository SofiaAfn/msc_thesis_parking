def plot_rolling_statistics(df, hours_list):
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(x=df.index, y=df['car_count'], mode='lines', name='Original Data'))
    
    # RGB values for the colors
    color_dict = {
        hours_list[0]: '39, 176, 245',  # blue
        hours_list[1]: '245, 40, 145', # red
        hours_list[2]: '39, 245, 99'  # green
    }
    
    for hours in hours_list:
        window_str = f"{hours}H"
        
        # Calculate rolling statistics
        rolling_mean = df['car_count'].rolling(window=window_str).mean()
        rolling_std = df['car_count'].rolling(window=window_str).std()
        
        # Get the RGB value for the current window size
        color_rgb = color_dict.get(hours, '0,0,0')  # default to black if not found
        
        # Visualization
        legend_group = f"group_{hours}"
        
        fig.add_trace(go.Scatter(x=df.index,
                                 y=rolling_mean,
                                 mode='lines',
                                 line=dict(color=f'rgb({color_rgb})',
                                           width= 3),
                                 name=f'{window_str} Rolling Mean',
                                 legendgroup=legend_group
                                 ))
        
        fig.add_trace(go.Scatter(x=df.index,
                                 y=rolling_mean + rolling_std,
                                 mode='lines',
                                 fill= 'tonexty',
                                 fillcolor= f'rgba({color_rgb}, 0.2)',
                                 line=dict(color=f'rgba({color_rgb}, 0.8)',
                                           dash= 'dash',
                                           width= 2,),
                                 name=f'+1 std ({window_str})',
                                 legendgroup=legend_group
                                 ))
        
        fig.add_trace(go.Scatter(x=df.index,
                                 y=rolling_mean - rolling_std,
                                 mode='lines',
                                 fill= 'tonexty',
                                 fillcolor= f'rgba({color_rgb}, 0.2)',
                                 line=dict(color=f'rgba({color_rgb}, 0.8)',
                                           dash= 'dash',
                                           width= 2),
                                 name=f'-1 std ({window_str})',
                                 legendgroup=legend_group
                                 ))
        
        
        
    fig.update_layout(title='Rolling Statistics Plot', xaxis_title='Timestamp', yaxis_title='Car Count', height=800)
    fig.show()

# Usage:
plot_rolling_statistics(data_df, [3, 6, 24])

# helper functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.subplots as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_evaluation_metrics(gpr_model, y_true, y_pred):
    ''' 
        This function will take the gpr model and true values and predicted values to create 
    '''
    # Calculate residuals
    residuals = y_true - y_pred

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    log_marginal_likelihood = gpr_model.log_marginal_likelihood(gpr_model.kernel_.theta)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    kernel = gpr_model.kernel_

    # Create subplots with an additional column for metrics
    fig = sp.make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("Residuals", "Predicted vs Actual"))

    # Plot residuals
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals ',
            marker=dict(color='blue')
        ),
        row=1, col=1
    )
    # Add mean line to residuals plot
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=[residuals.mean()]*len(y_pred),
            mode='lines',
            name='Mean Residual',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )

    # Plot predicted vs actual values
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predicted vs Actual ',
            marker=dict(color='red')
        ),
        row=1, col=2
    )
    # Add line y=x to predicted vs actual plot
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_true,
            mode='lines',
            name='y=x line',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=2
    )

    # Add metrics as text in the third column
    metrics_text = f"R-squared: {r2:.2f} | "\
                f"LML: {log_marginal_likelihood:.2f} | "\
                f"MAE: {mae:.2f} | "\
                f"MSE: {mse:.2f} | "\
                f"RMSE: {rmse:.2f} | "\
                #f"Learned kernel: {kernel} |"
    
    fig.add_annotation(dict(font=dict(size=15),
                                        x=0,
                                        y=-0.14,
                                        showarrow=False,
                                        text= metrics_text,
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    
    fig.add_annotation(dict(font=dict(size=15),
                                        x=0,
                                        y=1.08,
                                        showarrow=False,
                                        text= f"Learned kernel: {kernel}" ,
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
        
    
    

    # Update layout
    fig.update_layout(title="Model Evaluation Metrics", height=700)
    fig.update_xaxes(title_text="Predicted Values ", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Actual Values", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=2)
    fig.show()

def plot_gpr_samples_plotly( gpr_model, n_samples, X_train, y_train):
    '''
        Plots samples from a Gaussian Process Regression model using Plotly. By default plots from a distribution if gpr is not trained.
    '''
    
    x = x = np.linspace(X_train.min(), X_train.max(), 1000) # field to draw out prior and posteriors 
    X = x.reshape(-1,1)

    y_mean, y_std = gpr_model.predict( X, return_std=True)
    y_samples = gpr_model.sample_y( X, n_samples)

    fig = go.Figure()

    for idx, single_prior in enumerate(y_samples.T):
        fig.add_trace(
            go.Scatter(
                x=x, y=single_prior, mode="lines", name=f"Sampled function #{idx + 1}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean,
            mode="lines",
            line_color="red",
            name="Mean",
            line=dict(dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean - y_std,
            fill=None,
            mode="lines",
            line_color="rgba(255,0,0,0.1)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean + y_std,
            fill="tonexty",
            mode="lines",
            line_color="rgba(173, 216, 230, 0.5)",
            name=r"uncertainty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_train.ravel(),
            y=y_train.ravel(),
            mode='markers',
            name="Training Data",
            marker=dict(symbol='cross', size=4, color= 'royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=X_train.ravel(),
            y=y_train.ravel(),
            mode='lines',
            name="Training Data Signal",
            marker=dict(symbol='cross', size=4, color= 'royalblue')
        )
    )
    
    # fig.add_trace(
    #     go.Scatter(
    #         x=X_test.ravel(),
    #         y=y_test.ravel(),
    #         mode='lines+markers',
    #         name="Test Data",
    #         marker=dict(symbol='cross', size=4, color= 'green')
    #     )
    # )

    fig.update_layout(
        xaxis_title="Unix Epoch time",
        yaxis_title="Count",
        height= 800,
    )

    fig.show()

def gpr_train(gpr, x_train, y_train, x_test, y_test):
    '''
        Trains a Gaussian Process Regressor on the given training data and makes predictions on test data. 
    '''
    # Create the GPR model outside this func to have more control over kernels

    # Fit the GPR model to the training data
    gpr.fit(x_train, y_train)
    
    plot_gpr_samples_plotly(gpr, 5,x_train,y_train)
    # Perform predictions using the trained GPR model
    y_pred, y_std = gpr.predict(x_test, return_std=True)
    # y_pred: Predicted target values
    # y_std: Standard deviation of predictions

    # Access learned model properties
    kernel_params = gpr.kernel_  # Learned kernel parameters
    # noise_level = gpr.kernel_.get_params()['k2_noise_level']  # Estimated noise level (if available)
    noise_level = gpr.alpha_  # Estimated noise level (if available)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    lml = gpr.log_marginal_likelihood(gpr.kernel_.theta)

    # evaluation metrics
    print(f"MAE: {mae}") # 
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}%")
    print(
        f"Kernel parameters after fit: \n{gpr.kernel_} \n"
        f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}\n")

    # Print or analyze the results as needed
    print("Predicted values:", y_pred)
    # print("Prediction uncertainty (std):", y_std)
    print("Learned kernel parameters:", kernel_params)
    # print("Estimated noise level:", noise_level)

    return y_pred, y_std, gpr# --> handling index and timestamps first

data_df.reset_index(inplace=True)

data_df.drop(columns=["index","image name"], inplace=True)


data_df.set_index("timestamp_true", inplace=True)


data_df.sort_index(ascending=True, inplace= True)

# convert the timestamp into unix epoch, to minute level precision
data_df['num_timestamp'] =  (pd.to_datetime(data_df.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1min')

# # --> type setting for vars
data_df["car_count"] = data_df["car_count"]

# # --> deriving features

data_df["month_no"] = data_df.index.month

data_df["month_name"] = data_df.index.month_name()

data_df["day"] = data_df.index.day

data_df["day_of_week"] = data_df.index.dayofweek

data_df["day_of_week_name"] = data_df.index.day_name()

data_df["is_weekend"] = np.where(
    data_df.index.isin(["Sunday", "Saturday"]), 1, 0
)
data_df["hour_of_day"] = data_df.index.hour

data_df['minutes'] = data_df.index.minute


data_df["min_of_day"] = data_df['hour_of_day'] * 60 + data_df['minutes']

data_df['week_no']=  data_df.index.weekday
# Combine the columns in the desired format
data_df['combined'] = data_df.apply(lambda row: f"{str(row['month_no']).zfill(2)}{str(row['day']).zfill(2)}{str(row['hour_of_day']).zfill(2)}{str(row['minutes']).zfill(2)}", axis=1)


lengths = data_df['combined'].str.len()
unique_lengths = lengths.value_counts()
unique_lengths
# --> convert unix epoch time back to our timestamp format
# 
# pd.Timestamp("1970-01-01") + pd.to_timedelta(data_df['num_timestamp'][1], unit='min')

# Create the plot
fig = go.Figure()

# Create the line chart using Plotly Express
fig.add_trace(go.Scatter( x=data_df.index, y=data_df['car_count'], mode='lines+markers' ,marker=dict(symbol='cross', size=5),name = "Count over time"))
# Layout and titles
fig.update_layout(title='Car count distribution',
                  xaxis_title='timestamp',
                  yaxis_title='car_count',
                  showlegend=True,
                  height = 600
                  )
# Show the plot
fig.show()
# Create the plot
fig = go.Figure()

# Create the line chart using Plotly Express
fig.add_trace(go.Scatter( x=data_df['num_timestamp'], y=data_df['car_count'], mode='lines+markers' ,marker=dict(symbol='cross', size=3)))

# Show the plot
fig.show()
 #Create the plot
fig = go.Figure()

# Create the line chart using Plotly Express
fig.add_trace(go.Histogram(x=data_df['car_count'], name='car count dist'))

# Layout and titles
fig.update_layout(title='Car count distribution',
                  xaxis_title='num_timestamp',
                  yaxis_title='car_count',
                  showlegend=True)


# Show the plot
fig.show()
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Assuming data_df is your DataFrame
data_df = data_df.sort_index()  # Ensure the data is sorted by timestamp

# feature to shift the mean on y axis to 0, normalization 
data_df['centered_car_count'] = data_df['car_count'] - data_df['car_count'].mean() 
data_length = len(data_df)

# Splitting the data into training (70%), validation (15%), and test (15%) sets
train = data_df.iloc[:int(0.7 * data_length)]
valid = data_df.iloc[int(0.7 * data_length):int(0.8 * data_length)]
test = data_df.iloc[int(0.8 * data_length):]

# Extracting the features and target variable for each split
X_train = train['num_timestamp'].values.reshape(-1,1)
y_train = train['car_count']
X_valid = valid['num_timestamp'].values.reshape(-1,1)
y_valid = valid['car_count']
X_test  = test['num_timestamp']
y_test = test['car_count']


type(data_df['num_timestamp'][1])
# # Subtracting the mean from the 'car_count' column
# data_df['centered_car_count'] = data_df['car_count'] - data_df['car_count'].mean()

# Now, you can use 'centered_car_count' as your target variable for modeling.
# Extracting the features and target variable for each split
X_train_norm  = train['num_timestamp'].values.reshape(-1,1)
y_train_norm  = train['centered_car_count']
X_valid_norm  = valid['num_timestamp'].values.reshape(-1,1)
y_valid_norm  = valid['centered_car_count']
X_test_norm  = test['num_timestamp'].values.reshape(-1,1)
y_test_norm  = test['centered_car_count']

import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

# Assuming data_df is your DataFrame and 'centered_car_count' is the column you're analyzing
series = data_df['centered_car_count']

# 1. Visualize the time series
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Original Series'))
fig.update_layout(title='Time Series', xaxis_title='Date', yaxis_title='Value')
fig.show()

# 2. Perform the ADF test
result = adfuller(series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# A rule of thumb to interpret the p-value:
if result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")

# 3. Visualize the rolling mean and standard deviation
rolling_mean = series.rolling(window=12).mean()
rolling_std = series.rolling(window=12).std()

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Original Series'))
fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling Std Dev'))
fig.update_layout(title='Rolling Mean & Standard Deviation', xaxis_title='Date', yaxis_title='Value')
fig.show()

# Partial normalisation here 
data_df = data_df.sort_index()  # Ensure the data is sorted by timestamp

# feature to shift the mean on y axis to 0, normalization 
data_df['centered_car_count'] = data_df['car_count'] - data_df['car_count'].mean() 
data_length = len(data_df)
data_df['car_count', 'centered_']# helper functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.subplots as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_evaluation_metrics(gpr_model, y_true, y_pred):
    ''' 
        This function will take the gpr model and true values and predicted values to create 
    '''
    # Calculate residuals
    residuals = y_true - y_pred

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    log_marginal_likelihood = gpr_model.log_marginal_likelihood(gpr_model.kernel_.theta)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    kernel = gpr_model.kernel_

    # Create subplots with an additional column for metrics
    fig = sp.make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("Residuals", "Predicted vs Actual"))

    # Plot residuals
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals ',
            marker=dict(color='blue')
        ),
        row=1, col=1
    )
    # Add mean line to residuals plot
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=[residuals.mean()]*len(y_pred),
            mode='lines',
            name='Mean Residual',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )

    # Plot predicted vs actual values
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predicted vs Actual ',
            marker=dict(color='red')
        ),
        row=1, col=2
    )
    # Add line y=x to predicted vs actual plot
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_true,
            mode='lines',
            name='y=x line',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=2
    )

    # Add metrics as text in the third column
    metrics_text = f"R-squared: {r2:.2f} | "\
                f"LML: {log_marginal_likelihood:.2f} | "\
                f"MAE: {mae:.2f} | "\
                f"MSE: {mse:.2f} | "\
                f"RMSE: {rmse:.2f} | "\
                #f"Learned kernel: {kernel} |"
    
    fig.add_annotation(dict(font=dict(size=15),
                                        x=0,
                                        y=-0.14,
                                        showarrow=False,
                                        text= metrics_text,
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
    
    fig.add_annotation(dict(font=dict(size=15),
                                        x=0,
                                        y=1.08,
                                        showarrow=False,
                                        text= f"Learned kernel: {kernel}" ,
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
        
    
    

    # Update layout
    fig.update_layout(title="Model Evaluation Metrics", height=700)
    fig.update_xaxes(title_text="Predicted Values ", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Actual Values", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=2)
    fig.show()

def plot_gpr_samples_plotly( gpr_model, n_samples, X_train, y_train):
    '''
        Plots samples from a Gaussian Process Regression model using Plotly. By default plots from a distribution if gpr is not trained.
    '''
    
    x = x = np.linspace(X_train.min(), X_train.max(), 1000) # field to draw out prior and posteriors 
    X = x.reshape(-1,1)

    y_mean, y_std = gpr_model.predict( X, return_std=True)
    y_samples = gpr_model.sample_y( X, n_samples)

    fig = go.Figure()

    for idx, single_prior in enumerate(y_samples.T):
        fig.add_trace(
            go.Scatter(
                x=x, y=single_prior, mode="lines", name=f"Sampled function #{idx + 1}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean,
            mode="lines",
            line_color="red",
            name="Mean",
            line=dict(dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean - y_std,
            fill=None,
            mode="lines",
            line_color="rgba(255,0,0,0.1)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean + y_std,
            fill="tonexty",
            mode="lines",
            line_color="rgba(173, 216, 230, 0.5)",
            name=r"uncertainty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_train.ravel(),
            y=y_train.ravel(),
            mode='markers',
            name="Training Data",
            marker=dict(symbol='cross', size=4, color= 'royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=X_train.ravel(),
            y=y_train.ravel(),
            mode='lines',
            name="Training Data Signal",
            marker=dict(symbol='cross', size=4, color= 'royalblue')
        )
    )
    
    # fig.add_trace(
    #     go.Scatter(
    #         x=X_test.ravel(),
    #         y=y_test.ravel(),
    #         mode='lines+markers',
    #         name="Test Data",
    #         marker=dict(symbol='cross', size=4, color= 'green')
    #     )
    # )

    fig.update_layout(
        xaxis_title="Unix Epoch time",
        yaxis_title="Count",
        height= 800,
    )

    fig.show()

def gpr_train(gpr, x_train, y_train, x_test, y_test):
    '''
        Trains a Gaussian Process Regressor on the given training data and makes predictions on test data. 
    '''
    # Create the GPR model outside this func to have more control over kernels

    # Fit the GPR model to the training data
    gpr.fit(x_train, y_train)
    
    plot_gpr_samples_plotly(gpr, 5,x_train,y_train)
    # Perform predictions using the trained GPR model
    y_pred, y_std = gpr.predict(x_test, return_std=True)
    # y_pred: Predicted target values
    # y_std: Standard deviation of predictions

    # Access learned model properties
    kernel_params = gpr.kernel_  # Learned kernel parameters
    # noise_level = gpr.kernel_.get_params()['k2_noise_level']  # Estimated noise level (if available)
    noise_level = gpr.alpha_  # Estimated noise level (if available)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    lml = gpr.log_marginal_likelihood(gpr.kernel_.theta)

    # evaluation metrics
    print(f"MAE: {mae}") # 
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}%")
    print(
        f"Kernel parameters after fit: \n{gpr.kernel_} \n"
        f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}\n")

    # Print or analyze the results as needed
    print("Predicted values:", y_pred)
    # print("Prediction uncertainty (std):", y_std)
    print("Learned kernel parameters:", kernel_params)
    # print("Estimated noise level:", noise_level)

    return y_pred, y_std, gpr
# Assuming day_mean_df (or day_med_df) is loaded and you want to use it

# Feature Engineering
{}

# Splitting the data 
features = ['hour_of_day']  # You can add more features based on feature engineering
target = 'car_count'
X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(day_mean_df, features, target)

# GPR Training and Evaluation
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Using the training helper function from the notebook
y_pred, y_std, trained_gpr = gpr_train(gpr, X_train.values.reshape(-1, 1), y_train.values, X_valid.values.reshape(-1, 1), y_valid.values)
