import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.subplots as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
import wandb

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C

def preprocess_data(data_df):
    # Handling index and timestamps
    data_df.reset_index(inplace=True)
    data_df.drop(columns=["index", "image name"], inplace=True)
    data_df.set_index("timestamp_true", inplace=True)
    data_df.sort_index(ascending=True, inplace=True)
    
    # Convert the timestamp into unix epoch, to minute level precision
    data_df['num_timestamp'] = (pd.to_datetime(data_df.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1min')
    
    # Type setting for vars
    data_df["car_count"] = data_df["car_count"]
    
    # Deriving features
    data_df["month_no"] = data_df.index.month
    data_df["month_name"] = data_df.index.month_name()
    data_df["day"] = data_df.index.day
    data_df["day_of_week"] = data_df.index.dayofweek
    data_df["day_of_week_name"] = data_df.index.day_name()
    data_df["is_weekend"] = np.where(data_df.index.isin(["Sunday", "Saturday"]), 1, 0)
    data_df["hour_of_day"] = data_df.index.hour
    data_df['minutes'] = data_df.index.minute
    data_df["min_of_day"] = data_df['hour_of_day'] * 60 + data_df['minutes']
    data_df['week_no'] = data_df.index.weekday
    data_df['combined'] = data_df.apply(lambda row: f"{str(row['month_no']).zfill(2)}{str(row['day']).zfill(2)}{str(row['hour_of_day']).zfill(2)}{str(row['minutes']).zfill(2)}", axis=1)
    data_df['centered_car_count'] = data_df['car_count'] - data_df['car_count'].mean()
    
    return data_df

# To use the function:
# processed_df = preprocess_data(pd.read_pickle("Kris_updated_yolov8.pkl"))


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
    
    return fig

def plot_gpr_samples_plotly( gpr_model, n_samples, X_train, y_train, y_mean=None, y_std=None):
    '''
        Plots samples from a Gaussian Process Regression model using Plotly. By default plots from a distribution if gpr is not trained.
    '''
    
    x =  np.linspace(X_train.min(), X_train.max(), len(X_train)) # field to draw out prior and posteriors 
    X = x.reshape(-1,1)
    
        # Predict mean and standard deviation if they are not provided
    if y_mean is None or y_std is None:
        y_mean, y_std = gpr_model.predict(X, return_std=True)

    # Sample from the Gaussian Process model
    y_samples = gpr_model.sample_y(X, n_samples)
            

    fig = go.Figure()
   
    for idx, single_prior in enumerate(y_samples.T):
        fig.add_trace(
            go.Scatter(
                x=x, y=single_prior,
                mode="lines",
                name=f"Sampled fxn #{idx + 1}",
                line=dict(dash="dash")
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_mean,
            mode="lines",
            line_color="red",
            name="Mean",
            
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
            marker=dict(symbol='cross', size=6, color= 'royalblue')
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
        xaxis_title="Agg. Feature",
        yaxis_title="Count",
        height= 800,
    )

    #fig.show()

    
    return fig


def gpr_train_old(gpr,n_samples, x_train, y_train,run, x_test=None, y_test=None):
    '''
        Trains a Gaussian Process Regressor on the given training data and makes predictions on test data (if provided). 
    '''
    # Create the GPR model outside this func to have more control over kernels
    # priors
    img_prior = plot_gpr_samples_plotly(gpr,
                                        n_samples,
                                        x_train,
                                        y_train)
    
    wandb.log({'prior_samples': img_prior})
    
    #save initial kernel parameters as artifacts
    initial_kernel = str(gpr)
    with open("initial_kernel.txt", "w") as f:
        f.write(initial_kernel)
    
    # Fit the GPR model to the training data
    gpr.fit(x_train, y_train)
    
    # posterior 
    img_posterior = plot_gpr_samples_plotly(gpr,
                                            n_samples,
                                            x_train,
                                            y_train)
    
    wandb.log({'posterior_samples': img_posterior})

    # Kernel parameters after fit
    after_fit_kernel = str(gpr.kernel_)
    with open("after_fit_kernel.txt", "w") as f:
        f.write(after_fit_kernel)

    
    # Predict on training data
    y_train_pred, _ = gpr.predict(x_train, 
                                  return_std=True)

    # Metrics on training data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

    # If test data is provided, predict and calculate metrics on it
    if x_test is not None and y_test is not None:
        # validatioin
        y_pred, y_std = gpr.predict(x_test, return_std=True)
        
        mae_test = mean_absolute_error(y_test, y_pred)
        mse_test = mean_squared_error(y_test, y_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_pred)
        mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    else:
        y_pred, y_std = None, None
        mae_test, mse_test, rmse_test, r2_test, mape_test = None, None, None, None, None

    lml = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    # kernel_params = gpr.kernel_
    # noise_level = gpr.alpha_

    metrics_dict = {
        "MAE_train": mae_train,
        "MSE_train": mse_train,
        "RMSE_train": rmse_train,
        "R2_train": r2_train,
        "MAPE_train":mape_train,
        
        "MAE_test": mae_test,
        "MSE_test": mse_test,
        "RMSE_test": rmse_test,
        "R2_test": r2_test,
        "MAPE_test": mape_test,
        "LML": lml
    }
    valid_posterior = plot_gpr_posterior_plotly(gpr,x_test, y_test,y_pred, y_std )
    wandb.log({'posterior_samples on test': valid_posterior})
    # Create a wandb Table
    table = wandb.Table(columns=["Metric", "Value"])

    # Populate the table
    for key, value in metrics_dict.items():
        table.add_data(key, value)
        run.config[key] = value
    # Log the table
    wandb.log({"Metrics Table": table})
    # Log  artifacts
    wandb.save("initial_kernel.txt")
    wandb.save("after_fit_kernel.txt")
    
        # wandb.log(metrics_dict)

    return y_pred, y_std


def ad_fuller_test(series):
    res = adfuller(series)
    print('ADF Statistic:', res[0])
    print('p-value:', res[1])
    print('Critical Values:', res[4])

    # A rule of thumb to interpret the p-value:
    if res[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

def gpr_score_card(gpr, y_mean, y_pred):
    # Compute evaluation metrics
    mae = mean_absolute_error(y_mean, y_pred)
    mse = mean_squared_error(y_mean, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_mean, y_pred)
    mape = np.mean(np.abs((y_mean - y_pred) / y_mean)) * 100

    # Extract kernel parameters and other model-specific info
    kernel_params = gpr.kernel_.get_params()
    noise_level = kernel_params.get("k2__noise_level", None)  # Assumes WhiteKernel is named "k2"
    lml = gpr.log_marginal_likelihood()
    
    # wandb.log({
    #     "mae": mae,
    #     "mse": mse,
    #     "rmse": rmse,
    #     "r2": r2,
    #     "mape": mape,
    #     "noise_level": noise_level,
    #     "log_marginal_likelihood": lml
    # })
    metrics_dict = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R-squared": r2,
    "MAPE": f"{mape}%",
    "LML": lml,
    "intial kernel": str(gpr),
    "Kernel parameters after fit": str(gpr.kernel_),
    "noise level": noise_level,
    "Predicted values": y_pred,
    "Learned kernel parameters": kernel_params
}

    # Print the metrics and information
    # print(f"MAE: {mae}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    # print(f"R-squared: {r2}")
    # print(f"MAPE: {mape}%")
    # print(f'LML: {lml}')
        
    # print(f"Kernel parameters after fit: \n{gpr.kernel_} \n")
    # print(f"noise level: {noise_level}")

    # print("Predicted values:", y_pred)
    # print("Learned kernel parameters:", kernel_params)
    
    return metrics_dict

def log_data(data):
    wandb.log(data)
    

def split_data(df, train_ratio=0.6, valid_ratio=0.2):
    """
    Splits a dataframe into train, validate, and test sets.
    """
    
    df = df[['car_count', 'month_no', 'day', 'day_of_week', 'hour_of_day', 'min_of_day', 'centered_car_count','combined']]

    # Calculating the split indices
    train_idx = int(train_ratio * len(df))
    valid_idx = train_idx + int(valid_ratio * len(df))

    # Splitting the original dataframe
    train = df.iloc[:train_idx]
    valid = df.iloc[train_idx:valid_idx]
    test = df.iloc[valid_idx:]
    
    return train, valid, test


def aggregate_data(df, agg_column, agg_func='mean'):
    """
    Aggregates the data based on the given column and aggregation function.
    """
    agg_df = df.groupby(agg_column).agg(agg_func).reset_index()
    return agg_df

def split_target_feature(df, feature, target):
    
    X = df[feature].values.reshape(-1,1)
    y = df[target].values
    
    return X, y

def gpr_train(gpr, x_train, y_train):
    """
    Trains a Gaussian Process Regressor on the given training data.
    """
    # # Prior samples visualization
    # img_prior_train = plot_gpr_samples_plotly(gpr, n_samples, x_train, y_train)
    # img_prior_train.show()
    # wandb.log({'prior_samples': img_prior_train})
    
    # Save initial kernel parameters as artifacts
    initial_kernel = str(gpr)
    with open("initial_kernel.txt", "w") as f:
        f.write(initial_kernel)
    
    # Fit the GPR model to the training data
    gpr.fit(x_train, y_train)
    
    # # Posterior samples visualization
    # img_posterior_train = plot_gpr_samples_plotly(gpr, n_samples, x_train, y_train)
    # wandb.log({'posterior_samples': img_posterior_train})
    
    # Kernel parameters after fit
    after_fit_kernel = str(gpr.kernel_)
    with open("after_fit_kernel.txt", "w") as f:
        f.write(after_fit_kernel)
    
    # Log artifacts
    wandb.save("initial_kernel.txt")
    wandb.save("after_fit_kernel.txt")
    
    return gpr

def calculate_metrics(y_true, y_pred, gpr, suffix):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    lml = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    
    return {
        f"MAE_{suffix}": mae,
        f"MSE_{suffix}": mse,
        f"RMSE_{suffix}": rmse,
        f"R2_{suffix}": r2,
        f"MAPE_{suffix}": mape,
        f"LML_{suffix}": lml
    }

def log_metrics(run, metrics, step, suffix):
    table = wandb.Table(columns=["Metric", "Value"])
    for key, value in metrics.items():
        table.add_data(key, value)
        run.config[key] = value
    wandb.log({f"Metrics Table {suffix.capitalize()}": table, "step": step})
    wandb.log(metrics)

def gpr_pred(run, step, gpr, x_train, y_train, x_test=None, y_test=None, mode='unspecified'):
    
    # Predict on training data and calculate metrics
    y_train_pred, _ = gpr.predict(x_train, return_std=True)
    metrics_train = calculate_metrics(y_train, y_train_pred, gpr, "train")
    log_metrics(run, metrics_train, step, "train")
    
    # If test/validation data is provided, predict and calculate metrics
    if x_test is not None and y_test is not None:
        y_pred, y_std = gpr.predict(x_test, return_std=True)
        metrics_test = calculate_metrics(y_test, y_pred, gpr, mode)
        log_metrics(run, metrics_test, step, mode)
    else:
        y_pred, y_std = None, None

    # Log the kernel parameters
    kernel_str = str(gpr.kernel_)
    with open("kernel.txt", "w") as f:
        f.write(kernel_str)
    wandb.save("kernel.txt")

    return y_pred, y_std, gpr


def plot_gpr_posterior_plotly(gpr, X_test,y_test, y_pred, y_std):
    '''
    Plots the posterior distribution of a GPR model using Plotly.
    X_test here is just used to create the x axis margins of plot
    y_pred can be any predictions from test or validate set
    '''
    x = np.linspace(X_test.min(), X_test.max(), len(X_test))
    
    #pulling samples from test set on a trained gpr
    y_samples = gpr.sample_y(X_test, 5)

    fig = go.Figure()


    for idx, single_prior in enumerate(y_samples.T):
        fig.add_trace(
            go.Scatter(
                x=x, y=single_prior,
                mode="lines",
                name=f"Sampled fxn #{idx + 1}",
                line=dict(dash="dash")
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            line_color="red",
            name="Mean",
            
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred - y_std,
            fill=None,
            mode="lines",
            line_color="rgba(255,0,0,0.1)",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred + y_std,
            fill="tonexty",
            mode="lines",
            line_color="rgba(173, 216, 230, 0.5)",
            name=r"uncertainty",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=X_test.ravel(),
            y=y_test.ravel(),
            mode='lines+markers',
            name="Test Data",
            marker=dict(symbol='cross', size=4, color= 'green')
        )
    )

    fig.update_layout(
        xaxis_title="Aggregated feature",
        yaxis_title="Count",
        height= 800,
    )

    # 
    return  fig

def expanding_window_gpr_process(run,
                                train,
                                gpr,
                                test,
                                ini_train_win_len,
                                valid_win_len,
                                features,
                                target,
                                agg_func = "mean"):

    # Determine the number of steps for the expanding window process
    total_steps = (len(train) - ini_train_win_len) // 24

    # Create a progress bar with twice the total_steps (for train and valid steps)
    pbar = tqdm(total=2*total_steps, desc="Processing expanding window", ncols=100)

    for step in range(1, total_steps + 1):
        
        # Get current train window from data
        train_window_df = train[:ini_train_win_len]
        
        # Get current validation window from data
        valid_window_df = train[ini_train_win_len: ini_train_win_len + valid_win_len]

        # Aggregate train and valid data
        agg_train_win = aggregate_data(train_window_df,
                                       features, agg_func)
        agg_valid_win = aggregate_data(valid_window_df,
                                       features, agg_func)

        # Get target and feature series for each set
        X_train , y_train = split_target_feature(agg_train_win,
                                                 feature= features,
                                                 target=target)
        X_valid, y_valid = split_target_feature(agg_valid_win,
                                                feature= features,
                                                target= target)

        img_prior_train = plot_gpr_samples_plotly(gpr_model= gpr,
                                                  n_samples=5,
                                                  X_train= X_train,
                                                  y_train= y_train)
        
        wandb.log({'prior_samples': img_prior_train})
        
        # Fit the Gaussian Process model to the training data
        gpr = gpr_train(gpr,
                        X_train,
                        y_train)
        
        # Posterior samples visualization
        img_posterior_train = plot_gpr_samples_plotly(gpr,
                                                      5,
                                                      X_train,
                                                      y_train
                                                      )
        wandb.log({'posterior_samples': img_posterior_train})
        
        # Update progress bar after training
        pbar.update(1)

        # Predict on the validation set
        y_pred, y_std, gpr = gpr_pred(run,
                                      step,
                                      gpr,
                                      X_train,
                                      y_train,
                                      x_test=X_valid,
                                      y_test=y_valid,
                                      mode='validate'
                                      )

        # Optionally, you could log final model parameters, metrics, or other artifacts here.
        wandb.log({'Posterior_dist_on_valid': plot_gpr_posterior_plotly(gpr,
                                                                        X_valid,y_valid 
                                                                        ,y_pred,
                                                                        y_std
                                                                        )})
        
        wandb.log({'Residuals_plot_valid': plot_evaluation_metrics(gpr,
                                                                   y_valid, 
                                                                   y_pred
                                                                   )})
        
        # Update progress bar after validation
        pbar.update(1)

        # Expand the training window for the next iteration
        ini_train_win_len += 24
        
        print(f'Step: {step}')

    pbar.close()
    
    print('Testing final GPR on test set')
     
    # Test on unseen
    # set entire test as one window
    #test_win_len = len(test)
    test_step = 1
    # Aggregate test data
    agg_test_win = aggregate_data(test, features, agg_func)

    # Get target and feature series for the test set
    X_test, y_test = split_target_feature(agg_test_win,
                                          feature=features,
                                          target= target)

    # Predict on the test set
    y_test_pred, y_test_std, gpr = gpr_pred(run,
                                            test_step,
                                            gpr,
                                            X_train,
                                            y_train,
                                            x_test=X_test, 
                                            y_test=y_test,
                                            mode='test'
                                            )

    # Log test predictions and evaluate metrics
    wandb.log({'Posterior_dist_on_test': plot_gpr_posterior_plotly(gpr,
                                                                   X_test,
                                                                   y_test,
                                                                   y_test_pred, 
                                                                   y_test_std
                                                                   )})
    
    wandb.log({'Residuals_plot_test': plot_evaluation_metrics(gpr,
                                                              y_test,
                                                              y_test_pred
                                                              )})
              
    return gpr, y_pred, y_std

def sliding_window_gpr_process(run,
                                train,
                                gpr,
                                test,
                                features,
                                target,
                                agg_func = "mean",
                                train_win_len=24, 
                                valid_win_len=24):
    """
    Trains a Gaussian Process using a sliding window approach.
    
    Args:
    - run: The current run context for logging.
    - train: DataFrame containing the training data.
    - gpr: The Gaussian Process model.
    - aggregate_data: Function to aggregate data.
    - split_target_feature: Function to split dataframe into target and features.
    - gpr_train: Function to train the Gaussian Process.
    - gpr_pred: Function to make predictions with the Gaussian Process.
    - train_win_len: Training window length.
    - valid_win_len: Validation window length.
    
    Returns:
    - Trained Gaussian Process model.
    """
    
    # Determine the number of steps for the sliding window process
    total_steps = (len(train) - train_win_len) // valid_win_len

    # Create a progress bar with twice the total_steps (for train and valid steps)
    pbar = tqdm(total=2*total_steps, desc="Processing sliding window", ncols=100)

    for step in range(total_steps):
        
        # Calculate the start and end indices of the current training window
        train_start_idx = step * valid_win_len
        train_end_idx = train_start_idx + train_win_len
        
        # Calculate the start and end indices of the current validation window
        valid_start_idx = train_end_idx
        valid_end_idx = valid_start_idx + valid_win_len

        # Get current train window from data
        train_window_df = train[train_start_idx:train_end_idx]
        
        # Get current validation window from data
        valid_window_df = train[valid_start_idx:valid_end_idx]

        # Aggregate train and valid data
        agg_train_win = aggregate_data(train_window_df, 'hour_of_day', 'mean')
        agg_valid_win = aggregate_data(valid_window_df, 'hour_of_day', 'mean')
    
        # Get target and feature series for each set
        X_train, y_train = split_target_feature(agg_train_win, feature='hour_of_day', target='centered_car_count')
        X_valid, y_valid = split_target_feature(agg_valid_win, feature='hour_of_day', target='centered_car_count')

        
        img_prior_train = plot_gpr_samples_plotly(gpr_model= gpr,
                                                  n_samples=5,
                                                  X_train= X_train,
                                                  y_train= y_train)
        wandb.log({'prior_samples': img_prior_train})
        
        # Fit the Gaussian Process model to the training data
        gpr = gpr_train(gpr,
                        X_train,
                        y_train)
        
        img_posterior_train = plot_gpr_samples_plotly(gpr,5, X_train, y_train)
        wandb.log({'posterior_samples': img_posterior_train})
        
        # Predict on the validation set
        y_pred, y_std, gpr = gpr_pred(run, 
                                      step, 
                                      gpr, 
                                      X_train, 
                                      y_train,
                                      x_test=X_valid,
                                      y_test=y_valid,
                                      mode= 'validate')

        # Optionally, you could log final model parameters, metrics, or other artifacts here.
        wandb.log({'Posterior_dist_on_validate': plot_gpr_posterior_plotly(gpr, X_valid,y_valid ,y_pred, y_std)})
        wandb.log({'Residuals_plot_validate': plot_evaluation_metrics(gpr,y_valid, y_pred) })
        
        # Optionally, you could log final model parameters, metrics, or other artifacts here.
        # ... (logging code here)

        # Update progress bar after training and validation
        pbar.update(2)

        print(f'Step: {step + 1}')

    pbar.close()
    
    print('Testing final GPR on test set')
    # Test on unseen
    # set entire test as one window
    #test_win_len = len(test)
    test_step = 1
    # Aggregate test data
    agg_test_win = aggregate_data(test, features, agg_func)

    # Get target and feature series for the test set
    X_test, y_test = split_target_feature(agg_test_win,
                                          feature=features,
                                          target= target)

    # Predict on the test set
    y_test_pred, y_test_std, gpr = gpr_pred(run,
                                            test_step,
                                            gpr,
                                            X_train,
                                            y_train,
                                            x_test=X_test, 
                                            y_test=y_test,
                                            mode='test'
                                            )

    # Log test predictions and evaluate metrics
    wandb.log({'Posterior_dist_on_test': plot_gpr_posterior_plotly(gpr,
                                                                   X_test,
                                                                   y_test,
                                                                   y_test_pred, 
                                                                   y_test_std
                                                                   )})
    
    wandb.log({'Residuals_plot_test': plot_evaluation_metrics(gpr,
                                                              y_test,
                                                              y_test_pred
                                                              )})
    
    return y_pred, y_std, gpr

def single_window_gpr_process(run, train, validate, test, gpr, features, target, agg_func):
    """
    Trains a Gaussian Process using a single window approach on train, validate, and test datasets.
    
    Args:
    - run: The current run context for logging with wandb.
    - train, validate, test: DataFrames containing the training, validation, and testing data.
    - gpr: The Gaussian Process model.
    - aggregate_data: Function to aggregate data.
    - split_target_feature: Function to split dataframe into target and features.
    - gpr_train: Function to train the Gaussian Process.
    - gpr_pred: Function to make predictions with the Gaussian Process.
    - features: The name of the feature column.
    - target: The name of the target column.
    - agg_func: The aggregation function to apply.
    
    Returns:
    - Trained Gaussian Process model.
    """
    step =1
    # Aggregate train data
    agg_train = aggregate_data(train, features, agg_func)
    X_train, y_train = split_target_feature(agg_train, feature=features, target=target)

    # Plot prior distribution
    img_prior_train = plot_gpr_samples_plotly(gpr, 5, X_train, y_train)
    wandb.log({'prior_samples_train': img_prior_train})

    # Train the GPR on the training data
    gpr = gpr_train(gpr, X_train, y_train)

    # Plot posterior distribution on training data
    img_posterior_train = plot_gpr_samples_plotly(gpr, 5, X_train, y_train)
    wandb.log({'posterior_samples_train': img_posterior_train})

    # Validate the GPR on the validation set
    agg_validate = aggregate_data(validate, features, agg_func)
    X_validate, y_validate = split_target_feature(agg_validate, feature=features, target=target)
    y_validate_pred, y_validate_std, _ = gpr_pred(run,
                                                  gpr=gpr,
                                                  step=step,
                                                  x_train=X_train,
                                                  y_train=y_train,
                                                  x_test=X_validate,
                                                  y_test=y_validate,
                                                  mode='validate')

    # Plot posterior distribution on validation data
    img_posterior_validate = plot_gpr_samples_plotly(gpr, 5, X_validate, y_validate)
    wandb.log({'posterior_samples_validate': img_posterior_validate})

    # Test the GPR on the test set
    agg_test = aggregate_data(test, features, agg_func)
    X_test, y_test = split_target_feature(agg_test, feature=features, target=target)
    y_test_pred, y_test_std, gpr = gpr_pred(run,
                                          gpr=gpr,
                                          step=step,
                                          x_train=X_train,
                                          y_train=y_train,
                                          x_test=X_test,
                                          y_test=y_test,
                                          mode='test')

    # Log test predictions and evaluate metrics
    wandb.log({'Posterior_dist_on_test': plot_gpr_posterior_plotly(gpr,
                                                                   X_test,
                                                                   y_test,
                                                                   y_test_pred, 
                                                                   y_test_std
                                                                   )})
    
    wandb.log({'Residuals_plot_test': plot_evaluation_metrics(gpr,
                                                              y_test,
                                                              y_test_pred
                                                              )})
    

    return  y_test_pred, y_test_std, gpr




