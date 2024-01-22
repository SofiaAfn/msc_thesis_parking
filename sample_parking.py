import plotly.graph_objects as go
import numpy as np
# import time
import random
import scipy.optimize

days = 20
sigma = 1
day = []
parking_demand = []
for i in range(days+1):
    day.append(i)
    parking_demand.append(0)

# Split into smaller time segments and add some randomness
scatters = []
while day[1] - day[0] > 1/(24*60):
    tmp = []
    tmp_day = []
    for i in range(len(parking_demand) - 1):
        tmp.append(parking_demand[i])
        tmp.append(0.5 * (parking_demand[i]
                          + parking_demand[i+1])
                   + np.random.normal(0, sigma))
        tmp_day.append(day[i])
        tmp_day.append(0.5 * (day[i] + day[i+1]))
    tmp.append(parking_demand[-1])
    tmp_day.append(day[-1])
    parking_demand = tmp
    day = tmp_day
    sigma *= 0.5

day = np.array(day)
parking_demand = np.array(parking_demand)

# Some random function for demand over time
parking_demand += 2   # Some constant demand
# Should add some randomness to the demand as well.
# Probably fractal noise would look neat...
# Some periodic demand change
parking_demand += 7.25 * (np.sin(day * 2 * np.pi) + 1)
# Cant have negative demand...
parking_demand = np.clip(parking_demand, 0, 1e9)

# Lets say that we only have X spots
num_spots = 12
# Cant have more parked than spots. People also come as whole numbers.
parking_used = np.clip(np.round(parking_demand), 0, num_spots)

# Sample observations
num_samples = min([100, len(day)])
sample_ind = sorted(random.sample(range(len(day)), num_samples))
sample_day = day[sample_ind]
sample_used = parking_used[sample_ind]
optimal_sample_used = parking_used[sample_ind]

# Add noise to the observations
# TODO: binomial distribution for missing
#       Some binomial/poisson style distribution for duplicates
#       Some random chance for false positives
for i, sample in enumerate(sample_used):
    offset = random.choice([-2, -1, 0, 1, 2])
    sample_used[i] += offset

# Make an estimate


def predict_demand(estimate, day):
    return estimate[0] * np.ones(day.shape) + \
            estimate[1] * (np.sin(day * 2 * np.pi) + 1)


def predict_used(estimate, day):
    return np.clip(predict_demand(estimate, day), 0, num_spots)


def residual(estimate):
    return np.clip(np.round(sample_used), 0, num_spots) \
           - predict_used(estimate, sample_day)


estimate = scipy.optimize.least_squares(residual, [0, 1]).x


def optimal_residual(estimate):
    return np.clip(np.round(optimal_sample_used), 0, num_spots) \
           - predict_used(estimate, sample_day)


optimal_estimate = scipy.optimize.least_squares(optimal_residual, [0, 1]).x
print(estimate, optimal_estimate)

view_day = np.linspace(day[0], day[-1], int(1e4))
predicted_demand = predict_demand(estimate, view_day)
predicted_usage = predict_used(estimate, view_day)

optimal_predicted_demand = predict_demand(optimal_estimate, view_day)
optimal_predicted_usage = predict_used(optimal_estimate, view_day)

# Create scatters
parking_scatter = [go.Scatter(x=[day[0], day[-1]],
                              y=[num_spots, num_spots],
                              line_color='black', line_width=2,
                              name="Parking spots")]

demand_scatter = [go.Scatter(x=day,
                             y=parking_demand,
                             line_width=4,
                             line_color='Blue',
                             name="Spot demand")]

usage_scatter = [go.Scatter(x=day,
                            y=parking_used,
                            line_width=2,
                            line_color='Red',
                            name="Spot usage")]

lack_parking_scatter = [go.Scatter(x=day, y=np.clip(parking_demand,
                                                    num_spots,
                                                    np.inf),
                                   fill='tonexty',
                                   mode='lines',
                                   opacity=0,
                                   fillcolor='rgba(255,0,0,0.2)',
                                   name='Angry citizens driving around')]

unused_spots_scatter = [go.Scatter(x=day, y=parking_used,
                                   fill='tonexty',
                                   mode='lines',
                                   opacity=0,
                                   fillcolor='rgba(0,0,255,0.2)',
                                   name='Unused space = wasted space')]

optimal_obs_scatter = [go.Scatter(x=sample_day, y=optimal_sample_used,
                                  mode='markers',
                                  marker_line_width=2,
                                  marker_size=8, name="Observation points")]

optimal_demand_estimate_scatter = [
        go.Scatter(x=view_day, y=optimal_predicted_demand,
                   line_color='magenta',
                   line_width=3,
                   name="Estimated demand")]
optimal_usage_estimate_scatter = [
        go.Scatter(x=view_day, y=np.round(optimal_predicted_usage),
                   line_width=3,
                   line_color='lime',
                   name="Estimated usage")]

obs_scatter = [go.Scatter(x=sample_day,
                          y=sample_used,
                          mode='markers',
                          marker_symbol='star',
                          marker_line_width=2,
                          marker_size=8,
                          name="Noisy observation points")]


demand_estimate_scatter = [
        go.Scatter(x=view_day,
                   y=predicted_demand,
                   line_color='blue',
                   line_width=6,
                   name="Noisy measurements -> Estimated demand")]
usage_estimate_scatter = [ 
        go.Scatter(x=view_day,
                   y=np.round(predicted_usage),
                   line_color='red',
                   line_width=6,
                   name="Noisy measurements -> Estimated usage")]

# Set up drawing
layout = go.Layout(title="parking", width=1800, height=1000,
                   yaxis_title="Number of cars",
                   margin=dict(l=80, r=80, t=100, b=99),
                   xaxis_title="Day", xaxis_dtick=1, yaxis_dtick=1,
                   xaxis_range=[0, days], yaxis_range=[-0.1, 18.5],
                   showlegend=True)
layout.update(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

# Demand 
layout.title.text = "Parking spot demand over time"
go.Figure(data=demand_scatter,
          layout=layout).write_image("parking_spot_demand.png")


# Demand + usage 
layout.title.text = "Parking spot demand and usage over time"
go.Figure(data=parking_scatter+demand_scatter+usage_scatter,
          layout=layout).write_image("parking_spot_usage.png")


# Demand > spots 
layout.title.text = "Lack of parking spots"
go.Figure(data=parking_scatter + lack_parking_scatter +
          demand_scatter + usage_scatter,
          layout=layout).write_image("parking_lack_spots.png")


# Demand < spots 
layout.title.text = "Wasted parking spot space over time"
go.Figure(data=parking_scatter + unused_spots_scatter +
          demand_scatter + usage_scatter,
          layout=layout).write_image("parking_unused_spots.png")


# optimal observations + usage
layout.title.text = "Parking usage observations over time"
go.Figure(data=usage_scatter+optimal_obs_scatter,
          layout=layout).write_image("parking_optimal_observations.png")

# Optimal obs
layout.title.text = "Parking observations over time"
go.Figure(data=optimal_obs_scatter,
          layout=layout).write_image("parking_optimal_observations_raw.png")

# Obs
layout.title.text = "Creating Noisy parking observations"
go.Figure(data=obs_scatter + optimal_obs_scatter,
          layout=layout).write_image("prkng_noisy_vs_optimal_observations_raw.png")

# Obs
layout.title.text = "Noisy parking observations"
go.Figure(data=obs_scatter,
          layout=layout).write_image("parking_observations_raw.png")

# optimal predictions + observations + demand + usage
layout.title.text = "Parking spot usage estimate"
go.Figure(data=usage_scatter + optimal_obs_scatter +
          optimal_usage_estimate_scatter,
          layout=layout).write_image("parking_optimal_usage_estimated.png")

# optimal predictions + observations + demand + usage
layout.title.text = "Parking spot demand estimate"
go.Figure(data=demand_scatter + optimal_demand_estimate_scatter,
          layout=layout).write_image("parking_optimal_demand_estimated.png")

# optimal predictions + observations + demand + usage
layout.title.text = "Parking spot noisy vs optimal observations usage estimate"
go.Figure(data=usage_estimate_scatter + optimal_usage_estimate_scatter,
          layout=layout).write_image("parking_noisy_vs_optimal_usage_estimated.png")
 
# optimal predictions + observations + demand + usage
layout.title.text = "Parking spot noisy vs optimal demand estimate"
go.Figure(data=demand_estimate_scatter + optimal_demand_estimate_scatter,
          layout=layout).write_image("prkng_noisy_vs_optimal_demand_estimated.png")
# Observations + demand + usage
# layout.title.text = "Parking usage noisy observations over time"
# go.Figure(data=scatters+obs_scatter,
#        layout=layout).write_image("parking_observations.png")

# noisy + Predictions + observations + demand + usage
# layout.title.text = "Parking spot demand and usage estimates over time"
# go.Figure(data=scatters+estimates,
# layout=layout).write_image("parking_estimated.png")

# Noisy vs optimal estimates
# layout.title.text = "Parking spot demand and usage estimates over time"
# go.Figure(data=optimal_estimates+estimates,
#        layout=layout).write_image("parking_estimated_noisy_vs_optimal.png")

# Layout for the plot
# fig.show()
# fig.save_html("sample_parking.html")
