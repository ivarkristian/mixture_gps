# %%
# The purpose of the following is to demonstrate Gaussian mixture modelling on a 1D dataset.
# The aim is to fit two GPs to model differently weighted subsets of data points with different internal spatial correlations

# %%
# IMPORT LIBRARIES
import gpytorch.constraints
import torch
import gpytorch
import importlib
#from linetimer import CodeTimer
import matplotlib.pyplot as plt
#import seaborn as sns
from plotly.subplots import make_subplots
from copy import deepcopy
import plotly.graph_objects as go
#import plotly.io as pio
import pandas as pd
#import cairosvg
import gpt_classes
import gpt_functions
#import gpt_plot

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6, sci_mode=False, threshold=2000, linewidth=240)

from matplotlib import rc

# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# IMPORT DATA
toy_data = False

if toy_data:
    # 1 - IMPORT TOY DATA
    n = 500
    a = (0.2, 3)
    f = (7, 100)
    m = (2, 4)
    i = [(0.19, 0.26), (0.59, 0.76)]
    df_training = gpt_functions.generate_density_dataframe(n, a, f, m, i)
    n = 500
    a = (0.25, 4)
    f = (6, 107)
    m = (2, 4)
    i = [(0.20, 0.30), (0.65, 0.86)]
    df_validation = gpt_functions.generate_density_dataframe(n, a, f, m, i)
    n = 500
    a = (0.22, 5)
    f = (5, 122)
    m = (2, 4)
    i = [(0.20, 0.28), (0.64, 0.86)]
    df_test = gpt_functions.generate_density_dataframe(n, a, f, m, i)

    ch4_training = torch.Tensor(df_training['CH4 [ppmv]'])
    lon_m_training = torch.Tensor(df_training['Longitude_m']*10)
    df_training['Longitude_m'] = pd.Series(lon_m_training)
    ch4_validation = torch.Tensor(df_validation['CH4 [ppmv]'])
    lon_m_validation = torch.Tensor(df_validation['Longitude_m']*10 + 10)
    df_validation['Longitude_m'] = pd.Series(lon_m_validation)
    ch4_test = torch.Tensor(df_test['CH4 [ppmv]'])
    lon_m_test = torch.Tensor(df_test['Longitude_m']*10 + 20)
    df_test['Longitude_m'] = pd.Series(lon_m_test)

    training_smooth_coords = torch.arange(lon_m_training.min(), lon_m_training.max(), 0.01)
    test_smooth_coords = torch.arange(lon_m_test.min(), lon_m_test.max(), 0.01)

    # Show the data set
    fig_toy = make_subplots(rows=3, cols=1)
    showlegend=True
    fig_toy.add_trace(go.Scatter(x=lon_m_training, y=ch4_training, name='Training', marker_color='black', mode='lines', marker_size=2, showlegend=showlegend), row=1, col=1)
    fig_toy.add_trace(go.Scatter(x=lon_m_validation, y=ch4_validation, name='Validation', marker_color='black', mode='lines', marker_size=2, showlegend=showlegend), row=2, col=1)
    fig_toy.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='Test', marker_color='black', mode='lines', marker_size=2, showlegend=showlegend), row=3, col=1)
    fig_toy.show()
else:
    # 1-IMPORT SAMPLES-CH4
    df = pd.read_csv("../../gp/Barter_Island_methane.tab", delimiter="\t", skiprows=14)
    df = df.groupby(['Latitude', 'Longitude'])['CH4 [ppmv]'].max().reset_index()

    # It seems the data set is merged from smaller bits. The first part up to
    # sample # 2300 seems contiguous, so we use only this.
    df = df[:2300]
    df = df[df['CH4 [ppmv]'] < 150.0]
    # At the latitude of Barter Island (~70 deg north), 0.1 degree longitude corresponds
    # to 6 minutes or 3779 meters. This means that the dataset from
    # -143.700281 to -143.672884 covers ~1035 meters longitudinally.
    # (The latitude difference is from 70.13233082 to 70.13364652, which corresponds to
    # ~146 meters in the north-south direction.)
    latitudes = df['Latitude']
    longitudes = df['Longitude']
    # Sometimes it is better to work in meters. Make column of meters in lon from
    # starting point, which is also the westmost point.
    start = df['Longitude'][0]
    df['Longitude_m'] = (df['Longitude'] - start)*37790 #m/deg

    ch4 = torch.Tensor(df['CH4 [ppmv]'])
    lon_m = torch.Tensor(df['Longitude_m'])

    # Show the data set
    fig_data = make_subplots(rows=1, cols=1)
    showlegend=True
    fig_data.add_trace(go.Scatter(x=lon_m, y=ch4, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
    fig_data.show()

# %%
# Divide the data set into training and test
df_training = df[df['Longitude_m'].between(df['Longitude_m'].min(), 400)].copy()
df_training.reset_index(inplace=True)
ch4_training = torch.Tensor(df_training['CH4 [ppmv]'].to_numpy())
lon_m_training = torch.Tensor(df_training['Longitude_m'].to_numpy())

df_test = df[df['Longitude_m'].between(400, 1035)].copy()
df_test.reset_index(inplace=True)
ch4_test = torch.Tensor(df_test['CH4 [ppmv]'].to_numpy())
lon_m_test = torch.Tensor(df_test['Longitude_m'].to_numpy())

training_smooth_coords = torch.arange(lon_m_training.min(), lon_m_training.max(), 0.1)
test_smooth_coords = torch.arange(lon_m_test.min(), lon_m_test.max(), 0.1)

# Show the data sets
fig_sets = make_subplots(rows=2, cols=1, row_titles=('Training', 'Test'), shared_yaxes=True)
showlegend=True
fig_sets.add_trace(go.Scatter(x=lon_m_training, y=ch4_training, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
showlegend=False
fig_sets.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=2, col=1)
fig_sets['layout']['yaxis1'].update(range=[-3, 40])
fig_sets['layout']['yaxis2'].update(range=[-3, 40])
fig_sets.show()

lon_lat_np = pd.concat([longitudes, latitudes], axis=1).to_numpy()
lon_lat = torch.Tensor(lon_lat_np)

# Then we compute some support parameters to help with identification of plume regions
df['Mean'] = df['CH4 [ppmv]'].rolling(window=5, center=True).mean()
df['Std'] = df['CH4 [ppmv]'].rolling(window=5, center=True).std()
df['Median'] = df['CH4 [ppmv]'].rolling(window=1, center=True).median()

def variable_window_rolling(df, method, column_to_roll, column_for_window, window_value):
    results = []
    for i, row in df.iterrows():
        lower_bound = row[column_for_window] - window_value / 2.0
        upper_bound = row[column_for_window] + window_value / 2.0
        filtered_df = df[(df[column_for_window] >= lower_bound) & (df[column_for_window] <= upper_bound)]
        if method == 'median':
            value = filtered_df[column_to_roll].median()
        elif method == 'mean':
            value = filtered_df[column_to_roll].mean()
        elif method == 'std':
            value = filtered_df[column_to_roll].std()
        else:
            print(f'Error! Unknown method: {method}')
            return
        results.append(value)
    return pd.Series(results)

# Median filter to remove single samples from dominating the mean
df['rolling_median'] = variable_window_rolling(df, 'median', 'CH4 [ppmv]', 'Longitude_m', 1)
df['Std_of_median'] = variable_window_rolling(df, 'std', 'rolling_median', 'Longitude_m', 2)
df['Mean_of_median'] = variable_window_rolling(df, 'mean', 'rolling_median', 'Longitude_m', 2)

mean = torch.Tensor(df['Mean']) # 'Raw' mean
std = torch.Tensor(df['Std'])

median = torch.Tensor(df['rolling_median'])
mean_of_median = torch.Tensor(df['Mean_of_median'])
std_of_median = torch.Tensor(df['Std_of_median'])

# %%
# Set some common parameters, constraints etc.
noise_interval=gpytorch.constraints.Interval(0.0001, 1)
length_constraint=[gpytorch.constraints.GreaterThan(1.0), gpytorch.constraints.GreaterThan(1.0)]
type = 'matern' # kernel type

# %%
# Demonstrate training with early stopping on a single kernel
llh = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
mdl = gpt_classes.ExactGPModel(lon_m_training, ch4_training, llh, type, lengthscale_constraint=length_constraint[0])
mdl.mean_module.constant.data.fill_(ch4_training.min().item())
iter = 100
gpt_functions.train_model(lon_m_training, ch4_training, mdl, iter=iter, early_delta=(False, 'mll', None, None, 10), debug=False)
fig_train = make_subplots(rows=1, cols=1)
ls = []
os = []
for i, l in enumerate(mdl.lengthscales[:mdl.curr_trained]):
    ls.append(l[0].item())
    os.append(mdl.outputscales[i])

fig_train.update_layout(title_text='Single kernel MLL loss', title_x=0.5)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=mdl.losses[:mdl.curr_trained], name='Training'), row=1, col=1)
fig_train.add_trace(go.Scatter(x=mdl.val_iter[:mdl.saved_val_losses], y=mdl.val_losses[:mdl.saved_val_losses], name='Validation'), row=1, col=1)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=ls, name='Lengthscale'), row=1, col=1)
#fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=os, name='Outputscale'), row=1, col=1)
fig_train.show()
mdl.print_named_parameters()

# %%
# Plot predictions after early stopping
mdl.eval()
mdl.likelihood.eval()
xs = [lon_m_training, lon_m_test]
ys = [ch4_training, ch4_test]
s_fig = make_subplots(rows=1, cols=1)

showlegend=True
for i, item in enumerate(xs[:1]):
    mdl.set_train_data(item, ys[i], strict=False)
    x_coords = torch.arange(item.min(), item.max(), 0.01)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        single_pred = mdl.likelihood(mdl(x_coords))
        single_lower, single_upper = single_pred.confidence_region()

    s_fig.add_trace(go.Scatter(x=item, y=ys[i], name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=i+1, col=1)
    s_fig.add_trace(go.Scatter(x=x_coords, y=single_pred.mean, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=i+1, col=1)
    s_fig.add_trace(go.Scatter(x=x_coords, y=single_upper, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=i+1, col=1)
    showlegend=False
    s_fig.add_trace(go.Scatter(x=x_coords, y=single_lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=i+1, col=1) # fill to previous trace
    s_fig.update_layout(title_text='Single kernel predictions', title_x=0.5)
s_fig.show()


# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Mixture models init
num_models = 2
type = type
noise_interval = noise_interval
length_constraint = [gpytorch.constraints.GreaterThan(1.0), gpytorch.constraints.GreaterThan(1.0)]

mixture_likelihoods = []
mixture_models = []
for i in range(num_models):
    mixture_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
    mixture_likelihoods.append(mixture_likelihood)
    mixture_model = gpt_classes.ExactGPModel(lon_m_training, ch4_training, mixture_likelihoods[i], type, lengthscale_constraint=length_constraint[i])
    mixture_models.append(mixture_model)

# %%
# Train mixture models with EM
training_iter = 26
inner_iter = 5
eval_mode = 'mll'
ret = gpt_functions.EM_algorithm(mixture_models, lon_m_training, ch4_training, iter=training_iter, inner_iter=inner_iter, init='topk', visualization_coords=training_smooth_coords, early_delta=(False, eval_mode, None, None, 0))
mixture_models, train_responsibilities, figs, val_figs = ret

for i, model in enumerate(mixture_models):
    if i%5 == 0:
        print(f'model {i}:')
        model.print_named_parameters()
        print('')

# %%
# Plot training result of mixture models
fig_mix_train = make_subplots(rows=1, cols=1)

for i, model in enumerate(mixture_models):
    iter_offset = 0#i*init_iter
    x_l = []
    x_o = []
    for j, n in enumerate(model.lengthscales[:model.curr_trained]):
        x_l.append(n[0])
        x_o.append(model.outputscales[j])

    fig_mix_train.add_trace(go.Scatter(x=torch.arange(mixture_models[i].curr_trained), y=x_l, mode='lines', name='length_model %d' % (i)), row=1, col=1)
    #fig_mix_train.add_trace(go.Scatter(x=torch.arange(mixture_models[i].curr_trained), y=x_o, mode='lines', name='outputscale_model %d' % (i)), row=1, col=1)

text = 'Mixture GPs ' + eval_mode + ' loss'
fig_mix_train.update_layout(title_text=text, title_x=0.5)
fig_mix_train.show()


# %%
# Show combined figures
# Number of figures
figures = [figs[1], figs[11], figs[21]]
num_figures = len(figures)

# Each figure has 2 subplots (1 row, 2 columns), so total subplots in the combined figure
total_subplots = num_figures * 2

# Create a new figure with `num_figures` rows and 2 columns
combined_fig, axs = plt.subplots(num_figures, 2, figsize=(12, 4 * num_figures))

# Flatten the axs array if num_figures > 1
if num_figures > 1:
    axs = axs.reshape(num_figures, 2)
else:
    axs = [axs]  # If there is only one figure, make sure it's in list form

# Loop over the figures and extract subplots content
for i, fig in enumerate(figures):
    # Get all axes (subplots) from the original figure
    for j, ax_old in enumerate(fig.axes):
        # Get the corresponding axis in the new combined figure
        ax_new = axs[i, j]
        
        # Extract and re-plot all elements in the subplot
        # Fill_between (shaded regions)
        if len(ax_old.collections) > 0:
            for poly in ax_old.collections:
                # Check if it's a filled region
                if hasattr(poly, "get_paths"):
                    paths = poly.get_paths()
                    if len(paths) > 0:
                        verts = paths[0].vertices  # Vertices of filled regions
                        x_data = verts[:, 0]
                        y_data_lower = verts[:, 1]
                        ax_new.fill_between(x_data, y_data_lower, color='lightgrey', label='2 sigma')
        
        # Plot CH4 predicted
        for line in ax_old.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            label = line.get_label()
            color = line.get_color()
            ax_new.plot(x_data, y_data, color=color, label=label)
        
        # Scatter plots (CH4 observations and active observations)
        for path in ax_old.collections:
            offsets = path.get_offsets()  # X and Y data for scatter points
            x_data, y_data = offsets[:, 0], offsets[:, 1]
            label = path.get_label()
            color = path.get_facecolor()[0]  # Scatter point color
            
            # Handle case where there are no sizes
            sizes = path.get_sizes()
            if len(sizes) > 0:
                size = sizes[0]  # Use provided size
            else:
                size = 20  # Default size if sizes are not specified
            ax_new.scatter(x_data, y_data, color=color, s=size, label=label)

        # Set title, xlabel, and ylabel
        if j%2:
            ha = 'right'
        else:
            ha = 'left'
            
        ax_new.set_title(ax_old.get_title(), weight='bold', loc=ha)
        ax_new.set_xlabel('Flight distance from west to east [m]')
        ax_new.set_ylabel('CH$_4$ [ppm]')
        
        # Copy the legend if present
        if ax_old.get_legend():
            ax_new.legend(loc='best')
        
    # Extract and plot suptitle from the original figure over the corresponding row
    suptitle = fig._suptitle
    if suptitle:
        suptitle_text = suptitle.get_text()
        combined_fig.text(0.5, ((num_figures - i) / num_figures) - 0.11 + 0.05*i, suptitle_text, ha='center', fontsize=12, weight='bold')

plt.subplots_adjust(hspace=0.4, wspace=0.18)  # Adjust hspace to add more vertical space between rows
combined_fig.savefig('barter_island_moe_training.eps', format='eps', bbox_inches='tight', dpi=300)
# Show the combined figure
plt.show()

# %%
# Predict test set sequentially with one model at a time
means = []; lowers = []; uppers = []
for model in mixture_models:
    model.set_train_data(lon_m_test, ch4_test, strict=False)
    model.eval()
    likelihood = model.likelihood
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_smooth_coords))
        means.append(observed_pred.mean)
        lower, upper = observed_pred.confidence_region()
        lowers.append(lower); uppers.append(upper)

# %%
# Plot sequentially
fig = make_subplots(rows=1, cols=len(means), shared_yaxes=True)
showlegend=True
for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
    fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=test_smooth_coords, y=mean, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter(x=test_smooth_coords, y=upper, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=i+1)
    showlegend=False
    fig.add_trace(go.Scatter(x=test_smooth_coords, y=lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=i+1) # fill to previous trace

fig_text = f'Test data predictions. (Train_iter: {training_iter} - inner_iter: {inner_iter})'
fig.update_layout(title={'text': fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# %%
# Train gating function
gating_type = ['rbf', 'rbf']
gating_train_iter = [60, 60]

test_log_probs, _ = gpt_functions.compute_log_probs(mixture_models, lon_m_test, ch4_test)
test_responsibilities = torch.exp(gpt_functions.compute_responsibilities(test_log_probs, normalize=True))

gating_models = []
for i, gt in enumerate(gating_type):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0))
    gating_model = gpt_classes.ExactGPModel(lon_m_test, test_responsibilities[:, i], likelihood, gt)
    gpt_functions.train_model(lon_m_test, test_responsibilities[:, 0], gating_model, iter=gating_train_iter[i], lr=0.1, early_delta=(False, '', 0, 0, 0), debug=False)

    if gating_type == 'scale_rbf':
        print(f'Outputscale: {gating_model.covar_module.outputscale.item()}')
        print(f'Lengthscale: {gating_model.covar_module.base_kernel.lengthscale.item()}')
    elif gating_type == 'matern':
        print(f'Lengthscale: {gating_model.covar_module.lengthscale.item()}')

    gating_models.append(gating_model)

for gt in gating_models:
    x_l = []
    for n in gt.lengthscales[:gt.curr_trained]:
        x_l.append(n[0])

    plt.plot(gt.losses[:gt.curr_trained])
    gt.print_named_parameters()

# %%
# PLOT correlation distance
# Create a range of distances
r_max = 10  # Adjust based on the scale
r = torch.linspace(0, r_max, 500)  # 500 points from 0 to r_max

# Reference point
x0 = torch.tensor([[0.0]])
x_r = r.unsqueeze(1)

# Compute covariance values
plt.figure(figsize=(10, 6))
cov = []
for i, model in enumerate(gating_models):
    cov.append(model.covar_module(x0, x_r).evaluate().squeeze().detach().numpy())
    plt.plot(r.numpy(), cov[i], label=f"{gating_type[i]} kernel")

plt.xlabel('Distance')
plt.ylabel('Covariance')
plt.title('Covariance Function vs. Distance')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Predict gating function
gating_means = []; gating_lowers = []; gating_uppers = []
for gating_model in gating_models:
    gating_model.eval()
    likelihood = gating_model.likelihood
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gating_model(test_smooth_coords))
        gating_means.append(observed_pred.mean)
        gating_lower, gating_upper = observed_pred.confidence_region()
        gating_lowers.append(gating_lower); gating_uppers.append(gating_upper)

gating_fig = make_subplots(rows=1, cols=len(gating_models), shared_yaxes=True)
showlegend=True
for i, gating_mean in enumerate(gating_means):
    gating_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    gating_fig.add_trace(go.Scatter(x=lon_m_test[test_responsibilities[:, i]>=0.5], y=ch4_test[test_responsibilities[:, i]>=0.5], name='Responsibility > 0.5', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    gating_fig.add_trace(go.Scatter(x=test_smooth_coords, y=gating_mean, name='Gating', marker_color='purple', showlegend=showlegend), row=1, col=i+1)
    showlegend=False
    
gating_fig.show()

# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Predict mixture of models
observed_pred = gpt_functions.predict_mixture(test_smooth_coords, mixture_models, gating_means)
(mean_moe, lower_moe, upper_moe, p_of_z) = observed_pred

pz_fig = make_subplots(rows=1, cols=len(gating_means), shared_yaxes=True)
showlegend=True
for i, gating_mean in enumerate(gating_means):
    pz_fig.add_trace(go.Scatter(x=lon_m_test, y=test_responsibilities[:, i], name='Responsibilities', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    #pz_fig.add_trace(go.Scatter(x=lon_m_test, y=test_log_probs[:, i], name='Log_probs', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    pz_fig.add_trace(go.Scatter(x=test_smooth_coords, y=p_of_z[:, i], name='p(z)', marker_color='green', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    pz_fig.add_trace(go.Scatter(x=test_smooth_coords, y=gating_mean, name='Gating', marker_color='purple', showlegend=showlegend), row=1, col=i+1)
    showlegend=False
pz_fig.show()

# %%
# PLOT
fig, axs = plt.subplots(1, 1, figsize=(6, 6))

showlegend = True
axs.scatter(lon_m_test, test_responsibilities[:, 0], color='black', s=2, label='Observation' if showlegend else "")
axs.plot(
    test_smooth_coords,
    p_of_z[:, 0],
    color='green',
    linewidth=1,
    label='p(z$_{0}$) estimated' if showlegend else ""
)

axs.set_xlabel('Flight distance from west to east [m]', fontsize=12)

if showlegend:
    axs.legend(loc='upper right', fontsize=12)
    showlegend = False

# Set y-axis and x-axis limits
axs.set_ylim([-0.02, 1.02])
axs.set_xlim([713, 777])

axs.tick_params(axis='both', labelsize=12)

# Set a common y-axis label
axs.set_ylabel('p(z$_{0}$)', fontsize=12)
fig.suptitle('Estimated probability of anomaly model', fontsize=14, weight='bold')
# Adjust layout to avoid overlap
plt.tight_layout()
fig.savefig('barter_p_of_z.eps', format='eps', bbox_inches='tight', dpi=300)

plt.show()

# %%
# Plot mixture prediction
fig = make_subplots(rows=1, cols=1)
showlegend=True

fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
fig.add_trace(go.Scatter(x=test_smooth_coords, y=mean_moe, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=1, col=1)
fig.add_trace(go.Scatter(x=test_smooth_coords, y=upper_moe, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=1)
showlegend=False
fig.add_trace(go.Scatter(x=test_smooth_coords, y=lower_moe, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=1) # fill to previous trace

fig_text = f'Mixture prediction. Train_iter: {training_iter} - inner_iter: {inner_iter}'
fig.update_layout(title={'text': fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# %%
# Init, train and predict with single_model
llh = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
single_length_constraint = length_constraint[0]
mdl = gpt_classes.ExactGPModel(lon_m_training, ch4_training, llh, type, lengthscale_constraint=single_length_constraint)

iter = training_iter*inner_iter
gpt_functions.train_model(lon_m_training, ch4_training, mdl, iter=iter, early_delta=(False, 'mll', None, None, 0), debug=False)
fig_train = make_subplots(rows=1, cols=1)
fig_train.update_layout(title_text='Single kernel MLL loss', title_x=0.5)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=mdl.losses[:mdl.curr_trained], name='Training'), row=1, col=1)

fig_train.show()
mdl.print_named_parameters()

mdl.eval()
llh.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mdl.set_train_data(lon_m_test, ch4_test, strict=False)
    single_pred = llh(mdl(test_smooth_coords))
    single_lower, single_upper = single_pred.confidence_region()

s_fig = make_subplots(rows=1, cols=1)
showlegend=True
s_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
s_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_pred.mean, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=1, col=1)
s_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_upper, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=1)
showlegend=False
s_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=1) # fill to previous trace
s_fig.show()

# %%
# Plot single and mixture together
t_fig = make_subplots(rows=2, cols=2, shared_yaxes=True, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4"))
showlegend=True
t_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_pred.mean, name='CH4 pred. sgl', marker_color='crimson', showlegend=showlegend), row=1, col=1)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_upper, fill='none', marker_color='pink', name='2 sigma', showlegend=showlegend), row=1, col=1)
showlegend=False
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='pink', showlegend=showlegend), row=1, col=1) # fill to previous trace
text11 = f'Single prediction'

showlegend=True
t_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=mean_moe, name='CH4 pred. mix', marker_color='dodgerblue', showlegend=showlegend), row=1, col=2)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=upper_moe, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=2)
showlegend=False
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=lower_moe, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=2) # fill to previous trace
text12 = f'Mixture prediction'

t_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=2, col=1)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_pred.mean, name='CH4 pred. sgl', marker_color='crimson', showlegend=showlegend), row=2, col=1)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=mean_moe, name='CH4 pred. mix', marker_color='dodgerblue', showlegend=showlegend), row=2, col=1)
text21 = f'Prediction means'

t_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=2, col=2)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=upper_moe, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=2, col=2)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=lower_moe, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=2, col=2) # fill to previous trace
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_upper, fill='none', marker_color='pink', name='2 sigma', showlegend=showlegend), row=2, col=2)
t_fig.add_trace(go.Scatter(x=test_smooth_coords, y=single_lower, fill='none', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='pink', showlegend=showlegend), row=2, col=2) # fill to previous trace
text22 = f'Prediction variances'

t_fig_text = f'Comparison. (Iter: {iter})'
t_fig.update_layout(title={'text': t_fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
t_fig.layout.annotations[0].update(text=text11)
t_fig.layout.annotations[1].update(text=text12)
t_fig.layout.annotations[2].update(text=text21)
t_fig.layout.annotations[3].update(text=text22)
t_fig.show()


# %%
############ NLE METHODS ############

# %%
# NLE detect subsets
min_cluster_dist = 4
threshold = 4.0
regions_training_indices, anomaly, no_anomaly = gpt_functions.detect_clusters(lon_m_training, ch4_training, threshold=threshold, min_cluster_distance=min_cluster_dist, debug=False)
regions_training = gpt_functions.get_region_polygons(regions_training_indices, lon_m_training)
lon_m_training_subsets = [lon_m_training[anomaly], lon_m_training[no_anomaly]]
ch4_training_subsets = [ch4_training[anomaly], ch4_training[no_anomaly]]
regions_training_tensors = gpt_functions.polygons_to_1d_tensors(regions_training)

regions_test_indices, anomaly, no_anomaly = gpt_functions.detect_clusters(lon_m_test, ch4_test, threshold=threshold, min_cluster_distance=min_cluster_dist, debug=False)
regions_test = gpt_functions.get_region_polygons(regions_test_indices, lon_m_test)
lon_m_test_subsets = [lon_m_test[anomaly], lon_m_test[no_anomaly]]
ch4_test_subsets = [ch4_test[anomaly], ch4_test[no_anomaly]]
regions_test_tensors = gpt_functions.polygons_to_1d_tensors(regions_test)

fig_subsets = make_subplots(rows=2, cols=1, row_titles=('Training', 'Test'), x_title='Distance from west to east [m]', y_title='CH$_4$ [ppmv]')
xs = [lon_m_training_subsets, lon_m_test_subsets]
ys = [ch4_training_subsets, ch4_test_subsets]
regions = [regions_training_tensors, regions_test_tensors]

showlegend=True
for i, region in enumerate(regions):
    fig_subsets.add_trace(go.Scatter(x=xs[i][0], y=ys[i][0], marker_color='orange', mode='markers', marker_size=2, name='Anomaly', showlegend=showlegend), row=i+1, col=1)
    fig_subsets.add_trace(go.Scatter(x=xs[i][1], y=ys[i][1], marker_color='black', mode='markers', marker_size=2, name='Background', showlegend=showlegend), row=i+1, col=1)
    for sub_region in region:
        fig_subsets.add_trace(go.Scatter(x=sub_region, y=[0]*len(sub_region), marker_color='red', mode='lines', name='Anomaly region', showlegend=showlegend), row=i+1, col=1)
        showlegend=False

fig_subsets['layout']['yaxis1'].update(range=[-3, 40])
fig_subsets['layout']['yaxis2'].update(range=[-3, 40])

fig_subsets_text = f'Barter Island measurements, anomaly regions detected for NLE'
fig_subsets.update_layout(title={'text': fig_subsets_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig_subsets.show()

# %%
# Create the figure and axes for the 2 subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 3))

# Data for the plots
xs = [lon_m_training_subsets, lon_m_test_subsets]
ys = [ch4_training_subsets, ch4_test_subsets]
regions = [regions_training_tensors, regions_test_tensors]

# Titles for each row
row_titles = ['Training', 'Test']

# Plot settings
showlegend = True
show_anomaly_region = True

# Loop through the data and plot
for i, ax in enumerate(axs):
    # Scatter plots for anomaly and background
    ax.scatter(xs[i][0], ys[i][0], color='orange', s=1, label='Anomaly' if showlegend else "")
    ax.scatter(xs[i][1], ys[i][1], color='black', s=1, label='Background' if showlegend else "")
    
    # Plot anomaly regions as red lines
    for sub_region in regions[i]:
        ax.plot(sub_region, [0] * len(sub_region), color='red', label='Anomaly region' if show_anomaly_region else "")
        show_anomaly_region = False
    
    # Set y-axis range
    ax.set_ylim([-1, 40])
    #ax.set_title(row_titles[i])
    ax.set_xlabel('Flight distance from west to east [m]')
    ax.set_ylabel('CH$_4$ [ppm]')

    # Only show legend for the first subplot
    if showlegend:
        ax.legend(loc='upper right', fontsize=8)
        showlegend = False

# Add vertical row titles to the right of each subplot
fig.text(0.91, 0.73, 'Training', va='center', rotation='vertical', fontsize=11)
fig.text(0.91, 0.25, 'Test', va='center', rotation='vertical', fontsize=11)

# Set the title for the entire figure
fig.suptitle('Barter Island observations, anomaly regions detected for NLE', y=0.96, x=0.50, ha='center')

# Adjust layout
plt.subplots_adjust(hspace=0.6)
fig.savefig('barter_island_data.eps', format='eps', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# %%
# NLE models init
num_nle_models = 2
type = type
noise_interval = noise_interval
nle_length_constraint = length_constraint[0]

nle_likelihoods = []
nle_models = []
for i in range(num_nle_models):
    nle_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
    nle_likelihoods.append(nle_likelihood)
    nle_model = gpt_classes.ExactGPModel(lon_m_training_subsets[i], ch4_training_subsets[i], nle_likelihoods[i], type, lengthscale_constraint=nle_length_constraint)
    nle_models.append(nle_model)

# %%
# NLE train on subsets
fig_train = make_subplots(rows=2, cols=1)
fig_train.update_layout(title_text='NLE training MLL loss', title_x=0.5)    

iter = training_iter*inner_iter*6
e_delta = [False, False]
for i, model in enumerate(nle_models):
    gpt_functions.train_model(lon_m_training_subsets[i], ch4_training_subsets[i], model, iter=iter, early_delta=(e_delta[i], 'mll', None, None, 0), debug=False)
    fig_train.add_trace(go.Scatter(x=model.iter[:model.curr_trained], y=model.losses[:model.curr_trained], name='Training m%d' % (i)), row=1, col=1)
    #fig_train.add_trace(go.Scatter(x=model.val_iter[:model.saved_val_losses], y=model.val_losses[:model.saved_val_losses], name='Validation m%d' % (i)), row=1, col=1)
    model.print_named_parameters()

for i, model in enumerate(nle_models):
    iter_offset = 0#i*init_iter
    x_l = []
    for n in model.lengthscales[:model.curr_trained]:
        x_l.append(n[0])

    fig_train.add_trace(go.Scatter(x=torch.arange(nle_models[i].curr_trained), y=x_l, mode='lines', name='length_model %d' % (i)), row=2, col=1)

fig_train.show()

# %%
# NLE predict test set
print(f'-- Predicting {len(test_smooth_coords)} locations --')
mean_nle = torch.zeros(len(test_smooth_coords))
sigmas_lower_nle = torch.zeros(len(test_smooth_coords))
sigmas_upper_nle = torch.zeros(len(test_smooth_coords))

preds = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    model.set_train_data(lon_m_test_subsets[i], ch4_test_subsets[i], strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(test_smooth_coords)) # both models predicts entire domain
    preds.append(pred)

fig_nle_preds = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True)
fig_nle_preds.update_layout(title_text='NLE sequential predictions', title_x=0.5)    
for i, pred in enumerate(preds):
    showlegend=True
    fig_nle_preds.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=i+1, col=1)
    fig_nle_preds.add_trace(go.Scatter(x=test_smooth_coords, y=pred.mean, name='CH4 pred', marker_color='dodgerblue'), row=i+1, col=1)
    fig_nle_preds.add_trace(go.Scatter(x=test_smooth_coords, y=pred.confidence_region()[1], fill='none', marker_color='grey', name='2 stddev NLE', showlegend=showlegend), row=i+1, col=1)
    showlegend=False
    fig_nle_preds.add_trace(go.Scatter(x=test_smooth_coords, y=pred.confidence_region()[0], fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='grey', showlegend=showlegend), row=i+1, col=1) # fill to previous trace

fig_nle_preds.show()

# Then pick prediction model based on if location is within anomaly region or not
mask = torch.zeros(len(test_smooth_coords))
coords_2d = gpt_functions.oneD_to_twoD_coords(test_smooth_coords)

for polygon in regions_test:
    mask = mask + polygon.contains_points(coords_2d.numpy()).astype(int)
mask = (mask > 0)
mask_int = mask.int()
mask_inv = mask.__invert__().int()
mean_nle = mean_nle + preds[0].mean*mask_int
mean_nle = mean_nle + preds[1].mean*mask_inv
sigmas_lower_nle = sigmas_lower_nle + preds[0].confidence_region()[0]*mask_int
sigmas_lower_nle = sigmas_lower_nle + preds[1].confidence_region()[0]*mask_inv
sigmas_upper_nle = sigmas_upper_nle + preds[0].confidence_region()[1]*mask_int
sigmas_upper_nle = sigmas_upper_nle + preds[1].confidence_region()[1]*mask_inv

nle_fig = make_subplots(rows=1, cols=1)
showlegend=True
nle_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter(x=test_smooth_coords, y=mean_nle, name='CH4 pred. NLE', marker_color='crimson', showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter(x=test_smooth_coords, y=sigmas_upper_nle, fill='none', marker_color='pink', name='2 stddev NLE', showlegend=showlegend), row=1, col=1)
showlegend=False
nle_fig.add_trace(go.Scatter(x=test_smooth_coords, y=sigmas_lower_nle, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='pink', showlegend=showlegend), row=1, col=1) # fill to previous trace
nle_fig.update_layout(title_text='Predict test region', title_x=0.5)
nle_fig.show()


# %%
# Plot training result of mixture models
fig_mix_train, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
m = ['Anomaly', 'Background']
showlegend = True

linestyles = ['-', '--']
for i, model in enumerate(mixture_models):
    x_l = [n[0] for n in model.lengthscales[:model.curr_trained]]
    
    axs[0].plot(x_l, label='Lx %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)
    
for i, model in enumerate(nle_models):
    x_l = [n[0] for n in model.lengthscales[:model.curr_trained]]
    
    axs[1].plot(x_l, label='Lx %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)
    
# Setting the axis limits
axs[0].set_xlim((0, 130))
axs[0].set_ylim((0, 14))
axs[1].set_xlim((0, 780))
axs[1].set_ylim((0, 14))

# Adding labels and title with improved formatting
axs[0].set_xlabel('Iterations', fontsize=12)
axs[1].set_xlabel('Iterations', fontsize=12)
axs[0].set_ylabel('Lengthscale [m]', fontsize=12)
axs[1].set_ylabel('Lengthscale [m]', fontsize=12)

axs[0].set_title('MoE', fontsize=12)
axs[1].set_title('NLE', fontsize=12)

# Adding legend
axs[0].legend(fontsize=10, loc='upper left')
axs[1].legend(fontsize=10, loc='upper left')

# Adding grid lines
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Styling
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

fig_mix_train.subplots_adjust(wspace=0.3) 
fig_mix_train.suptitle('Lengthscale (Lx) training progression', y=1.0, fontsize=14, weight='bold')

# Save the plot
fig_mix_train.savefig('barter_mixture_gps_lengthscales.eps', format='eps', dpi=300)


# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# NLE-BMA prediction
preds = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        pred = model.likelihood(model(lon_m_test))
        preds.append(pred) 

nle_fig = make_subplots(rows=2, cols=1)
showlegend=True
for i, pred in enumerate(preds):
    lower, upper = pred.confidence_region()
    nle_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=i+1, col=1)
    nle_fig.add_trace(go.Scatter(x=lon_m_test, y=pred.mean, name='Pred. model 0', marker_color='crimson', showlegend=showlegend), row=i+1, col=1)
    nle_fig.add_trace(go.Scatter(x=lon_m_test, y=upper, fill='none', marker_color='pink', name='2 stddev NLE', showlegend=showlegend), row=i+1, col=1)
    showlegend=False
    nle_fig.add_trace(go.Scatter(x=lon_m_test, y=lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='pink', showlegend=showlegend), row=i+1, col=1) # fill to previous trace
nle_fig.show()

# %%
# Make predictions of the test area using the test observations
preds = []
coords = lon_m_test # test_smooth_coords
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()

    model.set_train_data(lon_m_test, ch4_test, strict=False)
    with torch.no_grad():
        pred = model.likelihood(model(coords))
        preds.append(pred) 

nle_fig = make_subplots(rows=2, cols=1)
showlegend=True
for i, pred in enumerate(preds):
    lower, upper = pred.confidence_region()
    nle_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=i+1, col=1)
    nle_fig.add_trace(go.Scatter(x=coords, y=pred.mean, name='Pred. model 0', marker_color='crimson', showlegend=showlegend), row=i+1, col=1)
    nle_fig.add_trace(go.Scatter(x=coords, y=upper, fill='none', marker_color='pink', name='2 stddev NLE', showlegend=showlegend), row=i+1, col=1)
    showlegend=False
    nle_fig.add_trace(go.Scatter(x=coords, y=lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='pink', showlegend=showlegend), row=i+1, col=1) # fill to previous trace
nle_fig.show()

df_test['preds.mean_m0'] = preds[0].mean
df_test['preds.mean_m1'] = preds[1].mean
df_test['preds.stddev_m0'] = preds[0].stddev
df_test['preds.stddev_m1'] = preds[1].stddev

# %%
# Compute rolling windows
df = df_test
window_value = 4
column_for_window = 'Longitude_m'
column_to_roll = 'CH4 [ppmv]'
scores = []
log_probs = torch.full((len(df), 2), 0.0)
log_probs_mll = torch.full((len(df), 2), 0.0)
logdetK = torch.full((len(df), 2), 0.0)
ymK = torch.full((len(df), 2), 0.0)
weight = torch.full((len(df), 2), 0.0)

for i, row in df.iterrows():
    lower_bound = row[column_for_window] - window_value / 2.0
    upper_bound = row[column_for_window] + window_value / 2.0
    filtered_df = df[(df[column_for_window] >= lower_bound) & (df[column_for_window] <= upper_bound)]
    
    # Using a data window of length n:
    lon_m = torch.Tensor(filtered_df[column_for_window].values)
    ch4 = torch.Tensor(filtered_df[column_to_roll].values)
    diff0 = (filtered_df['CH4 [ppmv]'] - filtered_df['preds.mean_m0']).mean().__abs__()
    weight[i, 0] = -diff0 * filtered_df['preds.stddev_m0'].mean()
    diff1 = (filtered_df['CH4 [ppmv]'] - filtered_df['preds.mean_m1']).mean().__abs__()
    weight[i, 1] = -diff1 * filtered_df['preds.stddev_m1'].mean()

    log_pr, pred_output = gpt_functions.compute_log_probs(nle_models, lon_m, ch4)

    log_probs[i] = log_pr.sum(dim=0).div(len(log_pr))
    
    K0 = pred_output[0].covariance_matrix
    K1 = pred_output[1].covariance_matrix
    m0 = pred_output[0].mean
    m1 = pred_output[1].mean

    ch4_here = row['CH4 [ppmv]']
    logdetK[i, 0] = -0.5*torch.logdet(K0)
    logdetK[i, 1] = -0.5*torch.logdet(K1)
    ymK[i, 0] = -0.5*(ch4_here - m0).matmul(torch.linalg.inv(K0).matmul((ch4_here - m0).unsqueeze(1)))
    ymK[i, 1] = -0.5*(ch4_here - m1).matmul(torch.linalg.inv(K1).matmul((ch4_here - m1).unsqueeze(1)))

df_test['weights_m0'] = torch.exp(weight[:, 0])
df_test['weights_m1'] = torch.exp(weight[:, 1])

df_test['weights_norm_m0'] = df_test['weights_m0']/(df_test['weights_m0'] + df_test['weights_m1'])
df_test['weights_norm_m1'] = df_test['weights_m1']/(df_test['weights_m0'] + df_test['weights_m1'])

nle_fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
showlegend=True

nle_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter(x=lon_m_test, y=df_test['weights_norm_m0'], name='weights_norm_m0', showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter(x=lon_m_test, y=df_test['weights_norm_m1'], name='weights_norm_m1', showlegend=showlegend), row=1, col=1)

nle_fig.add_trace(go.Scatter(x=lon_m_test, y=ch4_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=2, col=1)
nle_fig.add_trace(go.Scatter(x=lon_m_test, y=torch.exp(weight[:, 0]), name='weight_0', showlegend=showlegend), row=2, col=1)
nle_fig.add_trace(go.Scatter(x=lon_m_test, y=torch.exp(weight[:, 1]), name='weight_1', showlegend=showlegend), row=2, col=1)
nle_fig.show()

# %%
# Test the weights
preds = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds.append(model.likelihood(model(test_smooth_coords)))

p_m0 = torch.zeros_like(test_smooth_coords)
p_m1 = torch.zeros_like(test_smooth_coords)
mean_bma = torch.zeros_like(test_smooth_coords)
var_bma = torch.zeros_like(test_smooth_coords)

for i, pos in enumerate(test_smooth_coords.numpy()):
    df_closest = df_test.iloc[(df_test['Longitude_m']-pos).abs().argmin()]
    p_m0[i] = df_closest['weights_norm_m0']
    p_m1[i] = df_closest['weights_norm_m1']

    mean_bma[i] = p_m0[i] * preds[0].mean[i] + p_m1[i] * preds[1].mean[i]
    var_bma[i] = p_m0[i] * (preds[0].mean[i]**2 + preds[0].variance[i]) + p_m1[i] * (preds[1].mean[i]**2 + preds[1].variance[i]) - mean_bma[i]**2

sigmas_upper_bma = mean_bma + 2*torch.sqrt(var_bma) 
sigmas_lower_bma = mean_bma - 2*torch.sqrt(var_bma)


# %%
# Create the figure and axes for the 4 subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True, sharey=True)
ylabel = 'CH$_4$ [ppm]'

# MoE plot (row 1)
axs[1].plot(test_smooth_coords, mean_moe, color='dodgerblue', label='CH$_4$ pred.')
axs[1].fill_between(test_smooth_coords, lower_moe, upper_moe, color='lightgrey', label='2 std. dev.')
axs[1].scatter(lon_m_test, ch4_test, color='black', s=2, label='CH$_4$ obs.')
axs[1].set_title('MoE')
axs[1].set_ylabel(ylabel)

# NLE plot (row 2)
axs[2].plot(test_smooth_coords, mean_nle, color='dodgerblue', label='CH$_4$ pred. NLE')
axs[2].fill_between(test_smooth_coords, sigmas_lower_nle, sigmas_upper_nle, color='lightgrey', label='2 stddev NLE')
axs[2].scatter(lon_m_test, ch4_test, color='black', s=2)
axs[2].set_title('NLE')
axs[2].set_ylabel(ylabel)

# NLE-BMA plot (row 3)
axs[3].plot(test_smooth_coords, mean_bma, color='dodgerblue', label='CH$_4$ pred. NLE-MA')
axs[3].fill_between(test_smooth_coords, sigmas_lower_bma, sigmas_upper_bma, color='lightgrey', label='2 stddev NLE-MA')
axs[3].scatter(lon_m_test, ch4_test, color='black', s=2)
axs[3].set_title('NLE-MA')
axs[3].set_ylabel(ylabel)

# Single kernel plot (row 0)
axs[0].plot(test_smooth_coords, single_pred.mean, color='dodgerblue', label='CH$_4$ pred. single')
axs[0].fill_between(test_smooth_coords, single_lower, single_upper, color='lightgrey', label='2 stddev single')
axs[0].scatter(lon_m_test, ch4_test, color='black', s=2)
axs[0].set_title('Single kernel')
axs[0].set_ylabel(ylabel)

# Set common labels
fig.text(0.5, 0.04, 'Flight distance from west to east [m]', ha='center')

# Collect handles and labels for a common legend
handles, labels = axs[1].get_legend_handles_labels()

# Create a common legend outside the plot
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.77, 0.93))

# Adjust the layout
plt.subplots_adjust(hspace=0.4)

# Set y-axis and x-axis limits
for ax in axs:
    ax.set_ylim([-2, 28])
    ax.set_xlim([733, 757])

# Set the title for the entire figure
fig.suptitle('Comparison of predictions', y=0.96, weight='bold')

# Save the figure
plt.savefig('barter_island_comparison_prediction.eps', format='eps', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

