# %%
import importlib
import torch
import gpytorch
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import gpt_classes
import gpt_functions
import gpt_plot
#import cairosvg
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import geopandas as gpd
import contextily as ctx

# %%
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Setup for plotting
samples_cmap = 'YlOrRd'
samples_linewidth = 0
samples_edgecolor = 'k'

pred_cmap = 'YlOrRd'
pred_linewidth = 0
pred_edgecolor = 'k'

# Disable LaTeX rendering to avoid the need for an external LaTeX installation
# Use MathText for LaTeX-like font rendering
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX usage
    "font.family": "serif",  # Use a serif font that resembles LaTeX's default
    "mathtext.fontset": "dejavuserif"  # Use DejaVu Serif font for mathtext, similar to LaTeX fonts
})

# %%
# READ DATA
dfs = []
i = 0

path = '../../methane_flights/'
files = [f for f in sorted(os.listdir(path)) if f.endswith(".csv")]

for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(path + file)
        df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
        df = df.set_index('Time Stamp')
        #df.plot.scatter(x='Longitude', y='Latitude', c='CH4 (ppm)', colormap='jet')
        dfs.append(df)
        print(f'#{i}: {file} contains {len(df)} samples')
        i += 1

# inspection shows that file #8 20230812_082930_tjotta.csv contains the better CH4 data
df = pd.concat([dfs[0], dfs[3]])

test_survey_num = 8
df_test = dfs[test_survey_num]

# %%
# Convert to meters and setup tensors
lon_lat_deg = pd.concat([df['Longitude'], df['Latitude']], axis=1).to_numpy()
sample_coords_deg = torch.Tensor(lon_lat_deg)
# At 65 deg north, 1/60 degree in latitude corresponds to 0.7832 km
# This means 1 degree corresponds to 46992 m
df['Lon [m]'] = (df['Longitude'] - min(df['Longitude']))*46992
df['Lat [m]'] = (df['Latitude'] - min(df['Latitude']))*111180
lon_lat_np = pd.concat([df['Lon [m]'], df['Lat [m]']], axis=1).to_numpy()
sample_coords = torch.Tensor(lon_lat_np)
sample_values = torch.Tensor(df['CH4 (ppm)'])

df_test['Lon [m]'] = (df_test['Longitude'] - min(df_test['Longitude']))*46992
df_test['Lat [m]'] = (df_test['Latitude'] - min(df_test['Latitude']))*111180
lon_lat_np_test = pd.concat([df_test['Lon [m]'], df_test['Lat [m]']], axis=1).to_numpy()
sample_coords_test = torch.Tensor(lon_lat_np_test)
sample_values_test = torch.Tensor(df_test['CH4 (ppm)'])

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]])
fig = gpt_plot.visualize_tensors(sample_coords, sample_values, name='Samples', marker_color='black', fig=fig, subplot=[1, 1])
fig = gpt_plot.visualize_tensors(sample_coords_test, sample_values_test, name='Samples_test', marker_color='black', fig=fig, subplot=[1, 2])
fig.show()

smooth_coords = gpt_functions.generate_smooth_coords(df, padding=5)
smooth_coords_test = gpt_functions.generate_smooth_coords(df_test, padding=5)

# %%
# Set some common parameters, constraints etc.
noise_interval=gpytorch.constraints.Interval(0.0001, 1)
length_constraint=gpytorch.constraints.GreaterThan([25, 2.5])
type = 'scale_rbf_ard' # kernel type

# %%
# Rotate and normalize if necessary
cdir = -165.0
cdir_test = -135.0

if type in ['matern_ard', 'SMK', 'scale_rbf_ard']:
    normalize = False
    rotate = True
else:
    normalize = False
    rotate = False

if rotate and normalize:
    coords, params = gpt_classes.rotate_and_normalize_coords_v2(sample_coords, cdir)
    values = gpt_classes.normalize_values(sample_values)
    smooth, _ = gpt_classes.rotate_and_normalize_coords_v2(smooth_coords, cdir, params)
    coords_test, _ = gpt_classes.rotate_and_normalize_coords_v2(sample_coords_test, cdir_test, params)
    values_test = gpt_classes.normalize_values(sample_values_test)
    smooth_test, _ = gpt_classes.rotate_and_normalize_coords_v2(smooth_coords, cdir_test, params)
    # To restore: gpt_classes.denormalize_values(values, sample_values)
elif rotate:
    coords = gpt_classes.rotate_coords(sample_coords, cdir)
    values = sample_values
    smooth = gpt_classes.rotate_coords(smooth_coords, cdir)
    coords_test = gpt_classes.rotate_coords(sample_coords_test, cdir_test)
    values_test = sample_values_test
    smooth_test = gpt_classes.rotate_coords(smooth_coords_test, cdir_test)
else:
    coords = sample_coords
    values = sample_values
    smooth = smooth_coords
    coords_test = sample_coords_test
    values_test = sample_values_test
    smooth_test = smooth_coords

norm_fig = make_subplots(rows=1, cols=2, subplot_titles=(("Training data", "Test data")), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
norm_fig.update_layout(title_text='Rotate = %s, normalize = %s' % (rotate, normalize), title_x=0.5)
norm_fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=values, name='CH4 obs. train', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=1)
norm_fig.add_trace(go.Scatter3d(x=smooth[:, 0], y=smooth[:, 1], z=[1.8]*len(smooth), name='smooth train', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=True), row=1, col=1)
norm_fig.add_trace(go.Scatter3d(x=coords_test[:, 0], y=coords_test[:, 1], z=values_test, name='CH4 obs. test', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=2)
norm_fig.add_trace(go.Scatter3d(x=smooth_test[:, 0], y=smooth_test[:, 1], z=[1.9]*len(smooth_test), name='smooth test', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=True), row=1, col=2)
norm_fig.show()

# %%
# Show denormalized data:
orig_values = gpt_classes.denormalize_values(values, sample_values)
denorm_fig = make_subplots(rows=1, cols=2, subplot_titles=("denorm", "original"), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
denorm_fig.update_layout(title_text='Denormalized observations', title_x=0.5)
denorm_fig.add_trace(go.Scatter3d(x=sample_coords[:, 0], y=sample_coords[:, 1], z=orig_values, name='CH4 denorm', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=1)
denorm_fig.add_trace(go.Scatter3d(x=sample_coords[:, 0], y=sample_coords[:, 1], z=sample_values, name='CH4 observed', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=2)
denorm_fig.show()

denorm_fig = make_subplots(rows=1, cols=2, subplot_titles=("normalized", "original"), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
denorm_fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=values, name='CH4 normalized', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=1)
denorm_fig.add_trace(go.Scatter3d(x=smooth[:, 0], y=smooth[:, 1], z=[0]*len(smooth[:, 1]), name='test coords', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=True), row=1, col=1)
denorm_fig.add_trace(go.Scatter3d(x=sample_coords[:, 0], y=sample_coords[:, 1], z=sample_values, name='CH4 original', marker_color='black', mode='markers', marker_size=2, showlegend=True), row=1, col=2)
showlegend=False
denorm_fig.add_trace(go.Scatter3d(x=smooth_coords[:, 0], y=smooth_coords[:, 1], z=[0]*len(smooth[:, 1]), name='test coords', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=True), row=1, col=2)
denorm_fig.show()


# %%
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Demonstrate training with early stopping on a single kernel
llh = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
mdl = gpt_classes.ExactGPModel(coords, values, llh, type, lengthscale_constraint=length_constraint)

length_init = torch.tensor([25, 2.5])
if type in 'scale_rbf_ard':
    mdl.covar_module.base_kernel.lengthscale = length_init
elif type in 'matern_ard':
    mdl.covar_module.lengthscale = length_init

iter = 80
st = time.process_time()
gpt_functions.train_model(coords, values, mdl, iter=iter, early_delta=(False, 'mll', 0, 0, 0), debug=False)
single_kernel_training_time = time.process_time() - st

x_l = []
y_l = []
for n in mdl.lengthscales[:mdl.curr_trained]:
    x_l.append(n[0])
    y_l.append(n[1])

fig_train = make_subplots(rows=1, cols=2)
showlegend=True
fig_train.update_layout(title_text='Single kernel MLL loss', title_x=0.5)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=mdl.losses[:mdl.curr_trained], name='Loss'), row=1, col=1)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=x_l, name='Lengthscale 0'), row=1, col=2)
fig_train.add_trace(go.Scatter(x=mdl.iter[:mdl.curr_trained], y=y_l, name='Lengthscale 1'), row=1, col=2)
fig_train.show()
mdl.print_named_parameters()

# %%
# Plot predictions after early stopping
mdl.eval()
mdl.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mdl.set_train_data(coords_test, values_test, strict=False)
    st = time.process_time()
    single_pred = mdl.likelihood(mdl(smooth_test))
    single_kernel_prediction_time = time.process_time() - st

single_lower, single_upper = single_pred.confidence_region()

if normalize:
    single_pred_mean = gpt_classes.denormalize_values(single_pred.mean, sample_values)
    single_pred_upper = gpt_classes.denormalize_values(single_upper, sample_values)
else:
    single_pred_mean = single_pred.mean
    single_pred_upper = single_upper

s_fig = make_subplots(rows=1, cols=1, subplot_titles=(("")), specs=[[{'type': 'scene'}]])
s_fig.update_layout(title_text='Single kernel predictions', title_x=0.5)
showlegend=True
s_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=single_pred_mean, name='CH4 predicted', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
s_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 observed', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
s_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=single_pred_upper, marker_color='lightgrey', name='2 sigma', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
s_fig.show()


# %%
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Mixture models init
num_models = 2
type = type
noise_interval = noise_interval
length_constraint = length_constraint

mixture_likelihoods = []
mixture_models = []
for i in range(num_models):
    mixture_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
    mixture_likelihoods.append(mixture_likelihood)
    mixture_model = gpt_classes.ExactGPModel(coords, values, mixture_likelihoods[i], type, lengthscale_constraint=length_constraint)

    mixture_models.append(mixture_model)


# %%
# Train mixture models with EM
training_iter = 20#26
inner_iter = 10#10
eval_mode = 'mll'
st = time.process_time()
ret = gpt_functions.EM_algorithm(mixture_models, coords, values, iter=training_iter, inner_iter=inner_iter, visualization_coords=smooth, early_delta=(False, eval_mode, 0, 0, 0))
MoE_training_time = time.process_time() - st
mixture_models, train_responsibilities, figs, val_figs = ret

for i, model in enumerate(mixture_models):
    print(f'model {i}:')
    model.print_named_parameters()
    print('')


# %%
# Plot training result of mixture models
fig_mix_train = make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=True, subplot_titles=("", ""))

m = ['anomaly', 'background']
showlegend=True
for i, model in enumerate(mixture_models):
    iter_offset = 0#i*init_iter
    x_l = []
    y_l = []
    for n in model.lengthscales[:model.curr_trained]:
        x_l.append(n[0])
        y_l.append(n[1])

    fig_mix_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=model.losses[:model.curr_trained], name='Loss %s' % (m[i])), row=1, col=1)
    fig_mix_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=x_l, name='Lx %s' % (m[i])), row=2, col=1)
    fig_mix_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=y_l, name='Ly %s' % (m[i])), row=2, col=1)
    fig_mix_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=model.outputscales[:model.curr_trained], name='scale'), row=2, col=1)

fig_mix_train.update_layout(autosize=False, width=600, height=500)
text = 'Mixture GPs training'
fig_mix_train.update_layout(title_text=text, title_x=0.5)
fig_mix_train.show()


# %%
# Show figures for prediction on training
for i, fig in enumerate(figs[:3]):
    fname = 'gmm_train' + str(i)
    fig.show()
    

print(f'Noise model 0: {mixture_models[0].likelihood.noise.item()}')
print(f'Noise model 1: {mixture_models[1].likelihood.noise.item()}')

for val_fig in val_figs:
    val_fig.show()

# %%
# Predict test set sequentially with one model at a time
means = []; lowers = []; uppers = []
for i, model in enumerate(mixture_models):
    #mdl = deepcopy(model)
    model.set_train_data(coords_test, values_test, strict=False)
    model.eval()
    likelihood = model.likelihood
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(smooth_test)) # Generate test_coords along the longitude coords
        means.append(observed_pred.mean)
        lower, upper = observed_pred.confidence_region()
        lowers.append(lower); uppers.append(upper)

# %%
# Plot sequentially
fig = make_subplots(rows=1, cols=2, subplot_titles=(("Model 0", "Model 1")), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
showlegend=True
for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):

    if normalize:
        mean = gpt_classes.denormalize_values(mean, sample_values)
        lower = gpt_classes.denormalize_values(lower, sample_values)
        upper = gpt_classes.denormalize_values(upper, sample_values)

    fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=mean, name='CH4 predicted', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 observed', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z = upper, name='2 sigma', marker_color='grey', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    showlegend=False

fig_text = f'Test data predictions. (Train_iter: {training_iter} - inner_iter: {inner_iter})'
fig.update_layout(title={'text': fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# Plot variance
for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
    if normalize:
        mean = gpt_classes.denormalize_values(mean, sample_values)
        lower = gpt_classes.denormalize_values(lower, sample_values)
        upper = gpt_classes.denormalize_values(upper, sample_values)
    
    variance = (upper-mean).div(2.0).square()
    plt = gpt_functions.plot_2d(smooth_coords_test, variance, sample_coords_test, 'CH4 prediction variance model %d' % (i))
    plt.show()

# %%
# Train gating function
# Training gating function is equivalent to using threshold to separate into NLEs
length_constraint=length_constraint
gating_type = 'matern_ard'
gating_train_iter = 200

test_log_probs, _ = gpt_functions.compute_log_probs(mixture_models, coords_test, values_test)
test_responsibilities = torch.exp(gpt_functions.compute_responsibilities(test_log_probs, normalize=True))

gating_models = []
et = [0, 0]
for i, model in enumerate(mixture_models):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gating_model = gpt_classes.ExactGPModel(coords_test, test_responsibilities[:, i], likelihood, gating_type, length_constraint)
    st = time.process_time()
    gpt_functions.train_model(coords_test, test_responsibilities[:, i], gating_model, iter=gating_train_iter, lr=0.1, early_delta=(False, '', 0, 0, 0), debug=False)
    et[i] = time.process_time() - st

    gating_model.print_named_parameters()

    gating_models.append(gating_model)

gating_training_time = et[0] + et[1]

# %%
# Plot training result of gating models
fig_gating_train = make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=True, subplot_titles=("", ""))

m = ['anomaly', 'background']
showlegend=True
for i, model in enumerate(gating_models):
    iter_offset = 0#i*init_iter
    x_l = []
    y_l = []
    for n in model.lengthscales[:model.curr_trained]:
        x_l.append(n[0])
        y_l.append(n[1])

    fig_gating_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=model.losses[:model.curr_trained], name='Loss %s' % (m[i])), row=1, col=1)
    fig_gating_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=x_l, name='Lx %s' % (m[i])), row=2, col=1)
    fig_gating_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=y_l, name='Ly %s' % (m[i])), row=2, col=1)
    fig_gating_train.add_trace(go.Scatter(x=torch.arange(model.curr_trained), y=model.outputscales[:model.curr_trained], name='scale'), row=2, col=1)

fig_gating_train.update_layout(autosize=False, width=600, height=500)
text = 'Gating GPs training'
fig_gating_train.update_layout(title_text=text, title_x=0.5)
fig_gating_train.show()

# %%
# Predict gating function
gating_means = []; gating_lowers = []; gating_uppers = []
et = [0, 0]
for i, gating_model in enumerate(gating_models):
    gating_model.eval()
    likelihood = gating_model.likelihood
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        st = time.process_time()
        observed_pred = likelihood(gating_model(smooth_test))
        et[i] = time.process_time() - st
        gating_means.append(observed_pred.mean)
        gating_lower, gating_upper = observed_pred.confidence_region()
        gating_lowers.append(gating_lower); gating_uppers.append(gating_upper)

gating_prediction_time = et[0] + et[1]
gating_fig = make_subplots(rows=1, cols=2, subplot_titles=(("Anomaly", "Background")), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
showlegend=True
for i, gating_mean in enumerate(gating_means):
    gating_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 observed', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    gating_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=test_responsibilities[:, i], name='Responsibility', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    gating_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=gating_mean, name='Gating', marker_color='purple', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)

gating_fig.show()

# %%
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Predict mixture of models
for i, model in enumerate(mixture_models):
    model.set_train_data(coords_test, values_test, strict=False)

st = time.process_time()
observed_pred = gpt_functions.predict_mixture(smooth_test, mixture_models, gating_means, test_responsibilities)
MoE_prediction_time = time.process_time() - st
(mixture_mean, lower, sigmas_upper_MoE, p_of_z) = observed_pred

if normalize:
    mixture_mean = gpt_classes.denormalize_values(mixture_mean, sample_values)
    sigmas_upper_MoE = gpt_classes.denormalize_values(sigmas_upper_MoE, sample_values)

pz_fig = make_subplots(rows=1, cols=len(gating_means), subplot_titles=(("Anomaly model", "Background model")), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
showlegend=True
for i, gating_mean in enumerate(gating_means):
    resp_filter = test_responsibilities[:, i] >= 0.5
    pz_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0][resp_filter], y=sample_coords_test[:, 1][resp_filter], z=sample_values_test[resp_filter], name='Responsibilities', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    pz_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=p_of_z[:, i], name='p_of_z', marker_color='green', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    pz_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=gating_mean, name='Gating', marker_color='purple', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
pz_fig.show()

# %%
# Plot mixture prediction in 3D
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
showlegend=True

fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 measurements [ppmv]', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=mixture_mean, name='CH4 predicted [ppmv]', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=sigmas_upper_MoE, marker_color='lightgrey', name='2 sigma', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
showlegend=False
fig_text = f'Mixture prediction. Train_iter: {training_iter} - inner_iter: {inner_iter}'
fig.update_layout(title={'text': fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# %%
# Plot mixture prediction in 2D
df_sns = pd.DataFrame({
    'x': smooth_coords_test.numpy()[:, 0],
    'y': smooth_coords_test.numpy()[:, 1],
    'value': mixture_mean.numpy()
})

df_path = pd.DataFrame({
    'x': sample_coords_test.numpy()[:, 0],
    'y': sample_coords_test.numpy()[:, 1]
})

ax = plt.subplots()
ax = sns.scatterplot(data=df_sns, x='x', y='y', hue='value', palette='coolwarm', s=4, hue_norm=(mixture_mean.min(), mixture_mean.max()))
ax = sns.scatterplot(data=df_path, x='x', y='y', s=4, color='black')
plt.title('CH4 predictions (MoE) [ppmv]')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.legend(loc='upper right')
plt.show()

# %%
# Plot prediction variances in 2D
MoE_variance = (sigmas_upper_MoE-mixture_mean).div(2.0).numpy()
single_variance = (single_pred_upper-single_pred_mean).div(2.0).numpy()

plt = gpt_functions.plot_2d(smooth_coords_test, MoE_variance, sample_coords_test, 'CH4 prediction variance (MoE)')
plt.legend(loc='upper right')
plt.show()

plt = gpt_functions.plot_2d(smooth_coords_test, single_variance, sample_coords_test, 'CH4 prediction variance (single kernel)')
plt.legend(loc='upper right')
plt.show()

# %%
############ NLE METHODS ############
# %%
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# NLE detect subsets
min_cluster_dist = 14
threshold = 2.1
min_region_dist = 35
min_region_dist_test = 35

ret = gpt_functions.detect_clusters_wrapper(sample_coords, sample_values, threshold, min_cluster_dist, min_region_dist, debug=False)
anomaly, no_anomaly, sample_coords_subsets, sample_values_subsets, cluster, clusters_tensors, regions, regions_tensors = ret

ret_test = gpt_functions.detect_clusters_wrapper(sample_coords_test, sample_values_test, threshold, min_cluster_dist, min_region_dist_test, debug=False)
anomaly_test, no_anomaly_test, sample_coords_subsets_test, sample_values_subsets_test, clusters_test, clusters_tensors_test, regions_test, regions_tensors_test = ret_test

# --- training data set plotting
ax = plt.subplots()
plt.title('Anomaly clusters and regions for NLE (training)')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
df_nle = gpt_functions.tensors_to_df(sample_coords, sample_values)
ax = sns.scatterplot(data=df_nle, x='x', y='y', hue='value', palette='YlOrRd', s=4, hue_norm=(sample_values.min(), sample_values.max()))

showlegend=True

for i, cluster in enumerate(clusters_tensors):
    df_cluster = gpt_functions.tensors_to_df(cluster)
    ax = sns.lineplot(data=df_cluster, x='x', y='y', color='green', sort=False, estimator=None)
    showlegend=False

showlegend=True
for i, region in enumerate(regions_tensors):
    df_region = gpt_functions.tensors_to_df(region)
    ax = sns.lineplot(data=df_region, x='x', y='y', color='red', sort=False, estimator=None)
    showlegend=False

plt.show()

# --- test data set plotting
ax = plt.subplots()
plt.title('Anomaly clusters and regions for NLE (test)')
plt.xlabel('East [m]')
plt.ylabel('North [m]')
df_nle = gpt_functions.tensors_to_df(sample_coords_test, sample_values_test)
ax = sns.scatterplot(data=df_nle, x='x', y='y', hue='value', palette='YlOrRd', s=4, hue_norm=(sample_values.min(), sample_values.max()))


showlegend=True

for i, cluster in enumerate(clusters_tensors_test):
    df_cluster = gpt_functions.tensors_to_df(cluster)
    ax = sns.lineplot(data=df_cluster, x='x', y='y', color='green', sort=False, estimator=None)
    showlegend=False

showlegend=True
for i, region in enumerate(regions_tensors_test):
    df_region = gpt_functions.tensors_to_df(region)
    ax = sns.lineplot(data=df_region, x='x', y='y', color='red', sort=False, estimator=None)
    showlegend=False

plt.show()

# %%
# NLE models init
num_nle_models = 2
type = type
noise_interval = noise_interval
length_constraint = length_constraint

coords_subsets = [coords[anomaly], coords[no_anomaly]]
values_subsets = [values[anomaly], values[no_anomaly]]

nle_likelihoods = []
nle_models = []
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i in range(num_nle_models):
    gpt_functions.plot_2d_NLE(ax=axs[i], sample_coords=coords_subsets[i], sample_values=values_subsets[i], cmap=pred_cmap)
    nle_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_interval)
    nle_likelihoods.append(nle_likelihood)
    nle_model = gpt_classes.ExactGPModel(coords_subsets[i], values_subsets[i], nle_likelihoods[i], type, lengthscale_constraint=length_constraint)
    nle_models.append(nle_model)

# %%
# NLE train on subsets
fig_train = make_subplots(rows=2, cols=1)

iter = 80#training_iter*inner_iter
e_delta = [False, False]
et = [0, 0]
for i, model in enumerate(nle_models):
    st = time.process_time()
    gpt_functions.train_model(coords_subsets[i], values_subsets[i], model, iter=iter, early_delta=(e_delta[i], 'mll', sample_coords_subsets[i], sample_values_subsets[i], 0), debug=False)
    et[i] = time.process_time() - st
    x_l = []
    y_l = []
    s = []
    for n in model.lengthscales[:model.curr_trained]:
        x_l.append(n[0])
        y_l.append(n[1])
    
    fig_train.add_trace(go.Scatter(x=model.iter[:model.curr_trained], y=model.losses[:model.curr_trained], name='Loss %s' % (m[i])), row=1, col=1)
    fig_train.add_trace(go.Scatter(x=model.iter[:model.curr_trained], y=x_l, name='Lx %s' % (m[i])), row=2, col=1)
    fig_train.add_trace(go.Scatter(x=model.iter[:model.curr_trained], y=y_l, name='Ly %s' % (m[i])), row=2, col=1)
    fig_train.add_trace(go.Scatter(x=model.iter[:model.curr_trained], y=model.outputscales[:model.curr_trained], name='scale'), row=2, col=1)
    model.print_named_parameters()

NLE_training_time = et[0] + et[1]
fig_train.update_layout(autosize=False, width=600, height=500)
fig_train.update_layout(title_text='NLE GPs training', title_x=0.5)
fig_train.show()

# %%
# Plot training result of mixture models, lengthscales only
fig_mix_train, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
m = ['Anomaly', 'Background']
showlegend = True

linestyles = ['-', '--']
for i, model in enumerate(mixture_models):
    x_l = [n[0] for n in model.lengthscales[:model.curr_trained]]
    y_l = [n[1] for n in model.lengthscales[:model.curr_trained]]

    axs[0].plot(x_l, label='Lx %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)
    axs[0].plot(y_l, label='Ly %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)

for i, model in enumerate(nle_models):
    x_l = [n[0] for n in model.lengthscales[:model.curr_trained]]
    y_l = [n[1] for n in model.lengthscales[:model.curr_trained]]

    axs[1].plot(x_l, label='Lx %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)
    axs[1].plot(y_l, label='Ly %s' % (m[i]), linestyle=linestyles[i], linewidth=1, markersize=2)

# Setting the axis limits
axs[0].set_xlim((0, 200))
axs[0].set_ylim((0, 50))
axs[1].set_xlim((0, 80))
axs[1].set_ylim((0, 50))

# Adding labels and title with improved formatting
axs[0].set_xlabel('Iterations', fontsize=12)
axs[1].set_xlabel('Iterations', fontsize=12)
axs[0].set_ylabel('Lengthscale [m]', fontsize=12)
axs[1].set_ylabel('Lengthscale [m]', fontsize=12)
#axs[1].set_yticklabels([])
axs[0].set_title('MoE', fontsize=14)
axs[1].set_title('NLE', fontsize=14)

# Adding a legend with improved formatting
axs[0].legend(fontsize=10, loc='upper left')
axs[1].legend(fontsize=10, loc='upper left')

# Adding grid lines for better readability
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Improving the appearance of the plot
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

fig_mix_train.subplots_adjust(wspace=0.3)  # Adjust hspace to add more vertical space between rows
fig_mix_train.suptitle('Lengthscales (Lx, Ly) training progression', y=1.0, fontsize=14, weight='bold')
# Saving the plot with high resolution
fig_mix_train.savefig('tjotta_mixture_gps_lengthscales.eps', format='eps', dpi=300)

# %%
# NLE predict test set

print(f'-- Predicting {len(smooth_test)} locations --')

coords_subsets_test = [coords_test[anomaly_test], coords_test[no_anomaly_test]]
values_subsets_test = [values_test[anomaly_test], values_test[no_anomaly_test]]

# Find subsets of samples within region, not only clusters
mask = torch.zeros(len(sample_coords_test))

for polygon in regions_test:
    mask = mask + polygon.contains_points(sample_coords_test.numpy()).astype(int)
mask = (mask > 0)
mask_inv = mask.__invert__()
coords_region_subsets_test = [coords_test[mask], coords_test[mask_inv]]
values_region_subsets_test = [values_test[mask], values_test[mask_inv]]

preds_NLE = []

et = [0, 0]
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    model.set_train_data(coords_region_subsets_test[i], values_region_subsets_test[i], strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        st = time.process_time()
        pred = model.likelihood(model(smooth_test)) # both models predict entire domain
        et[i] = time.process_time() - st
    preds_NLE.append(pred)

NLE_prediction_time = et[0] + et[1]

# %%
# Then pick prediction model based on within anomaly region or not
mask = torch.zeros(len(smooth_coords_test))
preds_mean_NLE = torch.zeros(len(smooth_test))
sigmas_lower_NLE = torch.zeros(len(smooth_test))
sigmas_upper_NLE = torch.zeros(len(smooth_test))

st = time.process_time()
for polygon in regions_test:
    mask = mask + polygon.contains_points(smooth_coords_test.numpy()).astype(int)
NLE_contains_points_time = time.process_time() - st

mask = (mask > 0)
mask_int = mask.int()
mask_inv = mask.__invert__().int()
preds_mean_NLE = preds_mean_NLE + preds_NLE[0].mean*mask_int
preds_mean_NLE = preds_mean_NLE + preds_NLE[1].mean*mask_inv
sigmas_lower_NLE = sigmas_lower_NLE + preds_NLE[0].confidence_region()[0]*mask_int
sigmas_lower_NLE = sigmas_lower_NLE + preds_NLE[1].confidence_region()[0]*mask_inv
sigmas_upper_NLE = sigmas_upper_NLE + preds_NLE[0].confidence_region()[1]*mask_int
sigmas_upper_NLE = sigmas_upper_NLE + preds_NLE[1].confidence_region()[1]*mask_inv

# denormalize
if normalize:
    NLE_mean = gpt_classes.denormalize_values(preds_mean_NLE, sample_values)
    NLE_sigmas_upper = gpt_classes.denormalize_values(sigmas_upper_NLE, sample_values)
    preds_mean_NLE_m0 = gpt_classes.denormalize_values(preds_NLE[0].mean, sample_values)
    preds_mean_NLE_m1 = gpt_classes.denormalize_values(preds_NLE[1].mean, sample_values)
    sigmas_upper_NLE_m0 = gpt_classes.denormalize_values(preds_NLE[0].confidence_region()[1], sample_values)
    sigmas_upper_NLE_m1 = gpt_classes.denormalize_values(preds_NLE[1].confidence_region()[1], sample_values)
else:
    NLE_mean = preds_mean_NLE
    NLE_sigmas_upper = sigmas_upper_NLE
    preds_mean_NLE_m0 = preds_NLE[0].mean
    preds_mean_NLE_m1 = preds_NLE[1].mean
    sigmas_upper_NLE_m0 = preds_NLE[0].confidence_region()[1]
    sigmas_upper_NLE_m1 = preds_NLE[1].confidence_region()[1]

preds_stddev_NLE_m0 = (sigmas_upper_NLE_m0 - preds_mean_NLE_m0)/2.0
preds_stddev_NLE_m1 = (sigmas_upper_NLE_m1 - preds_mean_NLE_m1)/2.0

nle_fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
showlegend=True
nle_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0][mask], y=smooth_coords_test[:, 1][mask], z=NLE_mean[mask], name='CH4 pred. NLE', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0][mask.__invert__()], y=smooth_coords_test[:, 1][mask.__invert__()], z=NLE_mean[mask.__invert__()], name='CH4 pred. NLE', marker_color='blue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=NLE_sigmas_upper, name='Pred. variance NLE', marker_color='grey', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
nle_fig.update_layout(title_text='Predict test region', title_x=0.5)
nle_fig.show()

# %%
# Plot predictions and standard deviations in 2D
plt = gpt_functions.plot_2d(smooth_coords_test, NLE_mean, sample_coords_test, 'CH4 prediction mean (NLE)')
plt.legend(loc='upper right')
plt.show()

NLE_stddev = (NLE_sigmas_upper-NLE_mean).div(2.0).numpy()
plt = gpt_functions.plot_2d(smooth_coords_test, NLE_stddev, sample_coords_test, 'CH4 prediction stddev (NLE)')
plt.legend(loc='upper right')
plt.show()

figs, axs = plt.subplots(2, 2, figsize=(10, 10))

title='Anomaly mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 0], sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=preds_NLE[0].mean, title=title)
title='Anomaly stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 1], sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=preds_NLE[0].stddev, title=title)
title='Background mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 0], sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=preds_NLE[1].mean, title=title)
title='Background stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 1], sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=preds_NLE[1].stddev, title=title)
figs.show()

# %%
# Plot both NLE models
fig = make_subplots(rows=1, cols=2, subplot_titles=(("Anomaly", "Background")), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
showlegend=True
for i, pred in enumerate(preds_NLE):

    if normalize:
        mean = gpt_classes.denormalize_values(pred.mean, sample_values)
        lower = gpt_classes.denormalize_values(pred.confidence_region()[0], sample_values)
        upper = gpt_classes.denormalize_values(pred.confidence_region()[1], sample_values)
    else:
        mean = pred.mean
        lower = pred.confidence_region()[0]
        upper = pred.confidence_region()[1]

    fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 observed', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=mean, name='CH4 predicted', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
    fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=upper, name='2 sigma', marker_color='grey', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)

fig_text = f'NLE predictions. Train_iter: {training_iter}'
fig.update_layout(title={'text': fig_text,\
        'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()


# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# NLE-MA prediction

st = time.process_time()
window_value = 10
distances = torch.norm(smooth_coords_test.unsqueeze(1) - sample_coords_test.unsqueeze(0), dim=2)
nearest = distances.topk(window_value, largest=False, dim=1) # The window_value smallest elements

# Weighted MLL
preds_NLE_weight = []
for i, model in enumerate(nle_models):
    model.eval()
    model.likelihood.eval()
    model.set_train_data(coords_test, values_test, strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(coords_test)) # both models predict observations
    preds_NLE_weight.append(pred)

# Get the K's and u's to be weighted
K0 = preds_NLE_weight[0].covariance_matrix
u0 = preds_NLE_weight[0].mean
K1 = preds_NLE_weight[1].covariance_matrix
u1 = preds_NLE_weight[1].mean
log_pr = torch.full((len(distances), 2), 0.0)

for i, nearest_indices in enumerate(nearest.indices):
    weights = torch.tensor([0.0]*len(coords_test))
    weights[nearest_indices] = (torch.tensor(1.0).div(nearest.values[i])).clamp_max(1.0)
    log_pr[i, 0] = gpt_functions.compute_weighted_mll(values_test, K0, weights, u0, jitter=1e-6)
    log_pr[i, 1] = gpt_functions.compute_weighted_mll(values_test, K1, weights, u1, jitter=1e-6)
    if i%1000 == 0:
        print(f'{i}...')

max_log_probs = torch.max(log_pr, dim=1)[0].unsqueeze(1)
log_probs_n = log_pr - max_log_probs
factor = log_probs_n[:, 0].min()/log_probs_n[:, 1].min()
normed_log_pr = torch.full((len(distances), 2), 0.0)
normed_log_pr[:, 0] = log_probs_n[:, 0]
normed_log_pr[:, 1] = log_probs_n[:, 1]*factor

resps = torch.exp(gpt_functions.compute_responsibilities(normed_log_pr, normalize=False))
mean_bma, sigmas_lower_bma, sigmas_upper_bma, nle_bma_pofz = gpt_functions.predict_mixture(smooth_test, nle_models, (resps[:, 0], resps[:, 1]))
NLE_MA_prediction_time = time.process_time() - st

# %%
# Print figures
figs, axs = plt.subplots(1, 2, figsize=(10, 4.5))
title = 'Anomaly logpr'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=normed_log_pr[:, 0], title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'Background logpr'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=normed_log_pr[:, 1], title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

figs, axs = plt.subplots(1, 2, figsize=(10, 4.5))
title = 'Mix mean'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=mean_bma, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'Mix stddev'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=(sigmas_upper_bma-mean_bma)/2.0, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')


# %%
# OBSOLETE
use_coords = sample_coords_test[nearest.indices]
use_values = sample_values_test[nearest.indices]

use_norm_coords = coords_test[nearest.indices]
use_norm_values = values_test[nearest.indices]

use_preds_mean_m0 = preds_NLE[0].mean[nearest.indices]
use_preds_mean_m1 = preds_NLE[1].mean[nearest.indices]
use_preds_stddev_m0 = preds_NLE[0].stddev[nearest.indices]
use_preds_stddev_m1 = preds_NLE[1].stddev[nearest.indices]

diff = torch.full((len(use_coords), 2), 0.0)
weight = torch.full((len(use_coords), 2), 0.0)
logdetK = torch.full((len(use_coords), 2), 0.0)
log_probs = torch.full((len(use_coords), 2), 0.0)
ymK = torch.full((len(use_coords), 2), 0.0)
log_pr = torch.full((len(use_coords), 2), 0.0)

for c, ch4_here in enumerate(use_values):
    
    diff[c, 0] = (preds_NLE[0].mean[c] - ch4_here.mean()).__abs__()
    weight[c, 0] = (diff[c, 0])**2# * use_preds_stddev_m1[c].mean()

    
    diff[c, 1] = (preds_NLE[1].mean[c] - ch4_here.mean()).__abs__()
    weight[c, 1] = (diff[c, 1])**2# * use_preds_stddev_m0[c].mean()

st = time.process_time()
resps = torch.exp(gpt_functions.compute_responsibilities(log_pr, normalize=True))
nle_bma_mean, nle_bma_lower, nle_bma_upper, nle_bma_pofz = gpt_functions.predict_mixture(smooth_test, nle_models, (resps[:, 0], resps[:, 1]))
NLE_MA_prediction_time = time.process_time() - st + NLE_prediction_time

log_weights_m0 = weight[:, 0]
log_weights_m1 = weight[:, 1]
weights_m0 = torch.exp(weight[:, 0])
weights_m1 = torch.exp(weight[:, 1])
p_m0 = weights_m0.div(weights_m0 + weights_m1)
p_m1 = weights_m1.div(weights_m0 + weights_m1)
log_p_m1 = log_weights_m0.div(log_weights_m0 + log_weights_m1)
log_p_m0 = log_weights_m1.div(log_weights_m0 + log_weights_m1)

mean_bma = log_p_m0 * preds_mean_NLE_m0 + log_p_m1 * preds_mean_NLE_m1

mean_diff = mean_bma-NLE_mean

variances = torch.stack((preds_stddev_NLE_m0**2, preds_stddev_NLE_m1**2), dim=1)
means_stack = torch.stack((preds_mean_NLE_m0, preds_mean_NLE_m1), dim=1)
var_norm = torch.square(means_stack - mean_bma.unsqueeze(1))
log_p_stack = torch.stack((log_p_m0, log_p_m1), dim=1)
var_bma = torch.sum((variances + var_norm)*log_p_stack, dim=1)

sigmas_upper_bma = mean_bma + 2*torch.sqrt(var_bma)
sigmas_lower_bma = mean_bma - 2*torch.sqrt(var_bma)
stddev_bma = torch.sqrt(var_bma)

# %%
figs, axs = plt.subplots(2, 2, figsize=(10, 10))

title = 'CH4 prediction mean (NLE)'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=NLE_mean, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'CH4 prediction mean (NLE-BMA)'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=mean_bma, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'CH4 prediction stddev (NLE)'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=NLE_stddev, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'CH4 prediction stddev (NLE-BMA)'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=stddev_bma, title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

# %%
figs, axs = plt.subplots(1, 2, figsize=(10, 4.5))
title = 'Anomaly resps'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=resps[:, 0], title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

title = 'Background resps'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=resps[:, 1], title=title, cmap=pred_cmap)
fig.legend(loc='upper right')

# %%
# Plot weights 

weights_fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]])
showlegend=True
weights_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=p_m0, name='p model 0', marker_color='purple', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=p_m1, name='p model 1', marker_color='green', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
weights_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=log_p_m0, name='Log p model 0', marker_color='purple', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=log_p_m1, name='Log p model 1', marker_color='green', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=diff[:, 0], name='diff model 0', marker_color='purple', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
weights_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=diff[:, 1], name='diff model 1', marker_color='green', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=2)
weights_fig.show()

# %%
# Plot GMM, NLE, BMA
total_fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
showlegend=True
total_fig.add_trace(go.Scatter3d(x=sample_coords_test[:, 0], y=sample_coords_test[:, 1], z=sample_values_test, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
total_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=mean_bma, name='CH4 pred. NLE-BMA', marker_color='dodgerblue', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
total_fig.add_trace(go.Scatter3d(x=smooth_coords_test[:, 0], y=smooth_coords_test[:, 1], z=sigmas_upper_bma, marker_color='lightgrey', name='2 stddev NLE-BMA', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
showlegend=False
total_fig.show()

# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
def adjust_axis_labels(ax, minx, miny, maxx, maxy):
    # Get current tick positions
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    
    # Compute new tick labels relative to minx and miny
    x_labels = [int(x - minx) for x in x_ticks]
    y_labels = [int(y - miny) for y in y_ticks]
    
    # Set the new tick labels
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

# %%
# Create a GeoDataFrame from your coordinates
from mpl_toolkits.axes_grid1 import make_axes_locatable
df['m_size'] = df['CH4 (ppm)']**4
df_test['m_size'] = df_test['CH4 (ppm)']**4
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
gdf_test = gpd.GeoDataFrame(df_test, geometry=gpd.points_from_xy(df_test.Longitude, df_test.Latitude))

# Set the coordinate reference system (CRS) to WGS84
gdf.crs = {'init': 'epsg:4326'}
gdf_test.crs = {'init': 'epsg:4326'}

# Reproject to Web Mercator for compatibility with contextily
gdf = gdf.to_crs(epsg=3857)
gdf_test = gdf_test.to_crs(epsg=3857)

#plt.figure(figsize=(8, 5.2))
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Training plot
divider = make_axes_locatable(axs[0])
#cax = divider.append_axes("right", size="5%", pad=0.1)

# Plot the points on top of a basemap
ax = gdf.plot('CH4 (ppm)', ax=axs[0], cax=None, figsize=(10, 10), cmap=samples_cmap, markersize='m_size', alpha=1, linewidth=samples_linewidth, edgecolor=samples_edgecolor)#, legend=True, legend_kwds={"label":"CH4 (ppm)"})#, edgecolor='k')

minx, miny, maxx, maxy = gdf_test.total_bounds
z = 35
ax.set_xlim(minx-z, maxx+z)
ax.set_ylim(miny-z, maxy+z)
ax.set_title('Training survey')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Add clusters and regions
# At 65 deg north, 1/60 degree in latitude corresponds to 0.7832 km
# This means 1 degree corresponds to 46992 m
df['Lon [m]'] = (df['Longitude'] - min(df['Longitude']))*46992
df['Lat [m]'] = (df['Latitude'] - min(df['Latitude']))*111180

for i, cluster in enumerate(clusters_tensors):
    df_cluster = gpt_functions.tensors_to_df(cluster)
    df_cluster['Longitude'] = df_cluster['x']/46992.0 + min(df['Longitude'])
    df_cluster['Latitude'] = df_cluster['y']/111180.0 + min(df['Latitude'])
    gdf_cluster = gpd.GeoDataFrame(df_cluster, geometry=gpd.points_from_xy(df_cluster.Longitude, df_cluster.Latitude))
    gdf_cluster.crs = {'init': 'epsg:4326'}
    gdf_cluster = gdf_cluster.to_crs(epsg=3857)
    ax = gdf_cluster.plot(ax=axs[0], cax=None, figsize=(10, 10), color='green', markersize=2, linewidth=0)
    # Extract coordinates for plotting lines
    x_coords = [point.x for point in gdf_cluster.geometry]
    y_coords = [point.y for point in gdf_cluster.geometry]

    # Draw lines between points
    axs[0].plot(x_coords, y_coords, color='green', linewidth=2)

for i, region in enumerate(regions_tensors):
    df_region = gpt_functions.tensors_to_df(region)
    df_region['Longitude'] = df_region['x']/46992.0 + min(df['Longitude'])
    df_region['Latitude'] = df_region['y']/111180.0 + min(df['Latitude'])
    gdf_region = gpd.GeoDataFrame(df_region, geometry=gpd.points_from_xy(df_region.Longitude, df_region.Latitude))
    gdf_region.crs = {'init': 'epsg:4326'}
    gdf_region = gdf_region.to_crs(epsg=3857)
    ax = gdf_region.plot(ax=axs[0], cax=None, figsize=(10, 10), color='red', markersize=2, linewidth=0)
    # Extract coordinates for plotting lines
    x_coords = [point.x for point in gdf_region.geometry]
    y_coords = [point.y for point in gdf_region.geometry]

    # Draw lines between points
    axs[0].plot(x_coords, y_coords, color='red', linewidth=2)

# Test plot
divider = make_axes_locatable(axs[1])


# Plot the points on top of a basemap
ax = gdf_test.plot('CH4 (ppm)', ax=axs[1], cax=None, figsize=(10, 10), cmap=samples_cmap, markersize='m_size', alpha=1, linewidth=samples_linewidth, edgecolor=samples_edgecolor)#, legend=True, legend_kwds={"label":"CH4 (ppm)"})#, edgecolor='k')

minx, miny, maxx, maxy = gdf_test.total_bounds
z = 35
ax.set_xlim(minx-z, maxx+z)
ax.set_ylim(miny-z, maxy+z)
ax.set_title('Test survey')
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Coordinates for question mark 1
x_coord = 1.39620*1e6  # desired longitude
y_coord = 9.85*1e6 + 80  # desired latitude
ax.text(x_coord, y_coord, '?', fontsize=14, ha='center', va='center', color='blue')

# Coordinates for question mark 2
x_coord = 1.39615*1e6  # desired longitude
y_coord = 9.85*1e6 + 15  # desired latitude

# Coordinates for question mark 3
x_coord = 1.39616*1e6  # desired longitude
y_coord = 9.85*1e6 - 45  # desired latitude
ax.text(x_coord, y_coord, '?', fontsize=14, ha='center', va='center', color='black')

# Coordinates for question mark 4
x_coord = 1.39605*1e6  # desired longitude
y_coord = 9.85*1e6 - 45  # desired latitude
ax.text(x_coord, y_coord, '?', fontsize=14, ha='center', va='center', color='black')

# Coordinates for question mark 5
x_coord = 1.39605*1e6  # desired longitude
y_coord = 9.85*1e6 - 160  # desired latitude
ax.text(x_coord, y_coord, '?', fontsize=14, ha='center', va='center', color='black')

# Coordinates for question mark 6
x_coord = 1.39600*1e6  # desired longitude
y_coord = 9.85*1e6 - 105  # desired latitude

desired_max_x = 250.0
desired_max_y = 200.0
minx, miny, maxx, maxy = gdf_test.total_bounds
maxx_adj = maxx-minx
maxy_adj = maxy-miny

desired_max_x_converted = maxx_adj*desired_max_x/gdf_test['Lon [m]'].max()
desired_max_y_converted = maxy_adj*desired_max_y/gdf_test['Lat [m]'].max()

xticklabels = [0, 50, 100, 150, 200, 250]
yticklabels = [0, 25, 50, 75, 100, 125, 150, 175, 200]
num_xticks = len(xticklabels)
num_yticks = len(yticklabels)
x_ticks = np.linspace(minx, minx+desired_max_x_converted, num_xticks)
y_ticks = np.linspace(miny, miny+desired_max_y_converted, num_yticks)

for ax in axs:
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    # Set the new tick labels
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Adjust axes and labels
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")

# Manually create a legend
hue_min = min(sample_values.min(), sample_values_test.min())
hue_max = max(sample_values.max(), sample_values_test.max())

norm = mcolors.Normalize(vmin=hue_min, vmax=hue_max)
sm = cm.ScalarMappable(cmap=samples_cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', shrink=0.5, fraction=0.05, pad=0.05)
cbar.set_label('CH$_4$ [ppm]', loc='center')

fig.tight_layout(rect=[0, 0.05, 1, 1.])
fig.subplots_adjust(wspace=0.20)

fig.savefig('map_samples.eps', format='eps', dpi=300)


# %%
# CREATE PREDICTIVE MEAN FIGURE
figs, axs = plt.subplots(2, 2, figsize=(10, 10))
hue_min = min([mixture_mean.min(), mean_bma.min(), single_pred_mean.min(), NLE_mean.min(), sample_values_test.min()])
hue_max = max([mixture_mean.max(), mean_bma.max(), single_pred_mean.max(), NLE_mean.max(), sample_values_test.max()])
# Plot prediction means in 2D
title='MoE'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 1], sample_coords=sample_coords_test, sample_values=sample_values_test, smooth_coords=smooth_coords_test, smooth_values=mixture_mean, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'NLE-MA'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, sample_values=sample_values_test, smooth_coords=smooth_coords_test, smooth_values=mean_bma, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'NLE'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[1, 0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, sample_values=sample_values_test, smooth_values=NLE_mean, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'Single kernel'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs[0, 0], smooth_coords=smooth_coords_test, smooth_values=single_pred_mean, sample_coords=sample_coords_test, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)

axs[0, 0].set_xlabel("East [m]")
axs[0, 1].set_xlabel("East [m]")
axs[1, 0].set_xlabel("East [m]")
axs[1, 1].set_xlabel("East [m]")
axs[0, 0].set_ylabel("North [m]")
axs[0, 1].set_ylabel("North [m]")
axs[1, 0].set_ylabel("North [m]")
axs[1, 1].set_ylabel("North [m]")

axs[0, 0].yaxis.set_major_locator(MultipleLocator(50))
axs[0, 1].yaxis.set_major_locator(MultipleLocator(50))
axs[1, 0].yaxis.set_major_locator(MultipleLocator(50))
axs[1, 1].yaxis.set_major_locator(MultipleLocator(50))

# Collect all handles and labels
handles, labels = axs[0, 0].get_legend_handles_labels()

# Hide individual legends
for ax in axs.flat:
    ax.get_legend().remove()

# Manually creating a legend
cmap = pred_cmap
norm = mcolors.Normalize(vmin=hue_min, vmax=hue_max)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = figs.colorbar(sm, ax=axs, orientation='horizontal', shrink=0.5, fraction=0.05, pad=-0.18)
cbar.set_label('CH$_4$ [ppm]', loc='center')

figs.tight_layout(rect=[0, 0.05, 1, 0.97])
figs.subplots_adjust(hspace=0.24, wspace=0.20)

suptitle = "CH$_4$ predictive mean"
figs.suptitle(suptitle, fontsize=14, weight='bold')
figs.savefig('tjotta_comparison_mean' + '.eps', format='eps', dpi=300)
figs.show()

# %%

# Compute prediction variances in 2D
MoE_stddev = (sigmas_upper_MoE-mixture_mean).div(2.0).numpy()
NLE_stddev = NLE_stddev
NLE_BMA_stddev = (sigmas_upper_bma-mean_bma).div(2.0).numpy()
single_stddev = (single_pred_upper-single_pred_mean).div(2.0).numpy()

# %%
# CREATE PREDICTIVE STDDEV FIGURE
figs_stddev, axs_stddev = plt.subplots(2, 2, figsize=(10, 10))
hue_min = min([MoE_stddev.min(), NLE_BMA_stddev.min(), single_stddev.min(), NLE_stddev.min()])
hue_max = max([MoE_stddev.max(), NLE_BMA_stddev.max(), single_stddev.max(), NLE_stddev.max()])

# Plot prediction stddevs in 2D
title='MoE'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs_stddev[0, 1], sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=MoE_stddev, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'NLE-MA'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs_stddev[1, 1], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=NLE_BMA_stddev, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'NLE'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs_stddev[1, 0], cluster_tensors=clusters_tensors_test, region_tensors=regions_tensors_test, sample_coords=sample_coords_test, smooth_coords=smooth_coords_test, smooth_values=NLE_stddev, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

title = 'Single kernel'
fig, ax = gpt_functions.plot_2d_NLE(ax=axs_stddev[0, 0], smooth_coords=smooth_coords_test, smooth_values=single_stddev, sample_coords=sample_coords_test, hue_min=hue_min, hue_max=hue_max, title=title, cmap=pred_cmap)
fig.savefig(title + '.eps', format='eps')

axs_stddev[0, 0].set_xlabel("East [m]")
axs_stddev[0, 1].set_xlabel("East [m]")
axs_stddev[1, 0].set_xlabel("East [m]")
axs_stddev[1, 1].set_xlabel("East [m]")
axs_stddev[0, 0].set_ylabel("North [m]")
axs_stddev[0, 1].set_ylabel("North [m]")
axs_stddev[1, 0].set_ylabel("North [m]")
axs_stddev[1, 1].set_ylabel("North [m]")

axs_stddev[0, 0].yaxis.set_major_locator(MultipleLocator(50))
axs_stddev[0, 1].yaxis.set_major_locator(MultipleLocator(50))
axs_stddev[1, 0].yaxis.set_major_locator(MultipleLocator(50))
axs_stddev[1, 1].yaxis.set_major_locator(MultipleLocator(50))

# Collect all handles and labels
handles, labels = axs_stddev[0, 0].get_legend_handles_labels()

# Hide individual legends
for ax in axs_stddev.flat:
    ax.get_legend().remove()

# Manually creating a legend
cmap = pred_cmap
norm = mcolors.Normalize(vmin=hue_min, vmax=hue_max)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = figs_stddev.colorbar(sm, ax=axs_stddev, orientation='horizontal', shrink=0.5, fraction=0.05, pad=-0.18)
cbar.set_label('CH$_4$ [ppm]')

figs_stddev.tight_layout(rect=[0, 0.05, 1, 0.97])
figs_stddev.subplots_adjust(hspace=0.24, wspace=0.20)

suptitle = "CH$_4$ predictive stddev"
figs_stddev.suptitle(suptitle, fontsize=14, weight='bold')
figs_stddev.savefig('tjotta_comparison_stddev' + '.eps', format='eps', dpi=300)
figs_stddev.show()

# %%
# REIMPORT LIBRARIES
importlib.reload(gpt_classes)
importlib.reload(gpt_functions)

# %%
# Print training and prediction times
print(f'single_kernel_training_time: {single_kernel_training_time} seconds')
print(f'single_kernel_prediction_time: {single_kernel_prediction_time} seconds')
print(f'MoE_training_time: {MoE_training_time} seconds')
print(f'MoE_prediction_time: {MoE_prediction_time} seconds')
print(f'gating_training_time: {gating_training_time} seconds')
print(f'gating_prediction_time: {gating_prediction_time} seconds')
print(f'NLE_training_time: {NLE_training_time} seconds')
print(f'NLE_prediction_time: {NLE_prediction_time} seconds')
print(f'NLE_MA_prediction_time: {NLE_MA_prediction_time} seconds')

