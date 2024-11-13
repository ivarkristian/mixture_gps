import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import xarray as xr
import torch
import numpy as np

# functions for plotting

def scalar_field_tensors_to_dataframe(coordinates, values, x_coord='x', y_coord='y', value='value'):
    ''' Create a dictionary with the coordinates and values '''
    data = {'x': coordinates[:, 0], 'y': coordinates[:, 1], 'value': values}
    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame(data)
    return df

def tensors_to_dataframe(coordinates, values, x_coord='x', y_coord='y', value='value'):
    ''' Create a dictionary with the coordinates and values '''
    data = {'x': coordinates, 'value': values}
    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame(data)
    return df

def df_scalar_field_scatter(df, x='x', y='y', z='value', name=None, marker_symbol='circle', marker_size=2, marker_color='red', fig=None, mode='markers', subplot=[1, 1]):
    ''' Create a scatter plot with the scalar field values at each coordinate '''

    if not fig:
        fig = go.Figure()

    marker=dict(color=marker_color, size=marker_size)
    fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['value'], name=name, mode=mode, marker=marker, marker_symbol=marker_symbol), row=subplot[0], col=subplot[1])

    return fig

def df_tensors_scatter(df, x='x', y='value', name=None, marker_symbol='circle', marker_size=2, marker_color='red', fig=None, mode='markers', subplot=[1, 1]):
    ''' Create a scatter plot with the scalar field values at each coordinate '''

    if not fig:
        fig = go.Figure()

    marker=dict(color=marker_color, size=marker_size)
    fig.add_trace(go.Scatter(x=df['x'], y=df['value'], name=name, mode=mode, marker=marker, marker_symbol=marker_symbol), row=subplot[0], col=subplot[1])

    return fig

def df_scalar_field_surface(df, fig=None, subplot=[1, 1]):
    ''' use plotly to create a surface plot with the data from the dataframe '''
    if not fig:
        fig = go.Figure()
    
    fig.add_trace(go.Surface(x=df['x'], y=df['y'], z=df['value']), row=subplot[0], col=subplot[1])
    #fig = px.surface(df, x='x', y='y', z='value')
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                    width=40, height=10,
                    margin=dict(l=65, r=50, b=65, t=90))
    return fig

def visualize_tensors(coordinates, values, name=None, plot_type='scatter', marker_symbol='circle', marker_size=2, marker_color='red', fig=None, mode='markers', subplot=[1, 1]):
    ''' Wrapper for above functions '''
    if coordinates.ndim == 2:
        df = scalar_field_tensors_to_dataframe(coordinates, values)

        if plot_type.lower() == 'scatter':
            fig = df_scalar_field_scatter(df, name=name, marker_symbol=marker_symbol, marker_size=marker_size, marker_color=marker_color, fig=fig, mode=mode, subplot=subplot)
        elif plot_type.lower() == 'surface':
            fig = df_scalar_field_surface(df, fig=fig)
        else:
            print(f'plot_type ({plot_type}) not supported')
    elif coordinates.ndim == 1:
        df = tensors_to_dataframe(coordinates, values)
        
        if plot_type.lower() == 'scatter':
            fig = df_tensors_scatter(df, name=name, marker_symbol=marker_symbol, marker_size=marker_size, marker_color=marker_color, fig=fig, mode=mode, subplot=subplot)

    return fig


def plot_experiment(env, path, kernel):

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]])
    # Environment
    fig = visualize_tensors(env.coords, env.values, name='Environment', marker_color='dodgerblue', fig=fig, subplot=[1, 1])
    fig = visualize_tensors(path.sample_coords, path.sample_values, name='AUV path', marker_color='crimson', fig=fig, mode='markers+lines', subplot=[1, 1])
    #fig = gpt_plot.visualize_tensors(k1_coords, k1_vals, name='In plume', marker_color='orange', fig=fig, subplot=[1, 1])
    #fig = gpt_plot.visualize_tensors(k2_coords, k2_vals, name='Out of plume', marker_color='orange', fig=fig, subplot=[1, 1])
    fig = visualize_tensors(kernel.centres, 0, name='Train centre', marker_size=3, marker_color='black', fig=fig, subplot=[1, 1])
    # Prediction based on samples
    fig = visualize_tensors(env.coords, kernel.mm_prediction['values'], name='Estimate', marker_color='mediumseagreen', fig=fig, subplot=[1, 2])
    fig = visualize_tensors(path.sample_coords, path.sample_values, name='AUV path', marker_color='crimson', fig=fig, mode='markers+lines', subplot=[1, 2])
    fig = visualize_tensors(kernel.centres, 0, name='Train centre', marker_color='black', fig=fig, subplot=[1, 2])
    # Error prediction-environment
    fig = visualize_tensors(env.coords, kernel.mm_prediction['values']-env.values, name='Error', marker_color='rosybrown', fig=fig, subplot=[1, 3])
    fig = visualize_tensors(path.sample_coords, path.sample_values, name='AUV path', marker_color='crimson', fig=fig, mode='markers+lines', subplot=[1, 3])
    fig = visualize_tensors(kernel.centres, 0, name='Train centre', marker_color='black', fig=fig, subplot=[1, 3])

    pred_vals = kernel.mm_prediction['values']
    RMS_string = ''
    if env.values.size() == pred_vals.size():
        RMS_error = torch.sqrt(torch.sum(((env.values - pred_vals)**2)/env.values.size(0))).item()
        print(f'RMS error:  {RMS_error}')
        RMS_string = '{:.5f}'.format(RMS_error)
        fig_text = 'Truth                 -                 Estimated                 -                 Error (RMS ' + RMS_string + ')'

    fig.update_layout(title={'text': fig_text,\
            'y': 0.9, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
    return fig


# PRINT STUFF
def print_tensor_with_indexes(array, name=None):
    ''' prints a 1-dimensional or 2-dimensional torch tensor with indexes, similar to how they would be displayed in Excel '''
    if name is not None:
        print(f"Tensor '{name}':")

    # Convert the input to a PyTorch tensor if it's a NumPy array
    if isinstance(array, np.ndarray):
        tensor = torch.Tensor(array)
    else:
        tensor = array
    
    if len(tensor.shape) == 1:
        # Handle 1D tensors
        max_width = max(len("{:.4f}".format(val)) for val in tensor.tolist())
        header = [""] + [str(i).rjust(max_width) for i in range(tensor.shape[0])]
        header_str = "\t".join(header)
        print(header_str)
        row = ["0"] + ["{:.4f}".format(val.item()).rjust(max_width) for val in tensor]
        row_str = "\t".join(row)
        print(row_str)

    elif len(tensor.shape) == 2:
        # Handle 2D tensors
        max_widths = [max(len("{:.4f}".format(val)) for val in col) for col in tensor.t().tolist()]
        header = [""] + [str(i).rjust(width) for i, width in enumerate(max_widths)]
        header_str = "\t".join(header)
        print(header_str)
        for i in range(tensor.shape[0]):
            row = [str(i).rjust(len(header[0]))] + ["{:.4f}".format(val.item()).rjust(width) for val, width in zip(tensor[i], max_widths)]
            row_str = "\t".join(row)
            print(row_str)
    else:
        raise ValueError("Input tensor should be 1-dimensional or 2-dimensional.")


def print_2d_tensor_with_indexes(tensor):
    ''' prints a 2-dimensional torch tensor with indexes, similar to how they would be displayed in Excel '''
    
    max_widths = [max(len("{:.4f}".format(val)) for val in col) for col in tensor.t().tolist()]
    header = [""] + [str(i).rjust(width) for i, width in enumerate(max_widths)]
    header_str = "\t".join(header)
    print(header_str)
    for i in range(tensor.shape[0]):
        row = [str(i).rjust(len(header[0]))] + ["{:.4f}".format(val.item()).rjust(width) for val, width in zip(tensor[i], max_widths)]
        row_str = "\t".join(row)
        print(row_str)


def print_torch(torch_class):
    for name, value in torch_class.named_parameters():
        if value.dim() == 0:
            print(f'{name} = {value.item()}')
        else:
            print(f'{name} = ', end='')
            print(value)

