# The classes needed to use the GPyTorch libraries
import torch
import gpytorch
from gpytorch.constraints import Interval
import numpy as np
import math
from shapely.geometry import Point, Polygon, MultiPoint
from sklearn.cluster import DBSCAN
from matplotlib.path import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gpt_plot


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, type, lengthscale_constraint=gpytorch.constraints.Positive()):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.max_losses = 10000
        self.losses = [0]*self.max_losses
        self.lengthscales = [0]*self.max_losses
        self.outputscales = [0]*self.max_losses
        self.iter = [0]*self.max_losses
        self.curr_trained = 0
        self.val_losses = [0]*self.max_losses
        self.val_iter = [0]*self.max_losses
        self.saved_val_losses = 0
        self.mix_losses = [0]*self.max_losses
        self.mix_iter = [0]*self.max_losses
        self.saved_mix_losses = 0
        self.add_noises = torch.tensor([0.0]*train_x.size(0))
        
        if type == 'scale_rbf':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint))
        elif type == 'rbf':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint)
        elif type == 'scale_rq':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(lengthscale_constraint=lengthscale_constraint))
        elif type == 'scale_rbf_ard':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2, lengthscale_constraint=lengthscale_constraint))
        elif type == 'scale_matern':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=lengthscale_constraint))
        elif type == 'matern':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=lengthscale_constraint)
        elif type == 'matern_ard':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=2, lengthscale_constraint=lengthscale_constraint)
        elif type == 'SMK':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8, ard_num_dims=2)
            self.covar_module.initialize_from_data(train_x, train_y)
        else:
            print(f"Error: Kernel type {type} not recognized.")

    def forward(self, x, add_training_noises=None, add_gating_noises=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        if add_training_noises is not None and add_gating_noises is not None:
            if covar_x.shape[0] == len(add_training_noises) + len(add_gating_noises):
                self.add_noises = torch.cat((add_training_noises, add_gating_noises))
                #print(f'adding resps and gating')
            elif covar_x.shape[0] == len(add_training_noises):
                #print(f'adding resps')
                self.add_noises = add_training_noises
            elif covar_x.shape[0] == len(add_gating_noises):
                self.add_noises = add_gating_noises
                #print(f'adding gating')    
            noise_matrix = torch.diag(self.add_noises)
            covar_x = covar_x + noise_matrix
        elif add_training_noises is not None:
            self.add_noises = add_training_noises
            noise_matrix = torch.diag(self.add_noises)
            covar_x = covar_x + noise_matrix
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def save_loss(self, loss):
        i, ls = loss
        self.losses[self.curr_trained] = ls
        self.lengthscales[self.curr_trained] = self.get_lengthscale()
        self.outputscales[self.curr_trained] = self.get_outputscale()
        self.iter[self.curr_trained] = i
        self.curr_trained += 1
        return

    def save_val_loss(self, loss):
        i, ls = loss
        self.val_losses[self.saved_val_losses] = ls
        self.val_iter[self.saved_val_losses] = i
        self.saved_val_losses += 1
        return
    
    def save_mix_loss(self, loss):
        i, ls = loss
        self.mix_losses[self.saved_mix_losses] = ls
        self.mix_iter[self.saved_mix_losses] = i
        self.saved_mix_losses += 1
        return
    

    def visualize_loss(self):
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=self.iter,
            y=self.losses,
            mode='lines',
            name='Training loss',
            line=dict(color='blue'),  
        ))

        fig.add_trace(go.Scatter(
            x=self.val_iter,
            y=self.val_losses,
            mode='lines',
            name='Validation loss',
            line=dict(color='red'),  
        ))

        fig.update_layout(
            title="Training loss over time",
            xaxis_title="Iterations",
            yaxis_title="Loss value",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ),
            height = 500,
            width = 500
        )
        
        #fig.show()
        return fig
    
    def print_named_parameters(self):
        for name, value in self.named_parameters():
            name_no_raw = name.replace('raw_', '')
            param = 'self.' + name_no_raw
            num_params = len(value.size())
            with torch.no_grad():
                res = eval(param)
                print(f'{name_no_raw}:', end='')
                if num_params == 0:
                    print(f' {res.item()}')
                else:
                    print(f' {res}')
    
    def get_lengthscale(self):
        for name, value in self.named_parameters():
            if 'lengthscale' in name:
                name_no_raw = name.replace('raw_', '')
                param = 'self.' + name_no_raw
                with torch.no_grad():
                    res = eval(param)
                    num_params = len(value.size())
                    if num_params == 0:
                        return res.item()
                    else:
                        return res[0]
        return False
    
    def get_outputscale(self):
        for name, value in self.named_parameters():
            if 'outputscale' in name:
                name_no_raw = name.replace('raw_', '')
                param = 'self.' + name_no_raw
                with torch.no_grad():
                    res = eval(param)
                    return res.item()

        return False
    

class mmKernel():
    def __init__(self, params_dict):
        self.params = params_dict
        self.type = params_dict['type']
        self.likelihood = params_dict['likelihood']
        self.curr_trained = []
        self.centres = []
        self.num_peaks = 0
        self.subset_indices = []
        self.models = []
        self.likelihoods = []
        self.mm_prediction = {
            'values': torch.Tensor([]),
            'sigmas_lower': torch.Tensor([]),
            'sigmas_upper': torch.Tensor([])
        }
        self.bma_prediction = {
            'values': torch.Tensor([])
        }
        self.normalize = False
        self.rotate = False
        if self.type[-3:] == 'ard':
            self.normalize = True
            self.rotate = True


    def prepare(self, sample_coords, sample_values, debug=False):
        ''' 1. Select kernel centre locations. Want to capture plume gradients,
            but also represent entire data set/area
            2. Select data subset for each centre. If a kernel centre is inside
            the plume, we want training data to represent gradients of plume
        '''
        self.sample_coords = sample_coords
        self.sample_values = sample_values
        
        # Find the training centre locations. Two modes for test: 'peak+distance' and 'distance'
        self.centres, self.regions = self.train_centres(distance=self.params['min_peak_distance'], mode=self.params['train_centres_location_mode'], debug=debug)
        
        # Find the subset of sample data for each training centre.
        # 'euclidean' has only distance as parameter.
        # 'keyhole+plume' has close distance and angle computed from current strength and direction
        self.subset_indices = self.get_subsets(self.centres, data_subset_mode=self.params['train_subset_selection_mode'], subset_dist_cutoff=self.params['train_subset_dist_cutoff'], debug=debug)
        
        empty = 0
        for indice in self.subset_indices:
            if len(indice) == 0:
                empty += 1
        
        if debug:
            print(f'Found {len(self.centres)} centres/subsets of which {empty} have no samples to train on.')

        return
    
    def train(self, noise_constraint=gpytorch.constraints.GreaterThan(0.0001), iterations=1000, lr=0.1, early_delta=False, debug=False):

        if len(self.centres) == 0:
            print(f'No centres defined. Run mmKernel.prepare first!')
            return
        
        if self.normalize:
            coords = rotate_and_normalize_coords(self.sample_coords, self.params['current_direction'])
            #coords = normalize_coords(coords=self.sample_coords)
        else:
            coords = self.sample_coords
        
        #if self.rotate:
        #    coords = rotate_coords(coords, self.params['current_direction'])
        
        print(f'Training {len(self.subset_indices)} kernels')
        self.models = []
        for i, subset in enumerate(self.subset_indices):
            print(f'{i}: ', end='')
            #if debug:
                #print(f'')
                #print(f'Training subset {subset} on {len(subset)} samples:')
                #print(coords[subset])

            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
            model = ExactGPModel(coords[subset], self.sample_values[subset], likelihood, self.type)
            
            if iter is None:
                training_iter = model.max_losses
            else:
                training_iter = min(iterations, model.max_losses)
            
            if len(subset) > 0:
                model.train()
                likelihood.train()
                self.train_model(coords[subset], self.sample_values[subset], model, likelihood, training_iter, lr, early_delta=early_delta, debug=debug)
            self.models.append(model)
            self.likelihoods.append(likelihood)
            self.curr_trained.append(model.curr_trained)
        
        return

    def train_model(self, coords, values, model, likelihood, iter=100, lr=0.1, early_delta=False, debug=False):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        e_span = 20 # early stopping parameter
        e_num = 3 # early stopping parameter
        e_delta = early_delta # early stopping trigger
        e_stop = False

        percentage = 10
        p = round(iter/percentage)
        print_number = 0
        c = 0

        print(f'Training for {iter} iterations, lr = {lr}, early stopping = {e_delta}')

        for i in range(iter):
            c += 1
            if c > p:
                c = 0
                print_number += percentage
                if not debug:
                    print(f'...{print_number}%', end='')
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(coords)
            # Calc loss and backprop gradients
            loss = -mll(output, values)
            loss.backward()
            model.save_loss((i, loss.item()))
            if debug:
                if (i > 0) and (i%100 == 0):
                    print(f'Iter {i}/{iter} - Loss: {loss.item()} Lengthscale: {model.covar_module.lengthscale} Noise: {likelihood.noise.item()}')
            optimizer.step()

            if e_delta:
                if model.curr_trained > e_span+e_num:
                    w = model.max_losses - model.curr_trained
                    e_mean1 = sum(model.losses[-w-e_span-e_num:-w-e_span])/e_num
                    e_mean2 = sum(model.losses[-w-e_num:-w])/e_num
                    if abs(e_mean1-e_mean2) < e_delta:
                        e_stop = True
            
            if e_stop:
                print(f'')
                print(f'Early stopping at iteration {i} of {iter}: {abs(e_mean1-e_mean2)} < {e_delta}')
                print(f'Iter {i}/{iter} - Loss: {loss.item()}')
                model.print_named_parameters()
                print(f'')
                break
        
        if not debug:
            print(f'..100%')
        return

    def train_centres(self, distance=40, mode='grid', debug=False):
        ''' Returns a new tensor with locations evenly distributed over the grid defined by coords.
            We could imagine that n of the locations was allocated to the areas with highest gradients,
            and the rest was evenly distributed along path. Or, a distribution that is as even over the
            grid as possible, but always snapped to locations along the path. '''

        if debug:
            print(f'Finding training centres (mode={mode})')
        
        if mode == 'distance':  
            return self.sample_coords[int(distance/2)::int(distance), :], 0
        
        elif mode == 'peaks_per_pass':
            # separating into arrays (or AUV passes), assuming that x is 
            # constant for each pass
            passes_coords = []
            passes_values = []
            uniques = self.sample_coords[:, 0].unique()
            x_vals = self.sample_coords[:, 0]
            for u in uniques:
                mask = x_vals == u
                passes_coords.append(self.sample_coords[mask])
                passes_values.append(self.sample_values[mask])

            peak_centres = torch.Tensor()
            peak_regions = []
            for i, pass_values in enumerate(passes_values):
                pass_peak_indices, pass_regions_indices = self.detect_peaks(pass_values.numpy().flatten(), 50, min_peak_distance=10, debug=debug)
                peak_centres = torch.cat((peak_centres, passes_coords[i][pass_peak_indices]))
                for region in pass_regions_indices:
                    peak_regions.append(passes_coords[i][region])
            
            indices = []
            for region in peak_regions:
                reg_indices = []
                for coord in region:
                    equality = torch.all(self.sample_coords == coord, dim=1)
                    # Find the index where the row is equal to the target tensor
                    index = torch.nonzero(equality).squeeze()
                    reg_indices.append(index.item())

                indices.append(reg_indices)
            
            # Now the peaks_per_pass contains index of peaks inside each pass. We no longer
            # need to think of individual passes, so we flatten the coordinates to one array.

            if debug:
                gpt_plot.print_tensor_with_indexes(peak_centres, 'peak_centres')
            
            #n_peaks = peak_centres.size(0)*peak_centres.size(1)
            #peak_centres = peak_centres.reshape([n_peaks, 2])
            
            return peak_centres, indices

        elif mode == 'peaks':
            peak_indexes, peak_regions = self.detect_peaks(self.sample_values.numpy().flatten(), 100, min_peak_distance=distance, min_region_distance=self.params['min_region_distance'], debug=debug)
            self.num_peaks = len(peak_indexes)

            if debug:
                print(f'Found {len(peak_indexes)} peak indexes: {peak_indexes}')
                gpt_plot.print_tensor_with_indexes(self.sample_coords[peak_indexes], 'peak_coordinates')
                print(f'Found {len(peak_regions)} peak_regions: {peak_regions}')
            return self.sample_coords[peak_indexes], peak_regions
        
        elif mode == 'peaks+distance':
            # The peaks mode detects the peaks in the data set that are above 95% CI. Up to
            # ten peaks can be detected, but we will have 1-3 in practice. The peaks will 
            # have one centre each. From the largest
            # peak, kernel centres are spread out evenly with distance 'distance', but 
            # centres that are closer to the peak centres than 'distance' are removed
            peak_indexes, peak_regions = self.detect_peaks(self.sample_values.numpy().flatten(), 10, min_peak_distance=10)
            self.num_peaks = len(peak_indexes)
            distance_centres = self.select_elements(self.sample_coords, peak_indexes[0], distance)
            result_centres = self.exclude_centers_closer_than(self.sample_coords[peak_indexes], distance_centres, distance/2)
            if debug:
                print(f'distance: {distance}')
                gpt_plot.print_tensor_with_indexes(self.sample_coords[peak_indexes], 'peak_centres')
                gpt_plot.print_tensor_with_indexes(distance_centres, 'distance_centres')
                gpt_plot.print_tensor_with_indexes(result_centres, 'result_centres')
            return result_centres, peak_regions
        
        print(f'Mode {mode} not supported. Use either \'peaks+distance\', \'distance\' or \'peaks\'.')
        return
    

    def predict(self, test_coords, bma=False, bma_n=10, debug=False):
        ''' No particular data subset for prediction, uses training data only.
        '''
        
        print(f'-- Predicting {len(test_coords)} locations --')
        values = torch.zeros(len(test_coords))
        sigmas_lower = torch.zeros(len(test_coords))
        sigmas_upper = torch.zeros(len(test_coords))

        percentage = 10
        p = round(len(test_coords)/percentage)
        print_number = 0
        c = 0

        # Check if coord system is 2D, only then read current params
        cdir = 0; cstr = 0
        if 'current_direction' in self.params:
            cdir = self.params['current_direction']
        if 'current_strength' in self.params:
            cstr = self.params['current_strength']

        if self.normalize:
            print(f'Normalizing and rotating coordinates')
            coords = rotate_and_normalize_coords(test_coords, cdir)
            s_coords = rotate_and_normalize_coords(self.sample_coords, cdir)
            
            #coords = normalize_coords(test_coords)
            #s_coords = normalize_coords(self.sample_coords)
        else:
            coords = test_coords
            s_coords = self.sample_coords
        
        #if self.rotate:
        #    coords = rotate_coords(coords, self.params['current_direction'])
        #    s_coords = rotate_coords(s_coords, self.params['current_direction'])

        print(f'Predicting plume and background separately')

        predictions = []
        # Predict all coords with all models
        for idx, m in enumerate(self.models):
            likelihood = self.likelihoods[idx]
            model = self.models[idx]
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                prediction = likelihood(model(coords))
            predictions.append(prediction)

        # fill prediction list according to nearest kernel idx
        nrst_krn_mode = self.params['nearest_kernel_mode']
        if nrst_krn_mode == 'plume_sep_polygons':
            mask = np.zeros(len(test_coords))
            for polygon in self.polygons:
                mask = mask + polygon.contains_points(test_coords.numpy()).astype(int)
            mask = (mask > 0)
            mask_int = mask.astype(int)
            mask_inv = mask.__invert__().astype(int)
            values = values + predictions[0].mean*mask_int
            values = values + predictions[1].mean*mask_inv
            sigmas_lower = sigmas_lower + predictions[0].confidence_region()[0]*mask_int
            sigmas_lower = sigmas_lower + predictions[1].confidence_region()[0]*mask_inv
            sigmas_upper = sigmas_upper + predictions[0].confidence_region()[1]*mask_int
            sigmas_upper = sigmas_upper + predictions[1].confidence_region()[1]*mask_inv
        else:
            for i in range(len(coords)):
                c += 1
                if c > p:
                    c = 0
                    print_number += percentage
                    print(f'...{print_number}%', end='')

                idx = self.get_nearest_kernel_idx(test_coords[i], nearest_kernel_mode=nrst_krn_mode, debug=debug)
                values[i] = predictions[idx].mean[i]
                sigmas_lower[i] = predictions[idx].confidence_region()[0][i]
                sigmas_upper[i] = predictions[idx].confidence_region()[1][i]

            print(f'...100%')

        self.mm_prediction['values'] = values
        self.mm_prediction['sigmas_lower'] = sigmas_lower
        self.mm_prediction['sigmas_upper'] = sigmas_upper

        if bma is not False:
            # fill bma prediction list according to weight
            print(f'Predicting BMA')

            p_y_gvn_m0, p_y_gvn_m1, use_c = two_dimensional_mll(test_coords, self.sample_coords, s_coords, self.sample_values, bma_n, self.models, method=bma, cdir=cdir, cstr=cstr)

            # compute p(m|y) for both
            p_m0_gvn_y = p_y_gvn_m0/(p_y_gvn_m0 + p_y_gvn_m1)
            p_m1_gvn_y = p_y_gvn_m1/(p_y_gvn_m0 + p_y_gvn_m1)

            #p_m0_gvn_y_EMLL = p_y_gvn_m0_EMLL/(p_y_gvn_m0_EMLL + p_y_gvn_m1_EMLL)
            #p_m1_gvn_y_EMLL = p_y_gvn_m1_EMLL/(p_y_gvn_m0_EMLL + p_y_gvn_m1_EMLL)

            self.bma_prediction['values'] = p_m0_gvn_y*predictions[0].mean + p_m1_gvn_y*predictions[1].mean

            #return (p_m0_gvn_y, p_m1_gvn_y, p_m0_gvn_y_EMLL, p_m1_gvn_y_EMLL, use_c)
            return (p_m0_gvn_y, p_m1_gvn_y, use_c)
        
        return (0, 0, 0)


    def get_nearest_kernel_idx(self, test_coord, nearest_kernel_mode='euclidean', debug=False):
        
        if nearest_kernel_mode == 'distance':
            diff = abs(torch.Tensor(self.centres) - test_coord)
            if diff.min() < self.params['distance']:
                kernel_idx = diff.argmin()
            else:
                kernel_idx = len(self.centres)
        
        elif nearest_kernel_mode == 'euclidean':
            diff = torch.Tensor(self.centres) - test_coord
            if diff.ndim < 2:
                dists = diff
            elif diff.ndim == 2:
                dists = torch.sum(diff**2, dim=1)
            kernel_idx = torch.argmin(dists).item()
        
        elif nearest_kernel_mode == 'plume_sep_merged':
            if len(self.regions) >= 1:
                if debug:
                    print(f'test_coord: {test_coord}')
                for region in self.regions:
                    if debug:
                        print(f'self.sample_coords 0: {self.sample_coords[region[0]]}, -1: {self.sample_coords[region[-1]]}')
                    if (test_coord >= self.sample_coords[region[0]] and test_coord <= self.sample_coords[region[-1]]):
                        kernel_idx = 0
                        if debug:
                            print(f'kernel_idx = 0')
                        break
                    else:
                        kernel_idx = 1
            else:
                kernel_idx = self.get_nearest_kernel_idx(test_coord, nearest_kernel_mode='distance', debug=debug)

        elif nearest_kernel_mode == 'plume_sep_rectangle':
            if len(self.regions) > 1:
                point = Point(test_coord)
                if point.within(self.rectangle):
                    kernel_idx = 0
                else:
                    kernel_idx = 1
            else:
                kernel_idx = self.get_nearest_kernel_idx(test_coord, nearest_kernel_mode='peak_sep', debug=debug)
        
        elif nearest_kernel_mode == 'plume_sep_polygons':
            if len(self.polygons) >= 1:
                point = Point(test_coord)
                in_poly = False
                for polygon in self.polygons:
                    if point.within(polygon):
                        in_poly = True
                        break
                
                if in_poly:
                    kernel_idx = 0
                else:
                    kernel_idx = 1
                    
            else:
                kernel_idx = self.get_nearest_kernel_idx(test_coord, nearest_kernel_mode='peak_sep', debug=debug)

        elif nearest_kernel_mode == 'peak_sep':
            dist_penalty = self.params['dist_penalty']
            dist_cutoff = self.params['train_subset_dist_cutoff']
            self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
            penalty = self.keyhole_penalty(self.centres, test_coord.unsqueeze(dim=0), dist_cutoff, dist_penalty)
            self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
            # Find the row in penalty with the lowest value, this should correspond to the
            # corresponding centre. If no values are below a threshold, use the no_peaks subset.
            if penalty.size(1) != 1:
                print(f'penalty matrix has strange size: {penalty.size()}')
            
            if debug:
                print(f'penalty: {penalty}')
            
            if penalty.min() < dist_penalty:
                kernel_idx = penalty.argmin()
            else:
                kernel_idx = len(self.centres)

        elif nearest_kernel_mode == 'directional':
            direction = self.params['current_direction']
            dir_rad = torch.Tensor([(direction/180)*torch.pi])
            direction_vector = torch.tensor([torch.cos(dir_rad), torch.sin(dir_rad)])
            # project the points onto the direction
            diff = self.centres - test_coord
            proj_diff = torch.matmul(diff, direction_vector)
            # mask to remove points in the negative direction
            mask = proj_diff <= 3

            if debug:
                print(f'direction_vector: {direction_vector}')
                print(f'diff: {diff}')
                print(f'mask: {mask}')
                print(f'proj_diff: {proj_diff}')

            proj_diff = proj_diff[mask]
            # If there are no centres upstream, return euclidian nearest
            if proj_diff.nelement() == 0:
                return self.get_nearest_kernel_idx(test_coord, nearest_kernel_mode='euclidean', debug=debug)
            
            dists = torch.abs(proj_diff)
            # project the points onto the orthogonal direction
            orthogonal_direction = torch.tensor([-direction_vector[1], direction_vector[0]])
            diff = diff[mask]
            proj_diff_orth = torch.matmul(diff, orthogonal_direction)
            dists_orth = torch.abs(proj_diff_orth)
            # new approach that just computes euclidean distance weighing current strength
            c_strength = self.params['current_strength']
            c_strength /= self.params['c_normalizer']
            dists_norm = dists/c_strength
            norm_dist = torch.sqrt(dists_norm.square() + dists_orth.square())
            if debug:
                print(f'norm_dist: {norm_dist}')

            nearest_idx = torch.argmin(norm_dist)
            
            # find the nearest point, using the orthogonal distance as a tie-breaker
            #min_dist, min_idx = torch.min(dists, dim=0)
            #min_dists_orth = dists_orth[dists == min_dist]
            #nearest_idx = min_idx if len(min_dists_orth) == 1 else torch.argmin(min_dists_orth)
            # a lot of work to find centres[mask][nearest_idx] in centres:
            res = (self.centres == self.centres[mask][nearest_idx])
            kernel_idx = torch.logical_and(res[:, 0], res[:, 1]).nonzero().item()

        else:
            print(f'get_nearest_kernel: No mode called {nearest_kernel_mode}')
        
        if debug:
            if kernel_idx < len(self.centres):
                print(f'Test_coord {test_coord} has nearest centre {self.centres[kernel_idx]}')
            else:
                print(f'Test_coord {test_coord} has no peak centre')

        return kernel_idx
    
    
    def get_subsets(self, origin_coords, data_subset_mode='keyhole', subset_dist_cutoff=20, debug=False):
        ''' returns a list of subset indices referring to sample_coords for each
            origin_coord.  The list looks like this:
            [[], [], [24], [23, 24, 25], [], [22, 23, 24, 25, 26]] 
            (note that some can be empty) '''
        
        if data_subset_mode == 'distance':
            indices = []
            for coord in origin_coords:
                # origin_coords are the centres
                min_lim = coord-subset_dist_cutoff
                max_lim = coord+subset_dist_cutoff
                subset = ((self.sample_coords > min_lim) & (self.sample_coords < max_lim)).nonzero().squeeze().tolist()
                indices.append(subset)
            
            # Then append all indices not part of any peak subsets
            peak_indices = []
            for indice_list in indices:
                peak_indices += indice_list
            
            if debug:
                print(f'peak_indices: {peak_indices}')
            original_sample_indices = torch.arange(len(self.sample_coords))
            combined = torch.cat((original_sample_indices, torch.Tensor(peak_indices)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_peak_indices = uniques[counts == 1]
            indices.append(not_peak_indices.tolist())
            return indices

        elif data_subset_mode == 'euclidean':
            eucl_dist = torch.cdist(origin_coords, self.sample_coords, p=2)
            eucl_dist_mask = (eucl_dist < self.params['distance']).long()
            # inverting eucl_dist_mask so that values are either 
            # -1 (within distance) or 0 (outside distance)
            penalty = -eucl_dist_mask
            threshold = -0.9
            # Find the indices where the penalty is below the threshold
            return self.penalty_to_subset_indices(penalty, threshold, debug=debug)

        elif data_subset_mode == 'plume_sep':
            indices = []
            merge_regions = []
            for region in self.regions:
                indices.append(region)
                merge_regions += region
            
            # Then add all indices without plume at the end
            original_sample_indices = torch.arange(len(self.sample_coords))
            
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())
            return indices
        
        elif data_subset_mode == 'plume_sep_merged':
            indices = []
            merge_regions = []
            for region in self.regions:
                merge_regions += region
            
            indices.append(merge_regions)
            # Then add all indices without plume at the end
            original_sample_indices = torch.arange(len(self.sample_coords))
            
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())
            return indices
        
        elif data_subset_mode == 'plume_sep_rectangular':
            indices = []
            merge_regions = []
            for region in self.regions:
                merge_regions += region
            
            indices.append(merge_regions)
            # Then add all indices without plume at the end
            original_sample_indices = torch.arange(len(self.sample_coords))
            
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())

            # also define self.rectangle
            top = []
            btm = []
            # Then add top and bottoms to form a rectangle
            for region in self.regions:
                max_index = torch.argmax(self.sample_coords[region][:, 1])
                min_index = torch.argmin(self.sample_coords[region][:, 1])
                top.append(self.sample_coords[region][max_index])
                btm.append(self.sample_coords[region][min_index])

            btm.reverse()
            if len(self.regions) > 1:
                #self.rectangle = Polygon(top + btm).buffer(1)
                self.rectangle = MultiPoint(self.sample_coords[indices[0]]).convex_hull

            return indices
        
        elif data_subset_mode == 'plume_sep_rectangular_v2':
            indices = []

            # First add all indices, not just plume. This is what separates v1 and v2
            original_sample_indices = torch.arange(len(self.sample_coords))
            indices.append(original_sample_indices)

            merge_regions = []
            for region in self.regions:
                merge_regions += region

            # Then add all indices without plume at the end
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())

            # also define self.rectangle
            top = []
            btm = []
            # Then add top and bottoms to form a rectangle
            for region in self.regions:
                max_index = torch.argmax(self.sample_coords[region][:, 1])
                min_index = torch.argmin(self.sample_coords[region][:, 1])
                top.append(self.sample_coords[region][max_index])
                btm.append(self.sample_coords[region][min_index])

            btm.reverse()
            if len(self.regions) > 1:
                self.rectangle = Polygon(top + btm).buffer(1)

            return indices
        
        elif data_subset_mode == 'plume_sep_polygons':
            indices = []
            merge_regions = []
            for region in self.regions:
                merge_regions += region
            
            indices.append(merge_regions)
            # Then add all indices without plume at the end
            original_sample_indices = torch.arange(len(self.sample_coords))
            
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())

            # Then we want to cluster the regions, to see if some are far apart

            # First we create a matrix to contain distances between regions
            reg_dist = np.zeros((len(self.regions), len(self.regions)))
            # Then we convert regions into multipoints
            geom = []
            for region in self.regions:
                geom.append(MultiPoint(self.sample_coords[region]))
            
            # And compute distances between all the regions
            for i, region1 in enumerate(geom):
                for j, region2 in enumerate(geom):
                    reg_dist[i][j] = region1.distance(region2)

            # Then we do the clustering
            db = DBSCAN(eps=30, min_samples=1, metric='precomputed').fit(reg_dist)
            labels = db.labels_
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # It is time to make Multipoints of clusters of regions
            self.polygons = []
            for cluster in clusters:
                merge_regions = []
                for region_num in clusters[cluster]:
                    merge_regions += self.regions[region_num]
                #self.polygons.append(MultiPoint(self.sample_coords[merge_regions]).convex_hull.buffer(2))
                polygon = MultiPoint(self.sample_coords[merge_regions]).convex_hull.buffer(2)
                x = polygon.exterior.xy[0]
                y = polygon.exterior.xy[1]
                points = np.vstack((x, y)).T
                self.polygons.append(Path(points))

            return indices
        
        elif data_subset_mode == 'plume_sep_polygons_v2':
            indices = []
            
            # First add all indices, not just plume. This is what separates v1 and v2
            original_sample_indices = torch.arange(len(self.sample_coords))
            indices.append(original_sample_indices)
            
            merge_regions = []
            for region in self.regions:
                merge_regions += region

            # Then add all indices without plume at the end
            combined = torch.cat((original_sample_indices, torch.Tensor(merge_regions)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_plume_indices = uniques[counts == 1]
            indices.append(not_plume_indices.tolist())

            # Then we want to cluster the regions, to see if some are far apart

            # First we create a matrix to contain distances between regions
            reg_dist = np.zeros((len(self.regions), len(self.regions)))
            # Then we convert regions into multipoints
            geom = []
            for region in self.regions:
                geom.append(MultiPoint(self.sample_coords[region]))
            
            # And compute distances between all the regions
            for i, region1 in enumerate(geom):
                for j, region2 in enumerate(geom):
                    reg_dist[i][j] = region1.distance(region2)
            
            # Then we do the clustering
            try:
                db = DBSCAN(eps=30, min_samples=1, metric='precomputed').fit(reg_dist)
                labels = db.labels_
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
            except:
                self.polygons = []
                return indices
            
            # It is time to make Multipoints of clusters of regions
            self.polygons = []
            for cluster in clusters:
                merge_regions = []
                for region_num in clusters[cluster]:
                    merge_regions += self.regions[region_num]
                #self.polygons.append(MultiPoint(self.sample_coords[merge_regions]).convex_hull.buffer(2))
                polygon = MultiPoint(self.sample_coords[merge_regions]).convex_hull.buffer(2)
                x = polygon.exterior.xy[0]
                y = polygon.exterior.xy[1]
                points = np.vstack((x, y)).T
                self.polygons.append(Path(points))

            return indices

        elif data_subset_mode == 'peak_sep_keyhole':
            # We want the subsets corresponding to each peak/centre, using the 
            # downstream keyhole method. And if we
            # use the peak_sep mode we also get the final subset of samples which are
            # not belonging to a peak centre
            indices = []
            self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
            for coord in origin_coords:
                downstream_indices =\
                     self.get_subsets(origin_coords=coord.unsqueeze(0), data_subset_mode='keyhole', subset_dist_cutoff=subset_dist_cutoff, debug=debug)
                if isinstance(downstream_indices[0], list):
                        downstream_indices = downstream_indices[0]
                indices.append(downstream_indices)
            
            self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
            
            peak_indices = []
            for indice_list in indices:
                peak_indices += indice_list
            
            if debug:
                print(f'peak_indices: {peak_indices}')
            original_sample_indices = torch.arange(len(self.sample_coords))
            combined = torch.cat((original_sample_indices, torch.Tensor(peak_indices)))
            uniques, counts = combined.unique(dim=0, return_counts=True)
            not_peak_indices = uniques[counts == 1]

            # So the final element is the subset with no plume in it. Which coords
            # are included in the plume subset is determined by the keyhole penalties.
            # This could have been done simpler by using a concentration threshold
            # for the plume subset.
            indices.append(not_peak_indices.tolist())
            return indices

        elif data_subset_mode == 'keyhole':

            dist_penalty = self.params['dist_penalty']
            dist_cutoff = subset_dist_cutoff
            penalty = self.keyhole_penalty(origin_coords, self.sample_coords, dist_cutoff, dist_penalty, debug)
            # Find the indices where the penalty is below the threshold
            return self.penalty_to_subset_indices(penalty, dist_penalty, debug=debug)
        
        elif data_subset_mode == 'keyhole+plume':
            # 1. Get keyhole upstream indices
            upstream_indices = self.get_subsets(origin_coords=origin_coords, data_subset_mode='keyhole', debug=debug)
            
            # 2. Recognize whether data point is within plume (between peaks)
            downstream_indices = []
            plume_tolerance = self.params['plume_tolerance']
            for i, coord in enumerate(origin_coords):
                between_peaks = 0
                for idx in range(0, self.num_peaks - 1):
                    between_peaks += self.is_point_close_to_segment(self.centres[idx], self.centres[idx+1], coord, plume_tolerance)
                
                if between_peaks > 0:
                    # 3. In that case, get keyhole downstream indices and remove duplicate indices
                    # print(f'coord #{i}:] {coord} is between peaks')
                    self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
                    downstream = self.get_subsets(origin_coords=coord.unsqueeze(0), data_subset_mode='keyhole', debug=debug)
                    if isinstance(downstream[0], list):
                        downstream = downstream[0]
                    self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
                    
                else:
                    downstream = []
                
                downstream_indices.append(downstream)

            subset_indices = self.merge_lists(upstream_indices, downstream_indices)
            # 4. Return indices
            return subset_indices
        
        elif data_subset_mode == 'keyhole+plume2':
            # If between peaks, gets downstream keyhole and close sample subset
            # Else, gets euclidean subset
            
            # 1. Recognize whether data point is within plume (between peaks)
            subset_indices = []
            plume_tolerance = self.params['plume_tolerance']
            for i, coord in enumerate(origin_coords):
                between_peaks = 0
                for idx in range(0, self.num_peaks - 1):
                    between_peaks += self.is_point_close_to_segment(self.centres[idx], self.centres[idx+1], coord, plume_tolerance)
                
                if between_peaks > 0:
                    # 2. In that case, get close and keyhole downstream indices
                    # print(f'coord #{i}:] {coord} is between peaks')
                    self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
                    indices = self.get_subsets(origin_coords=coord.unsqueeze(0), data_subset_mode='keyhole', debug=debug)
                    self.params['current_direction'] = self.opposite_angle(int(self.params['current_direction']))
                else:
                    indices = self.get_subsets(origin_coords=coord.unsqueeze(0), data_subset_mode='euclidean', debug=debug)
                
                if isinstance(indices[0], list):
                    indices = indices[0]
                subset_indices.append(indices)

            # 4. Return indices
            return subset_indices
        
        else:
            print(f'Error: data_subset_mode {data_subset_mode} not recognized')
            return
    
    def keyhole_penalty(self, origin_coords, sample_coords, dist_cutoff, dist_penalty, debug=False):
        
        # subsets are defined based on the hyperparams cdir, cstrength and close
        delta_coords = origin_coords[:, None] - sample_coords
        
        current_penalty = self.penalty_factor(delta_coords, autocovar=0, mode='bezier')
        
        eucl_dist = torch.cdist(origin_coords, sample_coords, p=2)
        eucl_dist_penalty = (eucl_dist > dist_cutoff).long() * dist_penalty
        
        close = self.params['close']
        #two = torch.Tensor([2])
        #close_penalty = 1/(0.2 + torch.exp(torch.add(close*eucl_dist, torch.exp(two))))

        close_penalty = (eucl_dist > close).long()

        if debug:
            gpt_plot.print_tensor_with_indexes(current_penalty, 'current_penalty')
            gpt_plot.print_tensor_with_indexes(close_penalty, 'close_penalty')
            gpt_plot.print_tensor_with_indexes(eucl_dist_penalty, 'eucl_dist_penalty')

        penalty = current_penalty*close_penalty + eucl_dist_penalty

        return penalty
    
    def penalty_to_subset_indices(self, penalty, threshold, debug=False):
        # Find the indices where the penalty is below the threshold
        if debug:
            gpt_plot.print_tensor_with_indexes(penalty, 'get_subsets/penalty')
        
        subset_indices = []
        for row in penalty:
            indices = (row < threshold).nonzero().squeeze().tolist()
            if type(indices) is not list:
                indices = [indices]
            subset_indices.append(indices)
        
        return subset_indices
    

    def merge_lists(self, A, B):
        # Merge corresponding sublists in A and B, and remove duplicates
        C = [list(set(a + b)) for a, b in zip(A, B)]
        return C
    

    def exclude_centers_closer_than(self, A, B, n):
        # Compute the pairwise distance matrix
        dists = torch.cdist(A, B)

        # Find which points in B are more than n away from all points in A
        mask = torch.all(dists > n, dim=0)

        # Select these points from B
        B_filtered = B[mask]

        # Concatenate A and the filtered B
        result = torch.cat((A, B_filtered), dim=0)

        return result

    def detect_peaks(self, array, n, min_peak_distance=0, min_region_distance=0, debug=False):
        # This function should receive a smoothed array, as the
        # peak detection is vulnerable to outlier samples

        # Calculate the 95th percentile as the threshold
        #threshold = np.percentile(array, 95)
        #threshold = max((np.std(array) + np.mean(array)), self.params['peak_concentration_cutoff'])
        threshold = self.params['peak_concentration_cutoff']

        # Identify regions of the array that are above the threshold
        is_above_threshold = array > threshold
        diff = np.diff(is_above_threshold.astype(int))
        starts = np.where(diff > 0)[0] + 1  # +1 to point to start of region, not end of previous region
        ends = np.where(diff < 0)[0] + 1  # +1 because diff shifts everything to the left

        # If array starts or ends with a peak, diff won't catch it
        if is_above_threshold[0]:
            starts = np.insert(starts, 0, 0)
        if is_above_threshold[-1]:
            ends = np.append(ends, len(array) - 1)

        # Find highest peak in each region
        peak_regions = [array[start:end] for start, end in zip(starts, ends)]
        if len(peak_regions) == 0:
            return ([], [])
        
        peak_regions = [sublist for sublist in peak_regions if len(sublist) > 0] #remove empty sublists
        
        peak_indices = [region.argmax() + start for start, region in zip(starts, peak_regions)]
        peak_regions_indices = [list(range(start, end)) for start, end in zip(starts, ends)]
        peak_regions_indices = [sublist for sublist in peak_regions_indices if len(sublist) > 0] #remove empty sublists
        #print(f'detect_peak peak_regions: {peak_region_indices}')

        # Sort the peaks in descending order by their value
        #peaks = np.array(peak_indices)[np.argsort(-array[peak_indices])]
        peaks = np.array(peak_indices)
        n_peaks = len(peaks)
        if debug:
            print(f'Found {n_peaks} peaks at indices {peaks}')
            print(f'Found {len(peak_regions_indices)} peak_regions')

        # This merging of peaks simply does so two peaks close to each other
        # is treated as one peak between the two actual peaks. The regions of
        # samples beloning to each peak are merged. If you do not care about
        # peaks as training centres, it is probably better to use the 
        # min_region_distance parameter to merge regions below.
        if min_peak_distance:
            merged_peak_indices = []
            merged_peak_regions_indices = []
            i = 0
            while i < n_peaks:
                if (i < n_peaks - 1 and abs(self.dist(self.sample_coords[peaks[i+1]], self.sample_coords[peaks[i]])) < min_peak_distance):
                    loc1 = self.sample_coords[peaks[i+1]]
                    loc2 = self.sample_coords[peaks[i]]
                    #print(f'i: {i} - dist: {abs(self.dist(loc1, loc2))}')
                    new_peak = abs(self.sample_coords - self.midway(loc1, loc2)).argmin().item()  # Compute new peak midway between the two
                    merged_peak_indices.append(new_peak)
                    merged_peak_regions_indices.append(peak_regions_indices[i]+peak_regions_indices[i+1])
                    i += 2  # Skip the next peak since it has been merged
                else:
                    merged_peak_indices.append(peaks[i])
                    merged_peak_regions_indices.append(peak_regions_indices[i])
                    i += 1
            peak_indices = merged_peak_indices
            peak_regions_indices = merged_peak_regions_indices
        
        # This merging of regions is meant to merge regions separated by only a 
        # few samples/meters. Merging over distances should be handled later.
        if min_region_distance:
            found = True
            while found == True:
                found = False
                temp_regions_indices = []
                i = 0
                while i < len(peak_regions_indices[:-1]):
                    if debug:
                        print(f'len(peak_regions_indices[:-1]) = {len(peak_regions_indices[:-1])}')
                    start_cur_idx = peak_regions_indices[i][0]
                    end_cur_idx = peak_regions_indices[i][-1] # not used later
                    end_cur = self.sample_coords[peak_regions_indices[i][-1]]
                    start_next_idx = peak_regions_indices[i+1][0] # not used later
                    start_next = self.sample_coords[peak_regions_indices[i+1][0]]
                    end_next_idx = peak_regions_indices[i+1][-1]
                    distance = torch.linalg.vector_norm(end_cur - start_next)
                    if debug:
                        print(f'i: {i}, start_cur_idx {start_cur_idx} end cur_idx {end_cur_idx} start_next_idx {start_next_idx} end_next_idx {end_next_idx}')
                        print(f'end cur {end_cur} start_next {start_next} distance: {distance}')

                    if distance < min_region_distance:
                        if debug:
                            print(f'Merging regions...')
                        temp_regions_indices.append(list(range(start_cur_idx, end_next_idx+1)))
                        found = True
                        if i == len(peak_regions_indices) - 3:
                            temp_regions_indices.append(peak_regions_indices[-1])
                            if debug:
                                print(f'found, appending peak_regions_indices[-1]')
                        i += 2
                    else:
                        temp_regions_indices.append(peak_regions_indices[i])
                        if i == len(peak_regions_indices) - 2:
                            temp_regions_indices.append(peak_regions_indices[-1])
                            if debug:
                                print(f'not found, appending peak_regions_indices[-1]')
                        i += 1

                if len(peak_regions_indices) > 1:
                    peak_regions_indices = temp_regions_indices

        # Return the indices of the n highest peaks
        return (list(peak_indices[:n]), peak_regions_indices[:n])
    
    def dist(self, pointA, pointB):
        if (pointA.ndim <= 1) and (pointB.ndim <= 1):
            return pointB - pointA
        elif (pointA.ndim == 2) and (pointB.ndim == 2):
            return torch.cdist(pointA, pointB, p=2)
        else:
            print(f'Error: Wrong number of ndims in distance computation')
            print(f'pointA: {pointA} pointB: {pointB}')
        return
    
    def midway(self, pointA, pointB):
        if (pointA.ndim <= 1) and (pointB.ndim <= 1):
            return (pointB + pointA)/2
        elif (pointA.ndim == 2) and (pointB.ndim == 2):
            diff = pointB - pointA
            return pointB - diff/2
        else:
            print(f'Error: Wrong number of ndims in midway computation')
        return

    def is_point_close_to_segment(self, pt1, pt2, pt_test, tolerance):
        # Unpack the points
        x1, y1 = pt1
        x2, y2 = pt2
        x3, y3 = pt_test

        # Compute the numerator and the denominator of the distance formula
        numerator = np.abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Calculate the distance
        distance = numerator / denominator

        # Calculate the dot product to determine if the point falls on the line segment
        dot_product = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (denominator ** 2)
        
        # Return whether the distance is less than the tolerance and the point falls on the line segment
        return distance < tolerance and 0 <= dot_product <= 1
    

    def select_elements(self, tensor, x, n):
        # Reverse the tensor, select elements from the start to x (now at the end), and reverse again
        first_part = tensor[:x+1].flip(dims=[0])[n::n].flip(dims=[0])

        # Select elements from x to the end of the tensor, stepping by n
        second_part = tensor[x+n::n]

        # Join the two parts
        result = torch.cat((first_part, second_part), dim=0)

        return result
    

    def penalty_factor(self, delta_coords, autocovar, mode='bezier'):
        '''params is a list of [noise, cdir, cstrength, dilution, close] '''

        x1 = delta_coords[:, :, 0]
        y1 = delta_coords[:, :, 1]

        angle1 = torch.atan2(y1, x1)*180/torch.pi
        angle2 = self.params['current_direction'] # already in deg
        if autocovar:
            angle1_added = torch.add(angle1, torch.eye(angle1.size()[0])*angle2)
        else:
            angle1_added = angle1

        distance = shortest_angular_distance(angle1_added, angle2)
        cstrength = self.params['current_strength']

        if mode == 'bezier':
            return self.penalty_bezier(distance, (cstrength-1)/19) # Normalize for Bezier
        elif mode == 'exp':
            return self.penalty_exp(distance, cstrength)
        elif mode == 'scalar_step':
            return self.penalty_linear(x1, y1, distance, cstrength)
        else:
            print(f'Error: Mode {mode} not found')
        
        return torch.zeros_like(distance)

    def opposite_angle(self, angle):
        # Subtract 180 from the angle, which will give the opposite direction
        opposite = angle - 180

        # If the result is less than -180, add 360 to bring it within the [-180, 180] range
        if opposite < -180:
            opposite += 360

        return opposite

    def new_bezier(self, points, t):

        p = (1-t)**3 * points[0][0]
        p += 3 * (1-t)**2 * t * points[1][0]
        p += 3 * (1-t) * t**2 * points[2][0]
        p += t**3 * points[3][0]
        
        return p

    def penalty_bezier(self, distance, scalar):
        scalar_min = 0; scalar_max = 1
        P1_min = -1; P1_max = 9
        
        # compute P1_x [0, 8] from scalar (cstrength) [0, 1]
        # higher P1_x means more punishment of small angle-distances
        P1_x = (scalar-scalar_min)/(scalar_max-scalar_min)*(P1_max-P1_min) + P1_min
        P1 = [P1_x, 7]

        points = torch.Tensor([[0, 0], P1, [10, 3], [10, 10]])
        
        #use clamp and divide by 90 deg to normalize to [0, 1] for bezier function
        bezier = self.new_bezier(points, distance.clamp(0, 90)/90)
        
        return bezier
        
    def penalty_exp(self, distance, scalar):
        penalty = 0.5*scalar*distance**2
        return penalty

    def penalty_linear(self, x1, y1, distance, scalar):
        penalties = torch.zeros_like(distance)

        for i, row in enumerate(distance):
            for j, d in enumerate(row):
                if x1[i, j] == 0 and y1[i, j] == 0:
                    penalties[i, j] = 0
                if d < scalar:
                    penalties[i, j] = 0
                else:
                    penalties[i, j] = d - scalar
        
        return penalties
    
    def visualize_loss(self, array=[], subplot_height=600):

        if array == []:
            array = range(len(self.models))
        
        figs = []
        for idx in array:
            figs.append(self.models[idx].visualize_loss())
        
        subplot_titles = [f'Model {i+1}' for i in range(len(figs))]
        
        cols = 3
        rows = len(figs)//cols + 1
        matrix_fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        
        for idx, fig in enumerate(figs):
            for trace in fig.data:
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                matrix_fig.add_trace(trace, row=row, col=col)

        subplot_width = 500; subplot_height = 500
        matrix_fig.update_layout(title_text="Training loss over time", height=subplot_height*(rows-2))
        matrix_fig.show()

def normalize_coords(coords):

    min_dim_0 = min(coords[:, 0])
    max_dim_0 = max(coords[:, 0])
    min_dim_1 = min(coords[:, 1])
    max_dim_1 = max(coords[:, 1])
    trans = torch.Tensor([min_dim_0, min_dim_1])

    scale_dim_0 = max_dim_0 - min_dim_0
    scale_dim_1 = max_dim_1 - min_dim_1
    scale = torch.Tensor([scale_dim_0, scale_dim_1])

    norm_coords = coords - trans
    norm_coords /= scale

    return norm_coords

def rotate_and_normalize_coords(coords, degrees):
    # Convert degrees to radians
    radians = -math.radians(degrees)  # Negative sign for clockwise rotation
    
    # Create a rotation matrix
    rotation_matrix = torch.tensor([[math.cos(radians), -math.sin(radians)],
                                    [math.sin(radians), math.cos(radians)]])    
    # Rotate the coordinates
    rotated_coords = torch.mm(coords, rotation_matrix.T)  # Transpose rotation_matrix to align dimensions for matrix multiplication
    min_x = min(rotated_coords[:, 0])
    min_y = min(rotated_coords[:, 1])
    offset_coords = rotated_coords - torch.Tensor([min_x, min_y])
    
    scale_x = max(offset_coords[:, 0])
    scale_y = max(offset_coords[:, 1])
    normalized_coords = offset_coords/torch.Tensor([scale_x, scale_y])

    return normalized_coords

def rotate_and_normalize_coords_v2(coords, degrees, params=None):
    # Convert degrees to radians
    radians = -math.radians(degrees)  # Negative sign for clockwise rotation
    
    # Create a rotation matrix
    rotation_matrix = torch.tensor([[math.cos(radians), -math.sin(radians)],
                                    [math.sin(radians), math.cos(radians)]])
    # Rotate the coordinates
    rotated_coords = torch.mm(coords, rotation_matrix.T)  # Transpose rotation_matrix to align dimensions for matrix multiplication
    
    if params is None: # This is the sample coords
        min_x = min(rotated_coords[:, 0])
        min_y = min(rotated_coords[:, 1])
        offset_coords = rotated_coords - torch.Tensor([min_x, min_y])
        
        scale_x = max(offset_coords[:, 0])
        scale_y = max(offset_coords[:, 1])
    else: # This is the test coords
        min_x, min_y, scale_x, scale_y = params
        offset_coords = rotated_coords - torch.Tensor([min_x, min_y])

    normalized_coords = offset_coords/torch.Tensor([scale_x, scale_y])

    return normalized_coords, (min_x, min_y, scale_x, scale_y)

def rotate_coords(coords, degrees):
    # Convert degrees to radians
    radians = -math.radians(degrees)  # Negative sign for clockwise rotation
    
    # Create a rotation matrix
    rotation_matrix = torch.tensor([[math.cos(radians), -math.sin(radians)],
                                    [math.sin(radians), math.cos(radians)]])    
    
    #offset_dim_0 = int((max(coords[:, 0]) - min(coords[:, 0]))/2)
    #offset_dim_1 = int((max(coords[:, 1]) - min(coords[:, 1]))/2)
    offset_dim_0 = (max(coords[:, 0]) - min(coords[:, 0]))/2 + min(coords[:, 0])
    offset_dim_1 = (max(coords[:, 1]) - min(coords[:, 1]))/2 + min(coords[:, 1])
    #offset_dim_0 = coords[:, 0].mean()
    #offset_dim_1 = coords[:, 1].mean()
    offset = torch.Tensor([offset_dim_0, offset_dim_1])
    
    offset_coords = coords - offset
    
    # Rotate the coordinates
    rotated_coords = torch.mm(offset_coords, rotation_matrix.T)  # Transpose rotation_matrix to align dimensions for matrix multiplication
    rotated_coords += offset

    return rotated_coords

def normalize_values(values):
    norm_values = (values - values.min()) / (values.max() - values.min())
    #norm_values = values.sub(values.min()) # adjust bias to zero
    #norm_values = norm_values.div(norm_values.max()) # scale to [0, 1]
    return norm_values

def denormalize_values(values, sample_values):
    denorm_values = values * (sample_values.max() - sample_values.min()) + sample_values.min()
    #denorm_values = values.mul(sample_values.max()) # rescale to sample max
    #denorm_values = denorm_values.add(sample_values.min()) # adjust bias to sample min
    return denorm_values

def shortest_angular_distance_torch(angle1, angle2):
    ''' The function takes two angles, angle1 and angle2, and returns the 
        shortest angular distance between them. The calculation of delta 
        represents the difference between the two angles, which is then 
        normalized to the interval [-180, 180] using the modulo and subtraction
        operations. This ensures that the result of the function is the 
        shortest angular distance between the two input angles. '''

    delta = angle1.unsqueeze(0) - angle2.unsqueeze(1)
    delta = (delta + 180) % 360 - 180
    return delta.abs()


def shortest_angular_distance(angle1, angle2):
    delta = angle2 - angle1
    delta = (delta + 180) % 360 - 180
    return abs(delta)


def compute_distances_differences_and_angles(coordinates, scalars=None):
    """
    Computes the distances between all 2D coordinates, differences between corresponding scalars,
    and angles between all pairs of coordinates.

    Args:
    - coordinates (torch.Tensor): Tensor of shape (n, 2) containing 2D coordinates.
    - scalars (torch.Tensor): Tensor of shape (n,) containing scalars related to each coordinate.

    Returns:
    - combined (torch.Tensor): Tensor of shape (n * n, 4) containing starting point, distances, and angles between all pairs of coordinates.
    - differences (torch.Tensor): Flattened tensor containing differences between corresponding scalars.
    """
    n = coordinates.shape[0]

    # Ensure the input is a float tensor
    #coordinates = coordinates.float()
    #scalars = scalars.float()

    # Expand the coordinates to compute the pairwise differences
    diff_coords = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
    #print("diff_coords:", diff_coords)
    distances = torch.sqrt((diff_coords ** 2).sum(dim=-1))
    #print("distances:", distances)

    # Compute the angles between the coordinates
    angles = torch.atan2(diff_coords[..., 1], diff_coords[..., 0]) * (180 / torch.tensor(3.14159265))

    # Expand the coordinates and repeat them along the second dimension
    starting_points = coordinates.unsqueeze(1).repeat(1, n, 1)

    # Combine starting points, distances, and angles, and reshape to (n * n, 4)
    combined = torch.cat((starting_points, distances.unsqueeze(-1), angles.unsqueeze(-1)), dim=-1).view(-1, 4)

    if scalars != None:
        # Expand the scalars to compute the pairwise differences
        differences = scalars.unsqueeze(0) - scalars.unsqueeze(1)
        
        # Flatten the differences tensor
        differences = differences.flatten()
    else:
        differences = scalars

    return combined, differences


# %%
# BMA for 2D
def two_dimensional_mll(test_coords, sample_coords, sample_coords_rot, sample_values, window_value, models, method='euclidean', cdir=None, cstr=None):
    num_coords = len(test_coords)
    p_y_gvn_m0 = torch.empty(num_coords).detach()
    p_y_gvn_m1 = torch.empty(num_coords).detach()
    models[0].eval()
    models[1].eval()
    
    sample_coords_exp = sample_coords.unsqueeze(0)
    mmK0_mml = gpytorch.mlls.ExactMarginalLogLikelihood(models[0].likelihood, models[0])
    mmK1_mml = gpytorch.mlls.ExactMarginalLogLikelihood(models[1].likelihood, models[1])
    #mmK0_mml = gpytorch.mlls.LeaveOneOutPseudoLikelihood(models[0].likelihood, models[0])
    #mmK1_mml = gpytorch.mlls.LeaveOneOutPseudoLikelihood(models[1].likelihood, models[1])

    distances = torch.norm(test_coords.unsqueeze(1) - sample_coords_exp, dim=2)

    if method == 'elliptic':
        dist_elliptic = distances_ellipsis(distances, test_coords, sample_coords, cdir, cstr)
        nearest = dist_elliptic.topk(window_value, largest=False, dim=1) # The window_value smallest elements
    elif method == 'euclidean':
        nearest = distances.topk(window_value, largest=False, dim=1) # The window_value smallest elements
    else:
        print(f'method {method} unknown')
        return ([], [], [])

    use_coords = sample_coords_rot[nearest.indices]
    use_c = sample_coords[nearest.indices].tolist()
    use_values = sample_values[nearest.indices]
    
    #use_c = []
    n = 100
    num_batches = int(num_coords/n)
    percentage = 10
    p = round(num_batches/percentage)
    print_number = 0
    c = 0
    if num_coords%n == 0:
        num_runs = num_batches
    else:
        num_runs = num_batches + 1

    for i in range(num_runs):
        c += 1
        if c >= p:
            c = 0
            print_number += percentage
            print(f'...{print_number}%', end='')
        if i == int(num_batches):
            # remaining samples (modulo)
            #test_coords_batch = test_coords[i*n:]
            use_coords_batch = use_coords[i*n:]
            use_values_batch = use_values[i*n:]
        else:
            #test_coords_batch = test_coords[i*n:(i+1)*n]
            use_coords_batch = use_coords[i*n:(i+1)*n]
            use_values_batch = use_values[i*n:(i+1)*n]
        
        #distances = torch.norm(test_coords_batch.unsqueeze(1) - sample_coords_exp, dim=2)
        
        #if method == 'elliptic':
        #    dist_elliptic = distances_ellipsis(distances, test_coords_batch, sample_coords, cdir, cstr)
        #    nearest = dist_elliptic.topk(window_value, largest=False, dim=1) # The window_value smallest elements
        #elif method == 'euclidean':
        #    nearest = distances.topk(window_value, largest=False, dim=1) # The window_value smallest elements
        #else:
        #    print(f'method {method} unknown')
        #    return
        
        #use_coords = sample_coords_rot[nearest.indices]
        #use_c = use_c + sample_coords[nearest.indices].tolist()
        #use_values = sample_values[nearest.indices]
        #print(f'i: {i} use_coords: {use_coords}')
        #print(f'i: {i} use_values: {use_values}')

        out0 = models[0](use_coords_batch)
        out1 = models[1](use_coords_batch)

        #print(f'use_coords_batch: {use_coords_batch}')
        #print(f'use_c: {use_c[i*n:(i+1)*n][0]}')

        p_y_gvn_m0_batch = mmK0_mml(out0, use_values_batch).exp()
        #print(f'out0.mean: {out0.mean} out0.conf {out0.confidence_region()} use_values_batch: {use_values_batch}')
        #print(f'use_coords_batch: {use_coords_batch}')
        #print(f'use_c: {use_c[i*n:(i+1)*n][0]}')

        #print(f'p_y_gvn_m0_batch: {p_y_gvn_m0_batch}')
        #print(f'p_y_gvn_m1_batch: {p_y_gvn_m1_batch}')
        #print(f'covar_matrix: {covar_matrix}')
        #print(f'K: {K}')
        #print(f'covar_matrix is symmetrix: {np.all(covar_matrix-covar_matrix.T==0)}')
        #print(f'K is symmetrix: {np.all(K-K.T==0)}')
        #print(f'covar_matrix eigvals: {np.linalg.eigvals(covar_matrix)}')
        #print(f'K eigvals: {np.linalg.eigvals(K)}')
        
        p_y_gvn_m1_batch = mmK1_mml(out1, use_values_batch).exp()

        #print(f'out1: {out1} use_values_batch: {use_values_batch}')
        #print(f'use_coords_batch: {use_coords_batch}')
        #print(f'use_c: {use_c[i*n:(i+1)*n][0]}')

        #nlpd0 = gpytorch.metrics.negative_log_predictive_density(out0, use_values_batch)
        #nlpd1 = gpytorch.metrics.negative_log_predictive_density(out1, use_values_batch)
        #print(f'nlpd0: {nlpd0}')
        #print(f'nlpd1: {nlpd1}')

        #with torch.no_grad():
        #    covar_matrix0 = models[0].covar_module(use_coords_batch).evaluate()
        #    noise0 = models[0].likelihood.noise
        #    K0 = covar_matrix0 + torch.eye(use_coords_batch.size(0))*noise0

        #    covar_matrix1 = models[1].covar_module(use_coords_batch).evaluate()
        #    noise1 = models[1].likelihood.noise
        #    K1 = covar_matrix1 + torch.eye(use_coords_batch.size(0))*noise1

        #    p_y_gvn_m0_batch_EMLL = exact_log_marginal_likelihood((use_values_batch - out0.loc).numpy(), K0.numpy())
        #    p_y_gvn_m1_batch_EMLL = exact_log_marginal_likelihood((use_values_batch - out1.loc).numpy(), K1.numpy())
        
        p_y_gvn_m0[i*n:(i+1)*n] = p_y_gvn_m0_batch.detach()
        p_y_gvn_m1[i*n:(i+1)*n] = p_y_gvn_m1_batch.detach()
        #p_y_gvn_m0_EMLL[i*n:i*n+n] = p_y_gvn_m0_batch_EMLL
        #p_y_gvn_m1_EMLL[i*n:i*n+n] = p_y_gvn_m1_batch_EMLL
        #print(f'p_y_gvn_m0_batch: {p_y_gvn_m0_batch}, p_y_gvn_m1_batch: {p_y_gvn_m1_batch}')
        #print(f'p_y_gvn_m0_bEMLL: {p_y_gvn_m0_batch_EMLL}, p_y_gvn_m1_bEMLL: {p_y_gvn_m1_batch_EMLL}')
        
    print('')

    #return (p_y_gvn_m0.detach(), p_y_gvn_m1.detach(), p_y_gvn_m0_EMLL, p_y_gvn_m1_EMLL, use_c)
    return (p_y_gvn_m0, p_y_gvn_m1, use_c)

def exact_log_marginal_likelihood(y, K):
    """
    Compute the Exact Log Marginal Likelihood.
    
    Parameters:
    - y (numpy.array): An array containing the training targets. Shape (n, )
    - K (numpy.array): The covariance matrix. Shape (n, n)
    
    Returns:
    - float: The Exact Log Marginal Likelihood.

    Called like this:
    with torch.no_grad():
            covar_matrix0 = models[0].covar_module(use_coords_batch).evaluate()
            noise0 = models[0].likelihood.noise
            K0 = covar_matrix0 + torch.eye(use_coords_batch.size(0))*noise0

            covar_matrix1 = models[1].covar_module(use_coords_batch).evaluate()
            noise1 = models[1].likelihood.noise
            K1 = covar_matrix1 + torch.eye(use_coords_batch.size(0))*noise1

            p_y_gvn_m0_batch = exact_log_marginal_likelihood(use_values_batch.numpy(), K0.numpy())
            p_y_gvn_m1_batch = exact_log_marginal_likelihood(use_values_batch.numpy(), K1.numpy())
            print(f'p_y_gvn_m0_batch: {p_y_gvn_m0_batch}')
            print(f'p_y_gvn_m1_batch: {p_y_gvn_m1_batch}')

    """
    # Ensure K is a square matrix
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
        
    # Number of training points
    n = len(y)
    
    # Inverse of K
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix K is not invertible.")
    
    # Compute the log determinant
    log_det = np.linalg.slogdet(K)[1]
    #print(f'log_det: {log_det}')
    #print(f'y.T @ K_inv @ y: {y.T @ K_inv @ y}')
    
    # Compute the Exact Log Marginal Likelihood
    ELML = -0.5 * (y.T @ K_inv @ y + log_det + n * np.log(2 * np.pi))
    
    return ELML

def get_r_foci(e, th):
    a = 1.0
    #th = th*torch.pi/180
    #print(f'a: {a} b: {math.sqrt((a**2)*(1-e**2))} c: {a*e} th: {th*180/math.pi}')
    return a*(1 - e**2)/(1+e*torch.cos(th))

def distances_ellipsis(distances, test_coords, sample_coords, cdir, cstr):

    sample_coords_exp = sample_coords.unsqueeze(0)
    test_coords_exp = test_coords.unsqueeze(1)

    dy = sample_coords_exp[:, :, 1] - test_coords_exp[:, :, 1]
    dx = sample_coords_exp[:, :, 0] - test_coords_exp[:, :, 0]
    
    rel_angles = torch.atan2(dy, dx)
    angles_rot = rel_angles - cdir/180*math.pi #[-pi-cdir, pi-cdir]
    # Normalize the angle to be within [-pi, pi]
    angles_rot = (angles_rot + torch.pi) % (2 * torch.pi) - torch.pi

    e = max((cstr-1)/39, 0.9) # compute e from cstr
    
    rs = get_r_foci(e, angles_rot)
    normalizer = (1.0 + e)/rs
    return distances*normalizer

