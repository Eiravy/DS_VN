
#______________________________________PEP8____________________________________
#_______________________________________________________________________
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score
import shutil
import time
# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all TF warnings

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class DShap(object):
    
    def __init__(self, X, y, X_test, y_test, num_test, sources=None, 
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            overwrite: Delete existing data and start computations from 
                scratch
            **kwargs: Arguments of the model
        """
            
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family == 'logistic':
            self.hidden_units = []
        if self.model_family == 'vyngoc':
            self.hidden_units = []
        if self.directory is not None:
            if overwrite and os.path.exists(directory):
                for i in range(5):
                    try:
                        shutil.rmtree(directory)
                        break
                    except Exception:
                        time.sleep(1)            
#             if overwrite and os.path.exists(directory):
#                 tf.io.gfile.rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)  
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, 
                                      sources, sample_weight)
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        is_regression = (np.mean(self.y//1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
            
    def _initialize_instance(self, X, y, X_test, y_test, num_test, 
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""      
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            self._load_dataset(data_dir)
        else:
            self.X_heldout = X_test[:-num_test]
            self.y_heldout = y_test[:-num_test]
            self.X_test = X_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.X, self.y, self.sources = X, y, sources
            self.sample_weight = sample_weight
            data_dic = {'X': self.X, 'y': self.y, 'X_test': self.X_test,
                     'y_test': self.y_test, 'X_heldout': self.X_heldout,
                     'y_heldout':self.y_heldout, 'sources': self.sources}
            if sample_weight is not None:
                data_dic['sample_weight'] = sample_weight
                warnings.warn("Sample weight not implemented for G-Shapley")
            pkl.dump(data_dic, open(data_dir, 'wb'))        
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        n_points = len(self.X)
        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number,
            n_points, n_sources, self.model_family)
        
    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                   n_points, n_sources, model_family):
        tmc_dir = os.path.join(
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory, 
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        if model_family not in ['logistic', 'NN','vyngoc']:
            return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))
        
    def _load_dataset(self, data_dir):
        '''Load the different sets of data if already exists.'''
        data_dic = pkl.load(open(data_dir, 'rb'))
        self.X_heldout = data_dic['X_heldout']
        self.y_heldout = data_dic['y_heldout']
        self.X_test = data_dic['X_test']
        self.y_test = data_dic['y_test']
        self.X = data_dic['X'] 
        self.y = data_dic['y']
        self.sources = data_dic['sources']
        if 'sample_weight' in data_dic.keys():
            self.sample_weight = data_dic['sample_weight']
        else:
            self.sample_weight = None
        
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]        
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0' 
        return tmc_number, g_number
    
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y, 
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data 
                different from test set.
            y: Labels, if valuation is performed on a data 
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')
        
    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
        print('LOO values calculated!')
        tmc_run = True 
        g_run = g_run and self.model_family in ['logistic', 'NN','vyngoc']
        while tmc_run or g_run:
            if g_run:
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
            if self.directory is not None:
                self.save_results()
            
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory, 
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )  
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))  
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance         
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.one_iteration(
                tolerance=tolerance, 
                sources=sources
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            if self.sample_weight is None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(self.X, self.y,
                              sample_weight=self.sample_weight)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model, 
                    metric=self.metric,
                    X=self.X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))): ##FIXIT
                    self.restart_model()
                    if sample_weight_batch is None:
                        self.model.fit(X_batch, y_batch)
                    else:
                        self.model.fit(
                            X_batch, 
                            y_batch,
                            sample_weight = sample_weight_batch
                        )
                    new_score = self.value(self.model, metric=self.metric)       
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', 
                warm_start=False, address=address, 
                hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                model.fit(np.zeros((0, self.X.shape[-1])), self.y)
                model.fit(self.X, self.y)
                accs.append(model.score(self.X_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(
                    iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, 
                      sources=sources, metric=self.metric, 
                      max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.X))
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        if self.sample_weight is None:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y,
                          sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in sources.keys():
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            if self.sample_weight is not None:
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            if self.sample_weight is None:
                self.model.fit(X_batch, y_batch)
            else:
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)
                
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo
    
    def _merge_parallel_results(self, key, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) 
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.X.shape[0]))
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.X))
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory, 
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory, 
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self, max_samples=None):
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        if self.model_family not in ['logistic', 'NN','vyngoc']:
            return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    
    def performance_plots(self, vals, name=None, 
                          num_plot_markers=20, sources=None):
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 25
        plt.xlim(0, 50)
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                  for val in vals]
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0, 
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys())//num_plot_markers, 1)
        )
        perfs = [self._portion_performance(
            np.argsort(vals_source)[::-1], plot_points, sources=sources)
                 for vals_source in vals_sources]
#         print(vals)
#         for i, perf in enumerate(perfs):
#             print(f"perfs[{i}]: min={perf.min()}, max={perf.max()}")
        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])[::-1]),
            plot_points, sources=sources) for _ in range(10)], 0)
        plt.plot(plot_points/len(self.X) * 100, perfs[0] * 100, 
                 '-', lw=5, ms=10, color='b')
        if len(vals)==3:
            plt.plot(plot_points/len(self.X) * 100, perfs[1] * 100, 
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals)==2:
            legends = ['TMC-Shapley ', 'LOO', 'Random']
        else:
            legends = ['TMC-Shapley ', 'Random']
        plt.plot(plot_points/len(self.X) * 100, perfs[-1] * 100, 
                 '-.', lw=5, ms=10, color='g')
        plt.plot(plot_points/len(self.X) * 100, rnd * 100, 
                 ':', lw=5, ms=10, color='r')    
        plt.legend(legends)
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
            
    def _portion_performance(self, idxs, plot_points, sources=None):
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            keep_idxs = np.concatenate([sources[idx] for idx 
                                        in idxs[plot_points[i-1]:]], -1)
            X_batch, y_batch = self.X[keep_idxs], self.y[keep_idxs]
            if self.sample_weight is not None:
                sample_weight_batch = self.sample_weight[keep_idxs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (self.is_regression 
                    or len(set(y_batch)) == len(set(self.y_test))):
                    self.restart_model()
                    if self.sample_weight is None:
                        self.model.fit(X_batch, y_batch)
                    else:
                        self.model.fit(X_batch, y_batch,
                                      sample_weight=sample_weight_batch)
                    scores.append(self.value(
                        self.model,
                        metric=self.metric,
                        X=self.X_heldout,
                        y=self.y_heldout
                    ))
                else:
                    scores.append(init_score)
        return np.array(scores)[::-1]
    ###################VYNGOC ADDING
    def performance_plots_low_rev(self, vals, name=None, 
                          num_plot_markers=20, sources=None):
        """Plots the effect of removing LOW valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 25
        plt.xlim(0, 50)
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                  for val in vals]
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0, 
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys())//num_plot_markers, 1)
        )
        perfs = [self._portion_performance(
            np.argsort(vals_source), plot_points, sources=sources)
                 for vals_source in vals_sources]
        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])),
            plot_points, sources=sources) for _ in range(10)], 0)
        plt.plot(plot_points/len(self.X) * 100, perfs[0] * 100, 
                 '-', lw=5, ms=10, color='b')
        if len(vals)==3:
            plt.plot(plot_points/len(self.X) * 100, perfs[1] * 100, 
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals)==2:
            legends = ['TMC-Shapley ', 'LOO', 'Random']
        else:
            legends = ['TMC-Shapley ', 'Random']
        plt.plot(plot_points/len(self.X) * 100, perfs[-1] * 100, 
                 '-.', lw=5, ms=10, color='g')
        plt.plot(plot_points/len(self.X) * 100, rnd * 100, 
                 ':', lw=5, ms=10, color='r')    
        plt.legend(legends)
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
            
    def performance_plots_low_rev_real(self, vals, name=None, num_plot_markers=20, sources=None):
        """Plots the effect of removing LOW valuable points first."""

        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import warnings

        plt.figure(figsize=(8, 8))
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)')

        # Ensure vals is a list
        if not isinstance(vals, (list, tuple)):
            vals = [vals]

        # Convert sources to dict
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}

        num_sources = len(sources)
        if num_sources < 2:
            print("Not enough sources to plot performance.")
            return

        # Sum values per source
        vals_sources = [np.array([np.sum(val[sources[i]]) for i in range(num_sources)]) for val in vals]

        # Determine safe plot points
        plot_points = np.linspace(0, num_sources - 1, min(num_plot_markers, num_sources), dtype=int)

        # Compute performance for low-value removal
        perfs = []
        for vals_source in vals_sources:
            idxs_sorted = np.argsort(vals_source)  # low to high
            perf = self._portion_performance_real(idxs_sorted, plot_points, sources)
            perfs.append(perf)

        # Random baseline
        rnd = np.mean([self._portion_performance_real(np.random.permutation(np.argsort(vals_sources[0])),
                                                       plot_points, sources) for _ in range(10)], axis=0)

        # Plot
        plt.plot(plot_points / len(self.X) * 100, perfs[0] * 100, '-', lw=3, color='b')
        if len(vals) >= 2:
            plt.plot(plot_points / len(self.X) * 100, perfs[1] * 100, '--', lw=3, color='orange')
        if len(vals) == 3:
            plt.plot(plot_points / len(self.X) * 100, perfs[2] * 100, '-.', lw=3, color='g')

        plt.plot(plot_points / len(self.X) * 100, rnd * 100, ':', lw=3, color='r')

        # Legends
        if len(vals) == 3:
            legends = ['TMC-Shapley', 'G-Shapley', 'LOO', 'Random']
        elif len(vals) == 2:
            legends = ['TMC-Shapley', 'LOO', 'Random']
        else:
            legends = ['TMC-Shapley', 'Random']

        plt.legend(legends)
        plt.grid(True)

        # Save
        if self.directory is not None and name is not None:
            os.makedirs(os.path.join(self.directory, 'plots'), exist_ok=True)
            plt.savefig(os.path.join(self.directory, 'plots', f'{name}.png'), bbox_inches='tight')

        plt.show()
        plt.close()

    def performance_plots_low_rev_add(self, vals, name=None, 
                              num_plot_markers=20, sources=None,
                              add_steps=[100, 200, 300, 400, 500]):
        """Plots the effect of ADDING LOW valuable points to training set."""

        plt.rcParams['figure.figsize'] = 10, 8
        plt.rcParams['font.size'] = 14
        plt.figure()
        plt.xlabel('Number of low-value points added to training set')
        plt.ylabel('Prediction accuracy on test set (%)')
        plt.grid(True, alpha=0.3)

        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]

        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}

        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                      for val in vals]

        # Baseline
        baseline_train_size = 100
        baseline_accuracy = 0.7

        # Fixed test set
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            self.X_test = self.X[-1000:]
            self.y_test = self.y[-1000:]
            self.X = self.X[:-1000]
            self.y = self.y[:-1000]

        print(f"Test set size: {len(self.X_test)}")
        print(f"Baseline: {baseline_train_size} training samples → {baseline_accuracy*100}% accuracy")
        print(f"Adding points: {add_steps}")

        # Get baseline indices (first 100 samples)
        baseline_train_indices = list(range(min(baseline_train_size, len(self.X))))

        # Train baseline model once to verify accuracy
        print("\nVerifying baseline performance...")
        self.model.fit(self.X[baseline_train_indices], self.y[baseline_train_indices])
        verified_baseline = self.model.score(self.X_test, self.y_test)
        print(f"Verified baseline accuracy: {verified_baseline:.4f} ({verified_baseline*100:.2f}%)")

        # Use verified baseline or the specified one
        actual_baseline = verified_baseline if abs(verified_baseline - baseline_accuracy) > 0.01 else baseline_accuracy

        # Calculate performances
        perfs = []

        for idx, vals_source in enumerate(vals_sources):
            print(f"\n{'='*50}")
            print(f"Calculating for method {idx}...")
            print(f"{'='*50}")
            method_perfs = [actual_baseline]  # Start with actual baseline at 0 added

            # Sort by value (lowest first)
            train_vals_source = vals_source[:len(self.X)]
            sorted_indices = np.argsort(train_vals_source)
            print(f"Sorted {len(sorted_indices)} training points by value (lowest first)")

            for n_points in add_steps:
                print(f"\n  Adding {n_points} low-value points...")
                points_to_add = sorted_indices[:n_points]
                train_indices = baseline_train_indices + list(points_to_add)
                train_indices = list(set(train_indices))

                print(f"    Training on {len(train_indices)} samples ({baseline_train_size} baseline + {len(points_to_add)} added)")

                X_train = self.X[train_indices]
                y_train = self.y[train_indices]

                # Use your model
                self.model.fit(X_train, y_train)
                accuracy = self.model.score(self.X_test, self.y_test)

                print(f"    Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                method_perfs.append(accuracy)

            perfs.append(method_perfs)

        # Random baseline
        print(f"\n{'='*50}")
        print("Calculating random baseline...")
        print(f"{'='*50}")
        rnd_perfs = []
        for trial in range(5):
            print(f"  Trial {trial+1}/5")
            trial_perfs = [actual_baseline]  # Start with same baseline
            for n_points in add_steps:
                random_indices = np.random.permutation(len(self.X))[:n_points]
                train_indices = baseline_train_indices + list(random_indices)
                train_indices = list(set(train_indices))

                X_train = self.X[train_indices]
                y_train = self.y[train_indices]

                self.model.fit(X_train, y_train)
                accuracy = self.model.score(self.X_test, self.y_test)
                trial_perfs.append(accuracy)
            rnd_perfs.append(trial_perfs)

        rnd = np.mean(rnd_perfs, 0)
        rnd_std = np.std(rnd_perfs, 0)

        # Print table
        print("\n" + "="*80)
        print("PERFORMANCE TABLE (Accuracy %)")
        print("="*80)
        print(f"{'Method':<20} {'0 added':<12} {'100':<10} {'200':<10} {'300':<10} {'400':<10} {'500':<10}")
        print("-"*80)

        method_names = ['TMC-Shapley', 'G-Shapley', 'LOO']
        for idx, method_perf in enumerate(perfs):
            if idx < len(method_names):
                values = [f"{p*100:.2f}%" for p in method_perf]
                print(f"{method_names[idx]:<20} {values[0]:<12} {values[1]:<10} {values[2]:<10} {values[3]:<10} {values[4]:<10} {values[5]:<10}")

        random_values = [f"{p*100:.2f}%" for p in rnd]
        print(f"{'Random (average)':<20} {random_values[0]:<12} {random_values[1]:<10} {random_values[2]:<10} {random_values[3]:<10} {random_values[4]:<10} {random_values[5]:<10}")
        print(f"{'Random (std)':<20} {rnd_std[0]*100:.2f}%     {rnd_std[1]*100:.2f}%    {rnd_std[2]*100:.2f}%    {rnd_std[3]*100:.2f}%    {rnd_std[4]*100:.2f}%    {rnd_std[5]*100:.2f}%")
        print("="*80)

        # Plot
        print("\n" + "="*50)
        print("GENERATING PLOT...")
        print("="*50)

        # Plot each method
        if len(perfs) >= 1:
            plt.plot(add_steps, np.array(perfs[0][1:]) * 100, 
                    '-o', lw=3, ms=8, color='b', label='TMC-Shapley')
            plt.plot(0, perfs[0][0] * 100, 'bo', markersize=10, label='_nolegend_')

        if len(perfs) >= 2:
            plt.plot(add_steps, np.array(perfs[1][1:]) * 100, 
                    '--s', lw=3, ms=8, color='orange', label='G-Shapley')
            plt.plot(0, perfs[1][0] * 100, 's', color='orange', markersize=10, label='_nolegend_')

        if len(perfs) >= 3:
            plt.plot(add_steps, np.array(perfs[2][1:]) * 100, 
                    '-.^', lw=3, ms=8, color='g', label='LOO')
            plt.plot(0, perfs[2][0] * 100, '^', color='g', markersize=10, label='_nolegend_')

        # Plot random baseline
        plt.plot(add_steps, rnd[1:] * 100, ':d', lw=3, ms=8, color='r', label='Random')
        plt.plot(0, rnd[0] * 100, 'd', color='r', markersize=10, label='_nolegend_')

        # Add baseline line
        plt.axhline(y=actual_baseline*100, color='gray', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Baseline ({baseline_train_size} samples)')

        # Add value labels on the graph for ALL methods
        # TMC-Shapley (blue)
        if len(perfs) >= 1:
            for i, (x, y) in enumerate(zip([0] + add_steps, perfs[0])):
                plt.annotate(f'{y*100:.1f}%', (x, y*100), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, 
                           fontweight='bold', color='b')

        # G-Shapley (orange)
        if len(perfs) >= 2:
            for i, (x, y) in enumerate(zip([0] + add_steps, perfs[1])):
                offset = 15 if i == 0 else 5  # Different offset for baseline
                plt.annotate(f'{y*100:.1f}%', (x, y*100), 
                           xytext=(5, offset), textcoords='offset points', fontsize=9, 
                           color='orange')

        # LOO (green)
        if len(perfs) >= 3:
            for i, (x, y) in enumerate(zip([0] + add_steps, perfs[2])):
                offset = -15 if i == 0 else -10  # Different offset for baseline
                plt.annotate(f'{y*100:.1f}%', (x, y*100), 
                           xytext=(5, offset), textcoords='offset points', fontsize=9, 
                           color='g')

        # Random (red)
        for i, (x, y) in enumerate(zip([0] + add_steps, rnd)):
            offset = -20 if i == 0 else -15
            plt.annotate(f'{y*100:.1f}%', (x, y*100), 
                       xytext=(5, offset), textcoords='offset points', fontsize=9, 
                       color='r', alpha=0.8)

        plt.xlabel('Number of low-value points added to training set', fontsize=14)
        plt.ylabel('Prediction accuracy on test set (%)', fontsize=14)
        plt.title(f'Effect of Adding Low-Value Points to Training Set\n(Baseline: {baseline_train_size} samples at {actual_baseline*100:.1f}%)', fontsize=16)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Save if needed
        if self.directory is not None and name is not None:
            os.makedirs(os.path.join(self.directory, 'plots'), exist_ok=True)
            plt.savefig(os.path.join(self.directory, 'plots', f'{name}_add_low_value.png'), 
                       bbox_inches='tight', dpi=150)
            print(f"\n✓ Plot saved to {self.directory}/plots/{name}_add_low_value.png")

        print("\n" + "="*50)
        print("COMPLETE!")
        print("="*50)
        return [0] + add_steps, perfs, rnd