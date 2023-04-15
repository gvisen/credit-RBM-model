import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import datetime

def get_tail_from_pds(ts, pds, n_samples = 10000, n_runs = 100, confidence_level=0.95):
    """Estimate tail function of relative total portfolio losses from dataset of default probabilities.

    The tail function is estimated by Monte Carlo simulation. `n_samples` of losses
    are simulated from a dataset of default probabilities. Recovery rates are assumed to be
    iid Beta(4,4) random variables. 
    The function returns upper and lower bounds with prescribed confidence level using
    non-parametric bootstrap.

    Parameters:
    ----------
    ts : 1d numpy.ndarray
        Grid for tail function estimation.
        NB: The grid must be in [0, 1], since we are estimating *relative* portfolio losses.
    pds : 2d numpy.ndarray, size=(n_samples, n_obligors)
        Array of default probabilities.
    n_samples : int (default: 10000)
        Number of Monte Carlo simulated losses.
    n_runs : int (default: 100)
        Number of bootstrap samples.
        NB: increasing this parameter may substantially slow down computations.
    confidence_level : float (default: 0.95)
        Confidence level for bootstrap confidence interval.

    Returns:
    --------
    mean_value : numpy.ndarray, shape = (len(ts),)
        Estimator of tail function.
    upper_bound : numpy.ndarray, shape = (len(ts),)
        Upper bound of confidence band.
    lower_bound : numpy.ndarray, shape = (len(ts),)
        Lower bound of confidence band.
    """
    
    n_obligors = pds.shape[1]
    
    tail_fncs = np.zeros((n_runs, len(ts)))

    for i_run in range(n_runs):
        sampled_pds = pds[np.random.choice(range(pds.shape[0]), size=(n_samples,)), :]
        LGD = np.random.beta(4., 4., size=(n_samples, n_obligors))
        losses = np.sum(np.random.binomial(1, p=sampled_pds)*LGD, axis=1)/n_obligors
        tail_fncs[i_run, :] = np.array([np.mean(losses > t) for t in ts])
    
    upper_bound = np.quantile(tail_fncs, confidence_level, axis=0)
    lower_bound = np.quantile(tail_fncs, 1 - confidence_level, axis=0)
    mean_value = np.mean(tail_fncs, axis=0)
    
    # indices of at least one zero value
    zeroes_indices = np.logical_or(upper_bound == 0, lower_bound == 0, mean_value == 0)
    upper_bound[zeroes_indices] = 0.
    lower_bound[zeroes_indices] = 0.
    mean_value[zeroes_indices] = 0.
    
    return mean_value, upper_bound, lower_bound

class toy_dataset(Dataset):
    """Implements FRB stress test dataset.
    
    The data source is in 'data/dataset.csv'.
    This dataset subclasses the Dataset class in PyTorch.
    
    Attributes:
    -----------
    raw_data : torch.Tensor, shape = (n_samples, n_features)
        Raw (i.e. unnormalized) dataset values.
    data : torch.Tensor, shape = (n_samples, n_features)
        Dataset values (normalized).
    dates : list of datetime.datetime objects
        Dataset observation dates.
    n_features : int
        Number of firms in dataset.
    feature_names : list of strings
        Tickers of firms in dataset.
    min_vals : torch.Tensor, shape = (n_features,)
        Minimum features range. Used to rescale data using `normalize()` method.
    max_vals : torch.Tensor, shape = (n_features,)
        Maximum features range. Used to rescale data using `normalize()` method.

    Methods:
    --------
    to(device)
        Move dataset to device.
    get_device()
        Return device on which dataset is stored.
    normalize(data[, inverse])
        Scale data in [0,1] using dataset's min_vals and max_vals.
    """
    
    def __init__(self):
        
        # Load dataset as pandas DataFrame object
        df = pd.read_csv('data/dataset.csv', sep=';')
        
        tickers = df.columns[1:].values
               
        self.dates = self.dates = [datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10])) for date in df['dates'].values]
        self.n_features = len(tickers)
        self.feature_names = list(tickers)
        
        ### Fixing min and max values for dataset.
        # We fix range manually as [0.75 of min value, 1.25 of max value]
        
        self.raw_data = torch.from_numpy(df[tickers].values).detach().clone()
        
        self.min_vals = self.raw_data.min(axis=0).values*0.75

        if float(torch.__version__[:3]) <= 1.6:
            self.max_vals = torch.min(self.raw_data.max(axis=0).values*1.25, torch.tensor([1.]).double())
        else:
            self.max_vals = torch.minimum(self.raw_data.max(axis=0).values*1.25, torch.tensor([1.]))
        
        self.data = self.normalize(self.raw_data)
        
        
    def to(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            print('Cuda not available: dataset is kept on CPU. Training may be slow.')
        else:
            self.data = self.data.to(device)
            
    def get_device(self):
        return self.data.device.type
        
        
    def normalize(self, data, inverse=False):
        """Normalize data so that each feature is in [0,1].
        
        Normalized data can be used for training RBMs (instead of binarizing).
        This data pre-processing transformation is stored as a dataset method 
        because the feature range is a dataset-dependent choice. The choice of
        range is a delicate issue because it influences learning efficiency
        and the range of possible conditional queries once the model is trained.
        
        Parameters:
        -----------
        data : torch.Tensor
            Data to be scaled.
        inverse : bool (default: False)
            If False, it scales the data using the `min_vals` and `max_vals` attributes.
            If True, it performs the inverse operation of scaling the data.
            
        Returns:
        --------
        normalized_data : torch.Tensor
            Scaled data.        
        """        
        
        ### Input checking
        # If data tensor is 1d, cast it to 2d
        if not inverse:
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if not torch.is_tensor(data):
                raise ValueError('Data must be torch tensor.')

            with np.errstate(invalid='ignore'): # Supress Runtime warnings for nan comparisons
                if np.logical_and(data < self.min_vals, torch.logical_not(torch.isnan(data))).any():
                    raise ValueError('Values to be normalized are out of range.')
                elif np.logical_and(data > self.max_vals, torch.logical_not(torch.isnan(data))).any():
                    raise ValueError('Values to be normalized are out of range.')            
                elif data.shape[1] != self.n_features:
                    raise ValueError('Values to be normalized have wrong size.')
                else:
                    pass
        
        ### Transforming data.
        if inverse:
            normalized_data = data*(self.max_vals - self.min_vals) + self.min_vals
        else:
            normalized_data = (data - self.min_vals)/(self.max_vals - self.min_vals)
        
        return normalized_data
        
        
    def __len__(self):
        
        return self.data.shape[0]
    
    def __getdataset__(self):
        
        return self.__getitem__(range(self.__len__()))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx,:]

        return sample