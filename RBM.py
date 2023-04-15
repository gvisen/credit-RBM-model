import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class RBM(nn.Module):
    """RBM implementation."""

    def __init__(self, 
                 nv, 
                 nh, 
                 k=1,
                 training='CD',
                 dataset=None,
                 visible_normalized=False,
                 lr=0.001, 
                 lr_trend='linear',
                 bs=1024,
                 epochs=100, 
                 print_step=100,
                 verbose=True,
                 track_learning=False,
                 track_method='KDE',
                 track_step=100,
                 KDE_bandwidth=None):
        """

        Parameters:
        -----------
        nv : int
            Number of visible units.
        nh : int
            Number of hidden units.
        k : int
            Number of steps in Gibbs sampling.
        dataset : (optional) torch Dataset object
            Dataset on which to train the model.
            Optionally needed before training to correctly initialize the visible bias to log(p/(1-p)) of
            mean activation probability of the visible units, which centers the 
            model on the mean value of the visible units.
            If dataset is not passed, the visible bias is set to zero.
            If KDE is 
        visible_normalized : bool
            True if visible units take values in [0,1], or
            False if visible units take values in {0,1}.
        lr : float
            Initial learning rate.
        lr_trend : str
            'linear' if learning rate linearly decreases each epoch
            from initial value `lr` to 0.
            'constant' if learning rate remains constant across epochs.
        bs : int
            Mini-batch size.
        epochs : int
            Number of epochs (i.e. iterations through whole dataset).
        print_step : int
            Number of epochs before next print message during training.
        verbose : bool
            Set to False if you don't want print messages during training.
        track_learning : bool
            True if learning during training should be tracked.
        track_method : str or list of strings ('KDE', 'MMD', 'NDB', 'rec_error')
            Metrics(s) to be used to track learning in-sample.
        track_step : int
            Number of epochs before next learning tracking.
        """

        super(RBM, self).__init__()

        self.nv = nv # number of visible units
        self.nh = nh # number of hidden units

        self.lr = lr # learning rate
        self.lr_trend = lr_trend # learning rate trend
        self.bs = bs # batch size
        self.epochs = epochs
        self.print_step = print_step
        self.verbose = verbose
        self.track_learning = track_learning
        self.KDE_bandwidth = KDE_bandwidth
        
        if isinstance(track_method, str):
            self.track_method = [track_method]
        elif isinstance(track_method, list):
            self.track_method = track_method
        else:
            raise ValueError('Track method must be string or list of strings')
        
        for track_method in self.track_method:
            if track_method not in ['KDE', 'MMD', 'NDB', 'rec_error']:
                raise ValueError('Unknown track method:', track_method)

        self.track_step = track_step
        self.learning_curve = {track_method: [] for track_method in self.track_method}
        
        # calibrating KDE tracking method
        if self.track_learning and ('KDE' in self.track_method) and (self.KDE_bandwidth is None):
            if dataset is None:
                # force user to specify argument `dataset`
                raise ValueError('Specify dataset parameter to calibrate the KDE track method.')
            else:
                # find optimal KDE bandwidth on dataset by 5-fold cross-validation
                if self.verbose:
                    print('Calibrating KDE track method on dataset. May take a while.')
                data = dataset.data.numpy()
                np.random.shuffle(data)
    
                n_folds = 5

                bandwidths = 10**(np.linspace(-2.6, -1.7, 20))

                kf = KFold(n_splits=n_folds)

                logLs = np.zeros(len(bandwidths))

                for i_bandwidth, bandwidth in enumerate(bandwidths):
                    fold_scores = []
                    for train_indices, test_indices in kf.split(data):
                        train_data = data[train_indices, :]
                        test_data = data[test_indices, :]
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(train_data)
                        fold_scores.append(kde.score(test_data))
                    logLs[i_bandwidth] = np.mean(fold_scores)
                
                # set KDE bandwidth to optimal bandwidth
                self.KDE_bandwidth = bandwidths[np.argmax(logLs)]

                if self.verbose:
                    plt.plot(np.log(bandwidths), logLs)
                    plt.vlines(np.log(self.KDE_bandwidth), np.min(logLs), np.max(logLs), linestyles='dashed', color='black', alpha=0.2)
                    plt.title('Optimal KDE bandwidth.')
                    plt.show()

        # Whether value of visible units is activity (in [0,1]) or binary (in {0,1})
        self.visible_normalized = visible_normalized

        # Matrix initialized with a multidim normal
        self.W = torch.nn.Parameter(torch.normal(0, 0.01, size=(nh, nv)))
        # Visible bias
        if dataset is None:
            self.bv = torch.nn.Parameter(torch.zeros((1, nv)))
        else:
            # Compute mean activation probability of visible units on dataset.
            whole_dataset = dataset.__getitem__(list(range(len(dataset))))
            mean_activation = whole_dataset.mean(axis=0)
            # Set visible bias to log(p/(1-p)) where p is mean activation
            visible_bias_init = torch.log(mean_activation/(1-mean_activation)).type(torch.FloatTensor)
            self.bv = torch.nn.Parameter(visible_bias_init)
        # Hidden bias
        self.bh = torch.nn.Parameter(torch.zeros((1, nh)))
        
        # step in CD-k
        self.k = k

        # Training method
        self.training = training
        
    def get_device(self):
        """Returns current device for model's parameter storage."""
        return next(self.parameters()).device
    
    def move_to_cuda(self):
        """Move model parameters to cuda, if available."""
        if torch.cuda.is_available():
            self.to('cuda')
        else:
            print('Cuda not available: model is being trained on CPU. May be slow.')

    def _hidden_prob(self, v, temp=1.):
        """Probability vector of hidden units given visible, i.e. P(H = 1|V)."""
        return torch.sigmoid(temp*F.linear(v, self.W, self.bh)).detach()

    def _visible_prob(self, h, temp=1.):
        """Probability vector of visible units given hidden, i.e. P(V = 1|H)."""
        return torch.sigmoid(temp*F.linear(h, self.W.t(), self.bv)).detach()

    def _sample_hidden(self, v):
        """Sample hidden units given visible."""
        prob = self._hidden_prob(v)
        return torch.bernoulli(prob)

    def _sample_visible(self, h):
        """Sample visible units given hidden."""
        prob = self._visible_prob(h)
        if self.visible_normalized:
            return prob
        else:
            return torch.bernoulli(prob)

    def free_energy(self, v):
        """Compute free energy, i.e. F(x) = -\log \sum_h \exp (-E(x, h))."""
        v_term = torch.matmul(v, self.bv.t())
        w_x_h = F.linear(v, self.W, self.bh)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v_initial, v_given=None, n_steps=None):
        """Perform Gibbs sampling.

        Parameters:
        -----------
        v_initial : torch.tensor
            Initial value of visible units.
        v_given : (optional) list (length = self.nv)
            Values of visible units to condition on. Set value to None
            if you don't want to condition on that unit.
        n_steps : int
            Number of steps in Gibbs sampling.

        Returns:
        --------
        v_gibbs : torch.tensor
            Value of visible units after Gibbs sampling.
        """

        if n_steps == None:
            n_steps = self.k
            
        h = self._sample_hidden(v_initial)
        
        if v_given is not None:
            v_given_indices = torch.nonzero(~torch.isnan(v_given), as_tuple=True)[1]
            v_given_values = v_given[torch.nonzero(~torch.isnan(v_given), as_tuple=True)]
        
        for _ in range(n_steps):

            v_gibbs = self._sample_visible(h)

            # conditional sampling if v_given not None
            if v_given is not None:
                v_gibbs[:, v_given_indices] = v_given_values

            h = self._sample_hidden(v_gibbs)

        return v_gibbs

    def train(self, dataset):
        """Train model on dataset.

        Parameters:
        -----------
        dataset : torch Dataset object
            Training dataset.
        """
        
        # Move model to cuda if available.
        self.move_to_cuda()
        
        # Optimizer.
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Dataloader.
        # Attention: setting drop_last = False would raise error during PCD
        # traning (last batch in `data` may have different size than last Gibbs
        # state in `v_gibbs` when batch_size `bs` is not a divisor of
        # number of samples in dataset.        
        
        if dataset.__getitem__(torch.arange(len(dataset))).device.type == 'cuda':
        
            train_loader = DataLoader(dataset,
                                      batch_size=self.bs,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=False)
        else: 
            print('Dataset is on CPU. Memory pinning activated.')
            print('If the dataset is not too big, consider moving it to GPU calling dataset.to("cuda")')
            
            train_loader = DataLoader(dataset,
                                      batch_size=self.bs,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
            

        
        # Learning rate scheduler.
        if self.lr_trend == 'linear':
            lr_factor = lambda iep: 1. - iep/self.epochs
        elif self.lr_trend == 'constant':
            lr_factor = lambda iep: 1.
        else:
            print('Warning: unknown learning rate trend', self.lr_trend)
            print('Constant learning rate trend is assumed.')
            lr_factor = lambda iep: 1.
        scheduler = LambdaLR(optim, lr_lambda=lr_factor)     

        v_gibbs = None
        for iep in range(self.epochs):

            loss = 0.
            rec_loss = 0.

            for iter_num, data in enumerate(train_loader):
                optim.zero_grad()

                data = data.to(self.get_device()).type(torch.cuda.FloatTensor)

                # Implement PCD training
                if self.training == 'PCD' and v_gibbs is not None:
                    v_initial = v_gibbs
                else:
                    v_initial = data

                v_gibbs = self.forward(v_initial)
                loss_batch = self.free_energy(data) - self.free_energy(v_gibbs)
                
                loss_batch.backward()
                optim.step()
                
                loss = loss + float(loss_batch)/len(train_loader)
                rec_loss = rec_loss + float((v_initial - v_gibbs).norm(2, dim=1).mean()/len(train_loader))

            # Learning rate is updated according to scheduler
            scheduler.step()

            if iep % self.print_step == 0 and self.verbose:
                print("Epoch: {}, Loss: {}, Rec. error: {}".format(iep, loss, rec_loss))
            
            # Track (in-sample) learning.
            if self.track_learning and iep % self.track_step == 0:
                for track_method in self.track_method:
                    learning_estimate = self.get_insample_learning(dataset, track_method)
                    self.learning_curve[track_method].append(learning_estimate)
                    print(track_method, '=', learning_estimate)

    def save(self, filename=None, pickle_protocol=None):
        """Save model in pt file."""

        if filename is None:
            timestamp = datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)")
            self.autosave_filename = 'savepoint_' + timestamp + '.pt'
        
        if pickle_protocol is None:
            torch.save(self, filename)
        else:
            torch.save(self, filename, pickle_protocol = pickle_protocol)


    def check_overfit(self, train_data, val_data):
        """Returns the free energy differential between training dataset and validation
        dataset, which is used to monitor overfitting."""
        return self.free_energy(train_data) - self.free_energy(val_data)

    def sample(self, n_samples, v_given=None, therm=10**4):
        """Return sample from model.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples.
        v_given : (optional) torch.tensor
            Values of visible units to condition on. Set value to nan
            if you don't want to condition on that unit.
        therm : int
            Number of steps in Gibbs sampling (a.k.a. thermalization). 
            To obtain an iid sample the chain must have reached convergence, therefore
            it should be >= 10**3.
        """
        self.move_to_cuda()
        
        with torch.no_grad():
            if self.visible_normalized:
                random_init = torch.rand((n_samples,self.nv), dtype=torch.float).cuda()
            else:
                random_init = torch.randint(2,(n_samples,self.nv), dtype=torch.float).cuda()
            
            if v_given is not None:
                if not torch.is_tensor(v_given):
                    raise ValueError('v_given must be torch tensor.')
                v_gibbs = self.forward(random_init, v_given=v_given.to(self.get_device()), n_steps=therm)
            else:
                v_gibbs = self.forward(random_init, v_given=None, n_steps=therm)

        v_gibbs = v_gibbs.to('cpu')
        return v_gibbs
    
    def get_insample_learning(self, dataset, track_method):
        """Returns learning estimate for model."""
        if track_method == 'KDE':
            learning_estimate = self.get_kde_logL(dataset)           
        elif track_method == 'MMD':
            learning_estimate = self.get_mmd(dataset)
        elif track_method == 'NDB':
            learning_estimate = self.get_ndb(dataset)
        elif track_method == 'rec_error':
            learning_estimate = self.get_reconstruction_error(dataset)
            
            
        return learning_estimate
    
    def get_kde_logL(self, dataset, n_samples=10000, therm=10**4):
        """Kernel density estimation of log-likelihood of RBM on a dataset.
        
        A kernel density estimator is fitted to a sample from the RBM and the
        estimated density is used to compute the log-likelihood of the RBM on
        a dataset.
        
        Parameters:
        -----------
        dataset : pytorch Dataset object
            Dataset to compute log-likelihood on.
        bandwidth : float (default: None)
            KDE estimator bandwidth.
        n_samples : int (default: 10000)
            Number of samples from RBM to fit KDE estimator.
        therm : int (default: 10**4)
            Thermalization parameter for RBM sampling.
        
        
        Returns:
        --------
        logL : float
            Estimate of log-likelihood of RBM on dataset.
        """ 
        
        # Generate sample from RBM
        sample = self.sample(n_samples, therm=10**4).cpu().numpy()
        
        # Fit KDE estimator to sample
        kde = KernelDensity(kernel='gaussian', bandwidth=self.KDE_bandwidth).fit(sample)
        
        # Compute KDE estimator log-likelihood on dataset
        if len(dataset) > n_samples:
            data = dataset.__getitem__(np.random.choice(list(range(len(dataset))), size=n_samples, replace=False)).to('cpu')
        else:
            data = dataset.__getitem__(list(range(len(dataset)))).to('cpu') # get whole dataset
        
        if torch.is_tensor(data):
            logL = kde.score(data.numpy())
        elif isinstance(data, np.ndarray):
            logL = kde.score(data)
        else:
            raise ValueError('The __getitem__() method of dataset argument ' +
                             'must return either a torch tensor, or a numpy ndarray.')
        
        return logL
    
    def get_reconstruction_error(self, dataset, n_samples=10000, therm=10**4):
        """Reconstruction error (L^2 norm) of RBM on a dataset.
        
        Average minimum L^2 norm distance of points in a dataset from
        a sample of the RBM.
        
        Parameters:
        -----------
        dataset : pytorch Dataset object
            Dataset to compute log-likelihood on.
        therm : int (default: 10**4)
            Thermalization parameter for RBM sampling.
        
        Returns:
        --------
        error : float
            Average reconstruction error.
        """ 
             
        # Generate sample from RBM
        sample = self.sample(n_samples, therm=therm).cpu().numpy()
        
        if len(dataset) > n_samples:
            data = dataset.__getitem__(np.random.choice(list(range(len(dataset))), size=n_samples, replace=False)).to('cpu')
        else:
            data = dataset.__getitem__(list(range(len(dataset)))).to('cpu') # get whole dataset
        
                                       
        if torch.is_tensor(data):
            errors = list(map(lambda datapoint: np.min(np.sum((sample - datapoint)**2, axis=1)), data.numpy()))
        elif isinstance(data, np.ndarray):
            errors = list(map(lambda datapoint: np.min(np.sum((sample - datapoint)**2, axis=1)), data))
        else:
            raise ValueError('The __getitem__() method of dataset argument ' +
                             'must return either a torch tensor, or a numpy ndarray.')
        
        return np.mean(np.array(errors))
    
    def get_mmd(self, dataset, n_samples=10000, therm=10**4):
        """Maximum Mean Discrepancy (MMD) of RBM on a dataset.
        
        Parameters:
        -----------
        dataset : pytorch Dataset object
            Dataset to compute log-likelihood on.
        therm : int (default: 10**4)
            Thermalization parameter for RBM sampling.
        
        Returns:
        --------
        error : float
            MMD.
        """ 
        
        n_samples = np.minimum(n_samples, len(dataset))
        
        # Generate sample from RBM
        sample = self.sample(n_samples, therm=therm).cpu()
        
        if len(dataset) > n_samples:
            data = dataset.__getitem__(np.random.choice(list(range(len(dataset))), size=n_samples, replace=False)).to('cpu')
        else:
            data = dataset.__getitem__(list(range(len(dataset)))).to('cpu') # get whole dataset
        
        
        data = data.type(torch.FloatTensor)
        device='cpu'
        
        x = data
        y = sample
        
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
                
        return float(torch.mean(XX + YY - 2. * XY).numpy())
    
    def get_ndb(self, dataset, n_samples=10000, therm=10**4, confidence=0.95):
        """Number of statistically Different Bins (NDB) of RBM on a dataset.
        
        Points in dataset are binned into Voronoi cells using a k-means algorithm.
        NDB is the number of statistically different bins btw the dataset and a
        sample from the RBM, where the statistical difference is ascertained using
        a two-tailed z-test for each bin. 
        
        Parameters:
        -----------
        dataset : pytorch Dataset object
            Dataset to compute log-likelihood on.
        therm : int (default: 10**4)
            Thermalization parameter for RBM sampling.
        
        Returns:
        --------
        error : float
            NDB.
        """ 
        
        n_samples = np.minimum(n_samples, len(dataset))
        
        # Generate sample from RBM
        sample = self.sample(n_samples, therm=therm).cpu().numpy()
        
        if len(dataset) > n_samples:
            data = dataset.__getitem__(np.random.choice(list(range(len(dataset))), size=n_samples, replace=False)).to('cpu')
        else:
            data = dataset.__getitem__(list(range(len(dataset)))).to('cpu') # get whole dataset
            
        x = data
        y = sample.astype(float)        
        
        if len(dataset) > 1000:
            n_bins = int(np.ceil(n_samples/100))
        else:
            n_bins = int(np.ceil(n_samples/10))
        
        kmeans = KMeans(n_clusters=n_bins).fit(x)
        x_labels = kmeans.labels_
        y_labels = kmeans.predict(y)         

        n_x = len(x_labels)
        n_y = len(y_labels)

        ndb = 0   
        for this_bin in range(n_bins):
            n_x_points = np.sum(x_labels == this_bin)
            n_y_points = np.sum(y_labels == this_bin)

            num = (n_x_points/n_x) - (n_y_points/n_y)
            pooled_proportion = (n_x_points + n_y_points)/(n_x + n_y)
            den = np.sqrt(pooled_proportion*(1-pooled_proportion)*((1/n_x) + (1/n_y)))

            z = num/den

            significance = 1 - confidence
            if z <= norm.ppf(0.5*significance) or z >= norm.ppf(1- 0.5*significance):
                ndb = ndb + 1

        return ndb/n_bins


# Annealed Importance Sampling (AIS) routines for partition estimation are numerically unstable.
# Currently working on solution.
        
#     def get_logZ_estimate(self, n_temps = 10000, n_estimates = 10):
#         '''Computes an approximation to the partition function of an RBM model via
#         Annealed Importance Sampling (AIS).

#         Parameters
#         ----------
#         n_temps : int
#             Number of temperatures used in the AIS algorithm.
#         n_estimates : int
#             Number of estimates.

#         Returns
#         -------
#         AIS_Z : float
#             Average value of the estimates of the partition function of the model.
#         std_AIS_Z : float
#             Standard deviation of the estimates of the partition function of the
#             model.
#         '''

#         temps = torch.hstack((torch.linspace(0,0.5,int(0.05*n_temps)),
#                               torch.linspace(0.5,0.9,int(0.2*n_temps)),
#                               torch.linspace(0.9,1, int(0.85*n_temps))))

#         estimates = torch.ones(n_estimates).to(self.device)

#         v = torch.bernoulli(torch.sigmoid(torch.tile(self.bv, (n_estimates, 1)))).detach().to(self.device)

#         for i_temp in range(n_temps-1):

#             old_temp = temps[i_temp]
#             new_temp = temps[i_temp+1]

#             estimates += torch.sum(torch.log(1+torch.exp(new_temp*(v.matmul(self.W.t()) + self.bh)))
#                             - torch.log(1+torch.exp(old_temp*(v.matmul(self.W.t()) + self.bh))), axis = 1).detach()


#             h = torch.bernoulli(self._hidden_prob(v, temp=new_temp)).detach().to(self.device)

#             #h = torch.bernoulli(torch.nn.sigmoid(new_temp*torch.tile(v.matmul(self.W.t()) + self.bv, (n_estimates, 1)))).detach().to(self.device)

#             v = torch.bernoulli(self._visible_prob(h, temp=new_temp)).detach().to(self.device)

#         base_Z = torch.log((2**self.nh)) + torch.log(torch.sum(1+torch.exp(self.bv), dtype=torch.double).detach())

#         estimates = estimates + base_Z

#         AIS_Z = torch.mean(estimates)
#         std_AIS_Z = torch.std(estimates)/torch.sqrt(torch.tensor(n_estimates))

#         return AIS_Z, std_AIS_Z

#     def get_Z_exact(self):
#         '''Computes exact partition function of a binary-binary RBM.

#         Parameters
#         ----------
#         model : rbm.RBM object
#             Trained model.

#         Returns
#         -------
#         float
#             Value of the exact partition function of the model.

#         '''

#         k = np.minimum(self.nv, self.nh)
#         block_size = np.minimum(k, 16)

#         n_blocks = 2**(k - block_size)
#         Z = 0
#         for i_block in range(n_blocks):
#             block = np.zeros((2**block_size, k))
#             for (i_row, n) in enumerate(range(i_block*(2**block_size),
#                                               (i_block+1)*(2**block_size))):
#                 block[i_row, :] = np.array(list(np.binary_repr(n, width = k))).astype('float')
#             if k == model.d:
#                 Z += np.sum(np.exp(block.dot(self.bv))
#                             *np.prod(1 + np.exp(self.bh + block.dot(self.W.t())), axis=1))
#             elif k == self.nh:
#                 Z += np.sum(np.exp(block.dot(self.bh))
#                             *np.prod(1 + np.exp(self.bv + block.dot(self.W)), axis=1))
#         return Z
