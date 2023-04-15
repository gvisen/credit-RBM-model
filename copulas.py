from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import layers

class CopulaModel:

    def __init__(self):
        self.n_obligors = None
        self.n_factors = None
        pass

    def simulate_pds(self, n_samples):
        '''Simulate portfolio default probabilities.
        
        Parameters:
        -----------
        n_samples : int
            Number of simulations.
            
        Returns:
        --------
        pds : ndarray of size (n_samples, n_obligors).
            Dataset of simulated default probabilities.
        '''
        
        raise NotImplementedError
        
    def fit(self, dataset):
        
        raise NotImplementedError
   
class GaussianCopula(CopulaModel):

    def __init__(self, avg_pds=None, n_factors=None, coefs=None):
        
        # Define copula model coefficients
        self.avg_pds = avg_pds
        self.coefs = coefs

        # Define parent class parameters
        if self.coefs is not None:
            self.n_obligors, self.n_factors = np.shape(self.coefs)
        else:
            self.n_obligors = None
            self.n_factors = None

    def simulate_pds(self, n_samples):

        if self.avg_pds is None or self.coefs is None:
            raise ValueError('You must specify the copula parameters before simulating.')
        
        n_debtors, n_factors = self.coefs.shape
    
        z = tf.random.normal([n_samples, self.n_factors], mean=0., stddev=1.)
        y = self.coefs / tf.math.sqrt(1 - tf.reduce_sum(self.coefs**2, axis=1))[:, tf.newaxis]
        u = z @ tf.transpose(y)
        t = tfp.distributions.Normal(loc=0., scale=1.).quantile(self.avg_pds)/tf.math.sqrt(1 - tf.reduce_sum(self.coefs**2, axis=1))
        w = tf.transpose(t[:, tf.newaxis]) - u
        
        pds = tfp.distributions.Normal(loc=0., scale=1.).cdf(w)
    
        return pds.numpy()
    
    def fit(self, pds, n_factors=None, n_epochs=2000, verbose=False):
        '''Fit the multi-factor Gaussian copula model on a dataset of default probabilities.
        
        TODO: add brief description of algorithm.
        
        Parameters:
        -----------
        avg_pds : 1d ndarray of size (self.n_obligors,)
            Unconditional expected default probabiities of obligors.
        assets : 2d ndarray of size (N, self.n_obligors)
            Sample of (latent) *standardized* asset process.
        corr_structure : string
            If `equicorrelation` an equicorrelation structure is fitted,
            if `general` a general correlation one-factor structure is fitted.
        '''
        
        self.n_factors = n_factors

        transformed_data = norm.ppf(pds)
    
        # norm.ppf returns -np.inf for values below -6.
        # TODO: come up with a better solution here
        transformed_data[transformed_data == -np.inf] = -12.
        
        n_debtors = pds.shape[1]
        
        def loss(A, matrix):
        
            return tf.norm(A @ tf.transpose(A) - matrix)
        
        # initial value
        guess_coefs = tf.random.uniform([n_debtors, n_factors], minval=0., maxval=1/(3*np.sqrt(n_factors)))
        A = tf.Variable(guess_coefs/tf.math.sqrt(1 - tf.reduce_sum(guess_coefs**2, axis=1))[:, tf.newaxis])

        # learning rate
        lr_start = 0.1

        empirical_cov = tf.cast(tfp.stats.covariance(transformed_data), dtype=tf.float32)

        losses = np.zeros(n_epochs)
        best_A = A
        best_loss = 1.

        for epoch in range(n_epochs):

            lr = lr_start*(1 - (epoch/n_epochs))

            with tf.GradientTape() as t:
                current_loss = loss(A, empirical_cov)

            grad = t.gradient(current_loss, A)
            A.assign_sub(lr*(grad/tf.linalg.norm(grad)))

            if current_loss < best_loss:
                best_A = A
                best_loss = current_loss

            losses[epoch] = current_loss
            
            if verbose:
                print('Epoch:', epoch, 'Loss:', current_loss.numpy())
        
        if verbose:
            plt.plot(losses)
            plt.title('Learning curve')
            plt.show()

        normalizations = np.sqrt(1/(1+np.sum(best_A**2, axis=1)))
        
        # Set copula parameters to best estimates
        self.avg_pds = np.mean(pds, axis=0)
        self.coefs = best_A.numpy()*normalizations[:, np.newaxis]