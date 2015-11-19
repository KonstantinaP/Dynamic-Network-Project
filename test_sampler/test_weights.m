close all
clear all
addpath '../inference/'
addpath '../'
%seed=55; %alpha=3;
seed=55;
rand('seed',seed);
randn('seed',seed);


%% Generate a path from the process.
alpha = 3; sigma = 0; tau = 1; % Parameters gamma process
phi = 50;                       % tunes dependence in dependent gamma process
rho = 0.1;                      % death rate for latent interactions
T = 4;                         % Number of time steps
[Z, Nall, n_new, n_old, counts, Kevol, Knew, true_weights] = dyngraphrnd(alpha, sigma, tau, T, phi, rho);
for t=1:T
    mtx = squeeze(n_new(t,:, :));
    M(:, t) = sum(mtx, 1)' + sum(mtx, 2);
end

% Here we need to preprocess the parameters 'counts' that will be used as an
% input in the sampler. 
% Note that the 'true_weights' contains _part_ of the set of K weights (as 
% denoted in the posterior sampling section of the paper) that are
% created (observed through links or through no links) in the data Z. This 
% partial set caontains only the weights that were created using the dynamic 
% Plackett - Luce evolution model. We need to augment the 'counts' to contain the 
% counts for the Knew new nodes created at time t using the urn process. These 
% nodes might propagate their links through the Nold counts.

% n_old, n_new, Nall already contain these.


 counts = [counts zeros(T, sum(Knew))];
counts = counts';


%% Test the posterior sampling of the weights
N_Gibbs= 400;
settings.typegraph = 'simple';
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;

weights = ones(size(counts)); % initialize weights
%Increase the number of nodes by one to account for the unobserved ones.
 % sampler needs ot take into account the w_{t\ast}
weights = [weights ; zeros(1, size(weights,2))];
counts = [counts; zeros(1, T)];
[K N] = size(weights);
issimple =1;
rate=zeros(N, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(N,1); % Leapfrog stepsize


weights_samples = zeros(K, T, N_Gibbs);
for i = 1:N_Gibbs
    
    [weights, rate(:, i)] = sample_weights(weights, counts, M, epsilon, alpha, tau, phi, settings,issimple);   
    weights_samples(:, :, i) = weights;
    if i<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:i), 2) - 0.6));
    end
    
    
end








