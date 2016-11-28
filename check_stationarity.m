% Check that locally we have gamma(alpha, tau) marginals

clear all
close all

addpath '../'
addpath 'inference/'
addpath 'inference/utils/'

% seed=4;
% rand('seed',seed);
% randn('seed',seed);

alpha=5;
sigma =0;
tau =1;
T=14;
phi= 5;
rho =1;
dt=1;

settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-4;
[Z, w, c, K, N_new, N_old, N, M]= dyngraphrnd(alpha, sigma, tau, T, phi, rho, dt, settings);
keyboard

ids= ~isnan(w);
w(isnan(w))=0;
G = sum(w,2);

% Sample from a gamma distribution G(alpha,tau)
gd=zeros(1,T);
for s=1:T
    gd(s) =  gamrnd(alpha, 1/tau);
end


ll = max([gd(:)' G(:)']);
figure
histfit(G,T, 'gamma')
figure
histfit(gd,T, 'gamma')
% xlim([0, N_Gibbs])



