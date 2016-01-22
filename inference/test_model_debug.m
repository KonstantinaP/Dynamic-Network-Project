% Test dynamic model
addpath '../inference/'
addpath '../inference/utils/'
close all
clear all
K = 10 ;
T = 2;

rand('seed',21);
randn('seed',21);
%% Generate toy data
for t=1:T
    ID{t} = [1:K];
end


%% OR generate from the process
true_alpha=10;
true_sigma =0;
true_tau =1;
T=10;
true_phi= 50;
rho =1;
dt=1;

[G, trueN, trueN_new, trueN_old, trueC, trueK, trueK_new, trueW] = dyngraphrnd(true_alpha, true_sigma, true_tau, T, true_phi, rho, dt);
K = sum(trueK_new) + trueK(end); % this is the total number of nodes. A node's entry in trueN_new might be zero or non-zero.
ID = cell(1, T);

% for debugging purposes consider that the evidence is the Nnew. Ignore
% Nold and G.
for t=1:T
    ss=trueK(end)+sum(trueK_new(1:t-1));
    ID{t} = [1:trueK(t)  ss+1:ss+trueK_new(t)  ];
end


% keyboard
% Run time-varying Plackett-Luce, initialise
alpha = 1;
N_Gibbs = 1000;
N_burn = 100;
thin = 1;
phi = 1;%*ones(3, 19);
tau=1;
settings.typegraph = 'simple';
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
settings.times = 1:T;
settings.phi_a = .1;
settings.phi_b = .1;
settings.alpha_a = .1;
settings.alpha_b = .1;
settings.rho = 100; % decay rate. Controls the dependency on the interactions at the previous time step

settings.sample_correlation = 1;
[weights_st, alpha_st, phi_st, stats, weights, C, nnew] = run_inference_debug(trueN_new, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings);


