% Test ynamic model
addpath '../inference/'
addpath '../inference/utils/'
close all
clear all
K = 10 ;
T = 5;
%% Generate toy data
for t=1:T
    ID{t} = [1:K];
end
% ID{3} = [1:K];
% ID{4} = [1:K];
% Z = cell(1, T);
% % for t=1:T
%     r = rand(K, K);
%     r =  triu(r);
%     Z{t} = r>0.4;
%
% end

z=zeros(K, K);
for k=1:2:K
    z(k,:) = ones(1, K);
    
    %z(:, k) = ones(1, K);
    
end

% z=zeros(K, K);
% z(1, :)= ones(1, K);
% z(3, :)= ones(1, K);


z=triu(z);
for t=1:T
    
    Z{t} = z;
    
end


%% OR generate from the process
% true_alpha=5;
% true_sigma =0;
% true_tau =1;
% T=5;
% true_phi= 50;
% rho =1;
% dt=1;
%
% [G, true_N, true_N_new, true_N_old, true_c, true_K, true_K_new, true_w] = dyngraphrnd(true_alpha, true_sigma, true_tau, T, true_phi, rho, dt);
% K = sum(true_K_new) + true_K(end);
% ID = cell(1, T);
% for t=1:T
%
%     Z{t} = squeeze(G(t, :, :));
%     ID{t} = 1:K;
% end
%


% keyboard
% Run time-varying Plackett-Luce, initialise
alpha = 20;
N_Gibbs = 1000;
N_burn = 0;
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
[weights_st, alpha_st, phi_st, stats, weights, C, nnew, nold] = run_inference(Z, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings);


