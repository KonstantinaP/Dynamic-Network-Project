clear all
close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model 





%% Sample the latent C_1,C_2,...
tau=1; 
true_alpha=1; 
T=20;
phi=10;

indmax = 1000; % so that to allocate memory
c = zeros(T, indmax);
w = zeros(T, indmax);


w1_all = gamrnd(true_alpha, 1/tau);
c1_all = poissrnd(phi*w1_all);
if c1_all>0
    [~, m, K(1)] = crprnd(true_alpha, c1_all);
    for i=1:K(1)
        c(1, i) = m(i);
        theta(i) = true_alpha*rand;
    end
else
    K(1) = 0;
end


wrest = zeros(1, T);
crest = zeros(1, T);

for t=2:T
    % Sample counts for existing atoms
    ind = c(t-1,:)>0;
    w(t, ind) = gamrnd(c(t-1,ind), 1/(tau+phi));
    c(t,ind) = poissrnd(phi*w(t, ind));
    % Sample counts for new atoms
    w_rest = gamrnd(true_alpha, 1/(tau+phi));
    c_rest = poissrnd(phi * w_rest);
    if c_rest>0
        [~, m, K_new] = crprnd(true_alpha, c_rest);    
        for i=1:K_new
            c(t, K(t-1)+i) = m(i);
            theta(K(t-1)+i) = true_alpha*rand;
        end
        K(t) = K(t-1) + K_new;   
    else
        K(t) = K(t-1);
    end
     wrest(t) = w_rest;
    crest(t) = c_rest;
end
C = c(2:T-1, 1:K(T-1))';
C(end+1,:) = crest(2:end-1);

weights = w(2:T, 1:K(T-1))';
weights(end+1, :) = wrest(2:end);


alpha=1; % start sampler.

% TRY with gamma distribution prior
alpha_a =.1;
alpha_b= .1;

iters=10000;
alpha_samples=zeros(1, iters);
for t=1:iters
%alpha = sample_a(weights, C, alpha, phi, tau, alpha_a, alpha_b);
 alpha = slice_sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b)

    alpha_samples(t) = alpha;
end


mean(alpha_samples)
    
figure
plot(alpha_samples)
