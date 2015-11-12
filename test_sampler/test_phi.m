clear all
close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model





%% Sample the latent C_1,C_2,...
tau=1;
alpha=10;
T=20;
true_phi = 30;

indmax = 1000; % so that to allocate memory
c = zeros(T, indmax);
w = zeros(T, indmax);


w1_all = gamrnd(alpha, 1/tau);
c1_all = poissrnd(true_phi*w1_all);
if c1_all>0
    [~, m, K(1)] = crprnd(alpha, c1_all);
    for i=1:K(1)
        c(1, i) = m(i);
        theta(i) = alpha*rand;
    end
else
    K(1) = 0;
end


wrest = zeros(1, T);
crest = zeros(1, T);

for t=2:T
    % Sample counts for existing atoms
    ind = c(t-1,:)>0;
    w(t, ind) = gamrnd(c(t-1,ind), 1/(tau+true_phi));
    c(t,ind) = poissrnd(true_phi*w(t, ind));
    % Sample counts for new atoms
    w_rest = gamrnd(alpha, 1/(tau+true_phi));
    c_rest = poissrnd(true_phi * w_rest);
    if c_rest>0
        [~, m, K_new] = crprnd(alpha, c_rest);
        for i=1:K_new
            c(t, K(t-1)+i) = m(i);
            theta(K(t-1)+i) = alpha*rand;
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

% % sample from GGP in the local steps
% local_w = zeros(T, indmax);
% kmax=0;
% sigma=0;
% for t=1:T
%
%
%     [temp, threshold] = GGPrnd(alpha, sigma, tau);
%     local_w(t, 1:length(temp))=temp;
%     kmax= max(kmax, length(temp));
%
%
% end
%
% local_w = local_w(:, 1:kmax);



phi=2; % start sampler.
phi_a =.1;
phi_b= .1;

iters=1000;
phi_samples=zeros(1, iters);
for t=1:iters
    %     phi = sample_phi(phi, phi_a, phi_b, weights, alpha, tau)   ;
    phi = slice_sample_phi(phi, phi_a, phi_b, weights, alpha, tau);
    phi_samples(t) = phi;
end

figure
plot(phi_samples)
