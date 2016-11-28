clear all
 close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model

%
% seed=1;
% randn('seed', seed);
% rand('seed', seed);
% 



alpha =10; sigma = 0; tau = 1; % Parameters gamma process
true_phi = 40;                       % tunes dependence in dependent gamma process
true_rho = 0.1;                      % death rate for latent interactions
T = 10;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
N_burn = 0;
thin = 1;
settings.gcontrol=0;
true_gvar = ones(1,T);
[Z, w, c, K, N_new, N_old, N, M, indlinks]= dyngraphrnd(alpha, sigma, tau, T, true_phi, true_rho, true_gvar, settings);


indlog = false(1, length(indlinks)) ;
indlog(indlinks) = true;

wrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; wrem' ];



% start sampler.
phi_a = 0.1;
phi_b = 0.1;
 
iters=1000;
phi_sam=zeros(1, iters);
for itt=1:iters
    phi_sam(itt)=gamrnd(phi_a, 1/phi_b)    ;
end
figure;
hist(phi_sam, 30)

phi_samples=zeros(1, iters);
nchains=1;
phi_all = [];

for ch=1:nchains
    phi=gamrnd(phi_a, 1/phi_b);
    for t=1:iters
        [phi  ] = sample_phi(phi, phi_a, phi_b, tw, alpha, tau)   ;
%                phi = slice_sample_phi(phi, phi_a, phi_b, tw, alpha, tau);
        chain(ch).phi_samples(t) = phi;
    end
    phi_all =  [phi_all chain(ch).phi_samples];
end

figure
thin=1;
col = {'k', 'r', 'b'};
for ch=1:nchains
    hold on
    plot(thin:thin:iters,  chain(ch).phi_samples,  col{ch});
    %     plot(thin:thin:iters,  tet,  'y');
end

%end

plot(thin:thin:iters, true_phi*ones(iters, 1), '--g', 'linewidth', 3);
legend({'Chain 1','Chain 2',  'Chain 3', 'True'}, 'fontsize', 16, 'location', 'Best')
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\phi', 'fontsize', 16);
box off
xlim([0, iters])

figure
hist(phi_all, 30)
hold on
plot(true_phi, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\phi', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);


