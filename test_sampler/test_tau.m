clear all
 close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model

%
% seed=rand;
% randn('seed', seed);
% rand('seed', seed);
% 



alpha = 10; sigma = 0; true_tau = 2; % Parameters gamma process
phi = 10;                       % tunes dependence in dependent gamma process
rho = 0.1;                      % death rate for latent interactions

T = 11;                         % Number of time steps
gvar = ones(1,T);
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;
settings.gcontrol=0;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
N_burn = 0;
thin = 1;

[Z, w, c, K, N_new, N_old, N, M, indlinks]= dyngraphrnd(alpha, sigma, true_tau, T, phi, rho, gvar,  settings);


indlog = false(1, length(indlinks)) ;
indlog(indlinks) = true;

wrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; wrem' ];


crem =  sum(c(:, ~indlog),2);
tc=[c(:, indlog)';  crem'];
% start sampler.
tau_a = .1;
tau_b = .1;


iters=1000;
tau_sam=zeros(1, iters);
for itt=1:iters
    tau_sam(itt)=gamrnd(tau_a, 1/tau_b)    ;
end
figure;
hist(tau_sam, 30)

tau_samples=zeros(1, iters);
nchains=1;
tau_all = [];

for ch=1:nchains
    tau=gamrnd(tau_a, 1/tau_b);
    
    for t=1:iters
        if rem(t,2)==0
        [tau ] = sample_tau(tau, tau_a, tau_b, tw, alpha, phi, tc )   ;
% [tau] = slice_sample_tau(tw, tc, alpha, phi, tau, tau_a, tau_b)   ;     
            disp('random w')

        else
            disp('slice')
            [tau] = slice_sample_tau(tw, tc, alpha, phi, tau, tau_a, tau_b)   ;     
        end
                chain(ch).tau_samples(t) = tau;


    end
    tau_all =  [tau_all chain(ch).tau_samples];
end

figure
thin=1;
col = {'k', 'r', 'b'};
for ch=1:nchains
    hold on
    plot(thin:thin:iters,  chain(ch).tau_samples,  col{ch});
    %     plot(thin:thin:iters,  tet,  'y');
end

%end

plot(thin:thin:iters, true_tau*ones(iters, 1), '--g', 'linewidth', 3);
legend({'Chain 1','Chain 2',  'Chain 3', 'True'}, 'fontsize', 16, 'location', 'Best')
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\tau', 'fontsize', 16);
box off
xlim([0, iters])

figure
hist(tau_all, 30)
hold on
plot(true_tau, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\tau', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);




