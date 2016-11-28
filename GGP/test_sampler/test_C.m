clear all
close all


addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'

addpath '../../inference/utils/'



% cases considered and checked
% phi = 0 : counts all 0, as expected



%
% seed=10;
% randn('seed', seed);
% rand('seed', seed);




alpha = 10; sigma = 0.1; tau = 1; % Parameters gamma process
phi = 10;                         % tunes dependence in dependent gamma process
rho = 0.1;                       % death rate for latent interactions
T = 3;                           % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;

settings.gcontrol =0;
gvar= 1:T;
settings.g_a = .1;
settings.g_b = .1;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;


N_Gibbs=3000;
N_burn = 0.3*N_Gibbs;
thin=1;
N_samples = (N_Gibbs-N_burn)/thin;



[Z, w, c, KT,  N_new, N_old, N, M, indchain]= ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings);


Na = size(w,2);
%tweights=w(:, 1:K(T));test_C

indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

wrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; wrem' ];
crem =  sum(c(:, ~indlog),2);

tc=[c(:, indlog)';  crem'];


m = M(indchain, :);

% Init C
weights= tw;
[K T] =size(weights);
K=K-1;

% random C initialization
counts = poissrnd(phi.*weights)+100;
initc=counts;
%C(isnan(weights))=0;
c_samples =  zeros(K, T, N_samples);


rate=zeros(T, N_Gibbs);

counts_st = cell(1, T);

for t=1:N_Gibbs
    if mod(t, 200)==0
        t
    end
    counts = ggp_sample_C(counts, weights, phi, alpha, sigma, tau);
    
    if (t>N_burn && rem((t-N_burn),thin)==0)
        indd = ((t-N_burn)/thin);
        for t=1:T
            
            counts_st{t}(indd, :) = counts(1:end-1, t);
            
        end
            c_samples(:,:, indd) = counts(1:end-1, :);

    end
end


for t=1:T
    [~, indt{t}] = sort(m(:,t), 'descend');
end
thin=1;
Na =size(m,1);




for t=1:T
    figure
    for k=1:min(size(tw(1:end-1, :), 1), 50)
        plot([k, k],...
            quantile(counts_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(k, tc(indt{t}(k), t), 'xg', 'linewidth', 2)
    end
    xlim([0.1,min(Na, 50)+.5])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
    ylabel('counts parameter', 'fontsize', 16)
end



