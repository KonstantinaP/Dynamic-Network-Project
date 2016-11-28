close all
clear all

addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'

addpath '../../inference/utils/'



%seed=55; %alpha=3;
% seed=3;
% rand('seed',seed);
% randn('seed',seed);
% 

%% Generate a path from the process.
alpha =5; sigma = 0.1; tau = 4; % Parameters gamma process
phi = 0;                       % tunes dependence in dependent gamma process
rho = 0.001;                      % death rate for latent interactions
T = 1;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6; 
settings.gcontrol =0;
gvar= 1:T;
settings.g_a = .1;
settings.g_b = .1;

[Z, w, c, K, N_new, N_old, N, M, indchain]= ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings);

%[S, tNnew, tNold, tN, tM, tW, tC, tWrem, ind] = keep_active_nodes(w, c, indchain, Z, N_new, N_old, N);
%% Test the posterior sampling of the weights
N_Gibbs= 10000;
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
N_burn = 0;
thin = 1;


% Here we care only about the chain construction of w's ansd c's. Not the local nodes.
Na = size(w,2);
tweights=w(:, 1:K(T));
indlog = false(1, K(T)) ;
indlog(indchain) = true;
wrem =  sum(tweights(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights =tweights(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tweights = [tweights'; wrem' ];
counts=[c(:, indlog)';  zeros(1, T)];


m = M(indchain, :);


rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(Na-1)^(1/4).*ones(T,1); % Leapfrog stepsize


%weights_samples = zeros(Na+1, T, N_Gibbs);
N_samples = (N_Gibbs-N_burn)/thin;
weights_st = cell(1, T);

wts = rand(size(tweights)); % random initialization of the weights
wts(end, :) = wrem; % keep the wrem the same as in the truth
for i = 1:N_Gibbs
    
    if mod(i, 200)==0
        i
    end
    
    
    [wts, rate(:, i)] = ggp_sample_weights(wts, counts, m, epsilon, alpha, sigma, tau, phi, gvar, settings);
    %     wts
    if i<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:i), 2) - 0.6));
    end
    
    
    if (i>N_burn && rem((i-N_burn),thin)==0)
        indd = ((i-N_burn)/thin);
        for k=1:T
            weights_st{k}(indd, :) = wts(:, k);
        end
    end
end

for t=1:T
    [~, indt{t}] = sort(m(:,t), 'descend');    
end

for t=1:T
    figure
    for k=1:min(size(tweights(1:end-1,:),1), 50)
        plot([k, k],...
            quantile(weights_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(k, tweights(indt{t}(k), t), 'xg', 'linewidth', 2)
    end
    xlim([0.1,min(Na, 50)+.5])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
    ylabel('Sociability parameter', 'fontsize', 16)
end




