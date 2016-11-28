close all
clear all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'



%seed=55; %alpha=3;
seed=rand;
rand('seed',seed);
randn('seed',seed);


%% Generate a path from the process.
alpha = 10; sigma = 0; tau = 1; % Parameters gamma process
phi = 10;                       % tunes dependence in dependent gamma process
settings.rho = 10;                      % death rate for latent interactions
T = 10;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;
tgvar = ones(1, T);
settings.rho_a=1;
settings.rho_b=0.01;
settings.gcontrol=1;
[Z, w, c, K, N_new, N_old, N, M, indchain]= dyngraphrnd(alpha, sigma, tau, T, phi, tgvar, settings);

%[S, tNnew, tNold, tN, tM, tW, tC, tWrem, ind] = keep_active_nodes(w, c, indchain, Z, N_new, N_old, N);
%% Test the posterior sampling of the weights
N_Gibbs= 1000;
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
N_burn = 200;
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

tn_new = cell(1,T);
tn_old = cell(1,T);
for t=1:T
    % make it symmetric
    temp = squeeze(N_new(t, indchain, indchain)) + squeeze(N_new(t, indchain, indchain))';%- diag(n_new(t, indchain, indchain));
    
    % make it upper triangualr for the sampler to work correctly
    tn_new{t} =sparse( triu(temp, 1));
    
    tempo = squeeze(N_old(t, indchain, indchain)) + squeeze(N_old(t, indchain, indchain))';%- diag(n_new(t, indchain, indchain));
    tn_old{t} =sparse( triu(tempo, 1));
    
end



rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(Na-1)^(1/4).*ones(T,1); % Leapfrog stepsize


%weights_samples = zeros(Na+1, T, N_Gibbs);
N_samples = (N_Gibbs-N_burn)/thin;
rho_st = zeros(1,N_samples);
rho = gamrnd(settings.rho_a, 1/settings.rho_b);
tic
for i = 1:N_Gibbs
    if rem(i,100)==0
        i
    end
    
    [rho] = slice_sample_rho(rho, tn_old, tn_new, settings.rho_a, settings.rho_b, settings.dt);
    
    if (i>N_burn && rem((i-N_burn),thin)==0)
        indd = ((i-N_burn)/thin);
        rho_st(indd) = rho;
        
    end
end
toc

figure
plot(thin:thin:N_samples, rho_st);


hold on
plot(thin:thin:N_samples, settings.rho*ones(N_samples, 1), '--g', 'linewidth', 3);
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\rho', 'fontsize', 16);
box off
%xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
