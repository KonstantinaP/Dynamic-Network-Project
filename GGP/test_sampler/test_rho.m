clear all
close all

addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'
addpath '../../inference/utils/'

% cases considered and checked
% phi = 0 : counts all 0, as expected
%
%
% seed=10;
% randn('seed', seed);
% rand('seed', seed);




talpha = 40; tsigma = 0.1; ttau = .1; % Parameters gamma process
tphi =1;                       % tunes dependence in dependent gamma process
T =2;                         % Number of time steps
trho = 4;                      % death rate for latent interactions

settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-4;

settings.gcontrol =0;
gvar= 1:T;
settings.g_a = .1;
settings.g_b = .1;

settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;

N_Gibbs=5000;
N_burn = 0.2*N_Gibbs;
thin = 1;
N_samples = (N_Gibbs-N_burn)/thin;


settings.estimate_alpha=0;
settings.estimate_sigma=0;
settings.estimate_tau= 1;
settings.estimate_phi=1;
settings.estimate_rho=1;
settings.rw_alpha = 0;
settings.rw_std(1) = 0.02; %sigma
settings.rw_std(2) = 1; %tau
settings.rw_std(3) = 1; %phi
settings.rw_std(4) = 1; %rho
settings.hyper_alpha(1) = 0;
settings.hyper_alpha(2) = 0;
settings.hyper_tau(1)=0;
settings.hyper_tau(2)=0;
settings.hyper_phi(1)=0;
settings.hyper_phi(2)=0;
settings.hyper_sigma(1)=0;
settings.hyper_sigma(2)=0;
settings.hyper_rho(1)=10;
settings.hyper_rho(2)=1;

[Z, w, c, KT,  N_new, N_old, N, M, indchain, indcnz]= ggp_dyngraphrnd(talpha, tsigma, ttau, T, tphi, trho, gvar, settings);

%keyboard

indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

twrem =  sum(w(:, ~indlog), 2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights =w(:, indlog); % keep only the nodes that are active at least one time over T.
%Increase the number of nodes by one to account for the unobserved ones.
tweights = [tweights'; twrem' ];
K = size(tweights,1)-1;

crem =  sum(c(:, ~indlog),2);
tcounts=[c(:, indlog)';  crem'];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%UP TO HERE
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

keyboard



rho_st = zeros(1,N_samples);
rho = gamrnd(settings.hyper_rho(1), 1/settings.hyper_rho(2));
tic
for i = 1:N_Gibbs
    if rem(i,100)==0
        i
    end
    
    [rho] = ggp_sample_rho(rho, tn_old, tn_new, settings);
    
    if (i>N_burn && rem((i-N_burn),thin)==0)
        indd = ((i-N_burn)/thin);
        rho_st(indd) = rho;
        
    end
end
toc

figure
plot(thin:thin:N_samples, rho_st);


hold on
plot(thin:thin:N_samples, trho*ones(N_samples, 1), '--g', 'linewidth', 3);
plot(thin:thin:N_samples, cumsum(rho_st)./(thin:thin:N_samples), 'g-');
    
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\rho', 'fontsize', 16);
box off
%xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
