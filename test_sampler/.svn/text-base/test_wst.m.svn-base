close all
clear all
addpath '../inference/'
addpath '../inference/utils/'
addpath '../'
%seed=55; %alpha=3;
% seed=5;
% rand('seed',seed);
% randn('seed',seed);


%% Generate a path from the process.
alpha = 4; sigma = 0; tau = 1; % Parameters gamma process
phi = 50;                       % tunes dependence in dependent gamma process
settings.rho = 0.01;                      % death rate for latent interactions
T = 10;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;
settings.gcontrol =1;
settings.g_a=5; 
settings.g_b=1;
tgvar =1:T;
[Z, w, c, K, N_new, N_old, N, M, indchain]= dyngraphrnd(alpha, sigma, tau, T, phi, tgvar, settings);


%% Test the posterior sampling of the weights
N_Gibbs= 2000;
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
N_burn = 0;
thin = 1;


% Here we care only about the chain construction of w's ansd c's. Not the local nodes.
Na = size(w,2);
%tweights=w(:, 1:K(T));
tweights=w;
indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

wrem =  sum(tweights(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights =tweights(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tweights = [tweights'; wrem' ];
crem =  sum(c(:, ~indlog),2);

counts=[c(:, indlog)';  crem'];


m = M(indchain, :);




N_samples = (N_Gibbs-N_burn)/thin;


nchains = 1;
 gvar = gamrnd(settings.g_a, 1/settings.g_b,1,T);
% gvar= tgvar;

for t=1:T
    % make it symmetric
    temp = squeeze(N_new(t, indchain, indchain)) + squeeze(N_new(t, indchain, indchain))';%
    % make it upper triangualr for the sampler to work correctly
    tn_new{t} =sparse( triu(temp, 1));
end


for ch=1:nchains
    
    wts = rand(size(tweights)); % random initialization of the weights
    wts(1:end-1, :) = tweights(1:end-1, :);
    
    for i = 1:N_Gibbs
        
      if mod(i, 100)==0
        i
      end
%
            [wts] = sample_Wst(wts, counts, alpha, phi, tau, gvar);
%    
     [gvar] = sample_gvar_mh(wts, tn_new, alpha, tau, phi, gvar, settings.g_a, settings.g_b);
     
        if (i>N_burn && rem((i-N_burn),thin)==0)
            indd = ((i-N_burn)/thin);
            
            chain(ch).wst_samples(indd, :) = wts(end, :);
             gvar_st(indd, :) = gvar;
        end
    end
end


figure
for t=1:T
      hold on
    
        plot([t, t],...
            quantile(gvar_st(:, t),[.025,.975]), 'r', ...
            'linewidth', 3);
     
        plot(t, tgvar(t), 'xg', 'linewidth', 2)
%     xlim(-1:T])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    
    ylabel('gamma parameter', 'fontsize', 16)
end

for ch=1:nchains
    figure
    
    for t=1:T
        plot([t, t], quantile(chain(ch).wst_samples(:, t),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(t, tweights(end, t), 'xg', 'linewidth', 2)
      
    end
    
      
        xlim([0.1,min(Na, 50)+.5])
        legend('95% credible intervals', 'True value')
        legend boxoff
        box off
        xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
      
      ystr = sprintf('Sociability parameter w rem for all t');
        ylabel(ystr, 'fontsize', 16)
end


col = {'-r', '-b'};

for ch=1:nchains
    for i=1:T
        figure
        % for k=1:nchains
        plot(thin:thin:N_Gibbs,  chain(ch).wst_samples(:, i), col{ch});
        hold on
        %end
        plot(thin:thin:N_Gibbs, wrem(i)*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
        legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
        legend boxoff
        xlabel('MCMC iterations', 'fontsize', 16);
        ystr = sprintf('wrem for location t=%d', i);
        ylabel(ystr, 'fontsize', 16);
        box off
        xlim([0, N_Gibbs])
        
        figure
        hist(chain(ch).wst_samples(:,i), 30)
        hold on
        plot(wrem(i), 0, 'dg', 'markerfacecolor', 'g')
        %xlabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
        xstr = sprintf('location t=%d', i);
        xlabel(xstr, 'fontsize', 16);
        ylabel('Number of MCMC samples', 'fontsize', 16);
    end
    
end




