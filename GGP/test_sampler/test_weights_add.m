%close all
clear all



addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'

addpath '../../inference/utils/'



%seed=55; %alpha=3;
% seed=2;
% rand('seed',seed);
% randn('seed',seed);


%% Generate a path from the process.
alpha = 10; sigma = 0.2; 
tau = 1; % Parameters gamma process
phi = 5;                       % tunes dependence in dependent gamma process
rho=0.1; % death rate for latent interactions
settings.gcontrol=0;

T = 3;
settings.dt=1;

settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-6;
settings.alpha_a = .1    ;
settings.alpha_b = .1 ;
settings.phi_a = .1;
settings.phi_b = .1;
settings.g_a=5;
settings.g_b=1;
settings.rho_a=.1;
settings.rho_b=.1;

%tgvar =gamrnd(settings.g_a, 1/settings.g_b,1,T);
gvar=ones(1,T);
% Number of time steps
[Z, w, c, KT,  N_new, N_old, Nall, M, indchain] = ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings);



indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

twrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; twrem' ];

crem =  sum(c(:, ~indlog),2);
tc=[c(:, indlog)';  crem'];

tm = M(indchain, :);







%% Test the posterior sampling of the weights
N_Gibbs=5000;
N_burn=100;
thin=1;
settings.typegraph = 'simple';
settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;

weights_st = cell(1, T);

weights = rand(size(tw)); % random initialization of the weights
weights(end, :) = twrem; % keep the wrem the same as in the truth; zeros(1, size(weights,2))];

% initialise phi
% phi = true_phi;
%initialise counts
counts = poissrnd(phi.*weights);
counts(end,:) = crem;

[K T] = size(weights);
rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(T,1); % Leap-frog stepsize


% gvar = gamrnd(settings.g_a, 1/settings.g_b, 1, T);
%gvar= true_gvar;



N_samples = (N_Gibbs-N_burn)/thin;

for t = 1:N_Gibbs
    %     gvar=tgvar;
    if mod(t, 100)==0
        iter=t;
        iter
    end
    
    %
    [weights,  rate(:, t)] = ggp_sample_weights(weights, counts, tm, epsilon, alpha, sigma, tau, phi, gvar,  settings);
    counts = ggp_sample_C(counts, weights, phi, alpha, sigma, tau);
    
    
    if t<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:t), 2) - 0.6));
    end
    
    %
    
    
    [weights counts] = ggp_sample_weights_add(counts, weights, tm, alpha, sigma, tau, phi, gvar);
     %weights(end, :)
%     counts(end,:)
    
    
    
    if (t>N_burn && rem((t-N_burn),thin)==0)
        indd = ((t-N_burn)/thin);
        for k=1:T
            weights_st{k}(indd, :) = weights(:, k);
        end
        
        for k=1:T
            counts_st{k}(indd, :) = counts(:, k);
        end
        %         wst_samples(indd, :) = weights(end, :);
        %         cst_samples(indd, :) =counts(end, :);
        %
                    c_samples(:,:, indd) = counts(1:end-1, :);

    end
    
end


for t=1:T
         [~, indt{t}] = sort(tm(:,t), 'descend');
    %indt{t} = 1:K-1;
end



Na =size(tm,1);



%% Weights

for t=1:T
    figure
    for k=1:min(size(tw(1:end-1, :), 1), 50)
        plot([k, k],...
            quantile(weights_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(k, tw(indt{t}(k), t), 'xg', 'linewidth', 2)
    end
    xlim([0.1,min(Na, 50)+.5])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
    ylabel('Sociability parameter', 'fontsize', 16)
end


if 0
    for i=1:T
        figure
        % for k=1:nchains
        plot(thin:thin:N_samples,  wst_samples(:, i), '-r');
        hold on
        %end
        plot(thin:thin:N_samples, twrem(i)*ones(N_samples, 1), '--g', 'linewidth', 3);
        legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
        legend boxoff
        xlabel('MCMC iterations w rem', 'fontsize', 16);
        %ylabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
        box off
        xlim([0, N_samples])
        
        figure
        hist(wst_samples(:,i))
        hold on
        plot(twrem(i), 0, 'dg', 'markerfacecolor', 'g')
        xlabel('w rem', 'fontsize', 16);
        ylabel('Number of MCMC samples', 'fontsize', 16);
        
    end
    
end
%
%
%
%Counts

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

%
%
%



