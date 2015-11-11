function [weights_st, alpha_st, phi_st, stats] = run_inference(Z, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings)

addpath 'utils/'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [lambda_st, a_st, rho_st, stats] = run_inference(X, K, a, rho, rho_param,
% N_Gibbs, N_burn)
%
% Gibbs sampler for the dynamic model
%
%%%%%%%%%%
% INPUTS
%
%   Z:      is a cell of length N. Each element is matrix of size Nt x Nt where the observed
%   interactions are stored between the Nt nodes at time t.
%   ID:     cell of length N. Each element is a vector referring to time t
%   and contains the unique ids of the nodes at time t.
%   alpha:  is the shape parameter for the gamma prior (default alpha = 1)
%   phi:    is the parameter tuning correlations between sociabilities at
%           different times
%   tau:    is the rate parameter of the gamma prior (default tau =1)
%   N_Gibbs:is the number of Gibbs iterations
%   N_burn: is the number of burn-in iterations
%
% NOTE: the max id in ID should be equal to the total number of nodes
% appearing over all the time steps.

%%%%%%%%%%
% OUTPUTS
% weights_st gives the values of the sociabilities at each iteration
% alpha_st gives the values of the shape parameter at each iteration
% stats is a structure with some summary statistics on the parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Initialization for the dynamic model
%tag = 1; % 1 indicates that weights do not have to appear in the likelihood before first appearance in data
tag=1;
[G, weights, K, nodes_step, ind] = init_model(Z, ID, alpha, tag, settings);
N = length(Z); % Number of time steps
K = K + 1; %Increase the number of nodes by one to account for the unobserved ones.


% Storage of some values
N_samples = (N_Gibbs-N_burn)/thin;
alpha_st = zeros(N_samples, 1);
weights_st = cell(1, N_samples);
% pause
for k=1:N
    lambda_st{k}= zeros(N_samples, K, 'single');
    pi_st{k}= zeros(N_samples, K, 'single');
end
a_st(1) = alpha;
phi_st = zeros(N_samples, 1);

% Init C
C = poissrnd(phi.*(weights(:, 1:N-1)));
C(isnan(weights(:, 1:N-1)))=0;


if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end


for t=1:N
    rr = weights(1:K-1, t)*weights(1:K-1, t)';
    
    nnew = poissrnd(rr);
    nnew = nnew + nnew' -diag(diag(nnew));
    
    if issimple % If no self-loops
        nnew = triu(nnew, 1);
        
    else
        nnew = triu(nnew);
    end
    Nnew(:, :, t) = nnew;
    
    
    
    if t==1
        deltat=0;
        nold = zeros(size(nnew));
        
        
        
    else
        deltat= settings.times(t) - settings.times(t-1);
        pi = exp(-settings.rho*deltat);
        nold = binornd(Nnew(:, :, t-1), pi.*ones(size(nnew)));
        
        
    end
    
    
    if issimple % If no self-loops
        nold = triu(nold, 1);
        
    else
        nold = triu(nold);
    end
    Nold(:, :, t) = nold;
    
    
end



fprintf('***************\n')
fprintf('START dynamic infinite model\n')
fprintf('***************\n')
fprintf('K=%d individuals\nT=%d time steps\nn=%d number of nodes per step %d | %d | %d \n', K-1, N, nodes_step)
fprintf('%d MCMC iterations\n', N_Gibbs);
fprintf('***************\n')


rate=zeros(N, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(N,1); % Leapfrog stepsize





% Iterate Gibbs sampler
tic
for i=1:N_Gibbs
    if rem(i, 100)==0
        fprintf('i=%d\n', i)
        fprintf('phi=%.2f\n', phi);
        fprintf('alpha=%.2f\n', alpha);
    end
    
    % Sample the total weights at each time and rescale the marginal weights
    % correspondingly
    weights = sample_totalweights(weights, phi, alpha, tau);
    logweights = log(weights(1:(K-1), :));
    
    % Sample the latent C (C_{tk} and c_{t\ast}) for correlation given weights in the Pitt-Walker
    % dependence.
    C = sample_C(C, weights, phi, alpha, tau);
    
    % Sample alpha and w_{t\ast}, c_{t\ast} again here after alpha update
    % (included in the sample_alpha function)
    if i>1
        [alpha, weights(end, :), C(end, :)] = sample_alpha( weights, C, alpha, phi, tau);
    end
    
    
    
    % Sample w_{t\ast} (included in alpha)
    
    % Sample new interaction counts
    for t=1:N
        id=ind{t};
        ind1=id(:,1);
        ind2=id(:,2);
        logw=logweights(:, t); % should be K-1 in the rows
        [new_inter, old_inter, Mn] = update_interaction_counts(t, G{t}, logw, Nnew, Nold, ind1, ind2, N, settings);
        Nnew(:,:, t) = new_inter;
        Nold(:,:, t) = old_inter;
        M(:, t) = Mn; % matrix M should be of size K-1 x T
    end
    
    % Sample weights
    [weights, rate(:, i)] = sample_weights(weights, C, M, epsilon, alpha, tau, phi, settings,issimple);
    logweights = weights(1:K-1, :);
    
    if i<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:i), 2) - 0.6));
    end
    
    
    %***********
    % Additional moves for irreducibilitiy at times/objects with no links
    %TO BE DONE ********************
    
    %     % Sample lambda and V at times/items with no observations
    %     [lambda, V] = sample_weights_add();
    %
    % Sample correlation
    if settings.sample_correlation
        phi_a = settings.phi_a;
        phi_b = settings.phi_b;
        %phi = sample_phi(phi, phi_a, phi_b, weights, alpha, tau);
        phi = slice_sample_phi(phi, phi_a, phi_b, weights, alpha, tau);
    end
    
    %     % Store outputs
    %     rem((i-N_burn),2)
    if (i>N_burn && rem((i-N_burn),thin)==0)
        %         lambda
        
        
        indd = ((i-N_burn)/thin);
        for k=1:N
            weights_st{k}(indd, :) = weights(:, k);
            
            indnotNaN = isnan(weights(:, k))==0;
        end
        
        alpha_st(indd) = alpha;
        phi_st(indd) = phi;
    end
    
    if i==1
        time = toc * N_Gibbs / 3600;
        fprintf('Estimated computation time: %.1f hours\n', time);
        fprintf('Estimated end of computation: %s \n', datestr(now + toc * N_Gibbs/3600/24));
        fprintf('***************\n')
    end
    
    
end
time = toc / 3600;
fprintf('***************\n')
fprintf('END dynamic model\n')
fprintf('***************\n')
fprintf('Computation time = %.1f hours\n', time)
fprintf('***************\n')





stats=[];
% Get some summary statistics
for k=1:N
    stats.weights_mean(:, k) = mean(weights_st{k});
    stats.weights_std(:, k) = std(weights_st{k});
    stats.weights_05(:, k) = quantile(weights_st{k}, .05);
    stats.weights_95(:, k) = quantile(weights_st{k}, .95);
end

 stats.alpha_mean = mean(alpha_st(N_burn+1:N_Gibbs, :));
 stats.alpha_std = std(alpha_st(N_burn+1:N_Gibbs, :));
