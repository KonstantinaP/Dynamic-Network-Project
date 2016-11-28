function [weights_st, alpha_st, phi_st, stats, weights, C, Nnew] = run_inference_debug(Nnew, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings)

addpath 'utils/'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [weights_st, alpha_st, phi_st, stats, weights] = run_inference(Z, ID, alpha, tau, phi, N_Gibbs, N_burn, thin, settings)
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




% Initialization of the dynamic model
%tag = 1; % 1 indicates that weights do not have to appear in the likelihood before first appearance in data
tag=0;


N = size(Nnew, 1);


K = max(cell2mat(ID)); % K = total number of nodes appearing in the dynamic  network of all time steps.

appear =zeros(K+1, N);
for t=1:N
    appear(ID{t}, t) = 1;
    
end

appear(K+1, :) =  alpha;
                        
weights = ones(K+1, N); %Increase the number of nodes by one to account for the unobserved ones.
if tag
    for k=1:K+1
        first_time = find(appear(k, :), 1);
        weights(k, 1:(first_time-1)) = NaN;
    end
end

% `weights' is of size K+1 x N 
Kplus = size(weights,1); % Increased number of nodes by one to account for the unobserved ones. K updated here

% Storage of some values
N_samples = (N_Gibbs-N_burn)/thin;
alpha_st = zeros(N_samples, 1);
weights_st = cell(1, N); 
phi_st = zeros(N_samples, 1);
for k=1:N
    weights_st{k}= zeros(N_samples, Kplus, 'single');    
end
alpha_st(1) = alpha;


% Init C
C = poissrnd(phi.*(weights(:, 1:N-1)));
C(isnan(weights(:, 1:N-1)))=0;


if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end


    %interaction counts
    for t=1:N    
        
        
        M(:, t) = sum(squeeze(Nnew(t, :,:)),1)' + sum(squeeze(Nnew(t, :,:)), 2); % matrix M should be of size K-1 x T
    end
 


fprintf('***************\n')
fprintf('START dynamic infinite model\n')
fprintf('***************\n')
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
        fprintf('\n\n');
    end
    
    % Sample the total weights at each time and rescale the marginal weights
    % correspondingly
    
    % weights = sample_totalweights(weights, phi, alpha, tau);
    logweights = log(weights(1:(Kplus-1), :));
    
    % Sample the latent C (C_{tk} and c_{t\ast}) for correlation given weights in the Pitt-Walker
    % dependence.
    if N>1
        C = sample_C(C, weights, phi, alpha, tau);
    end
    
    % Sample alpha and w_{t\ast}, c_{t\ast} again here after alpha update
    % (included in the sample_alpha function)
    
    
    if i>1
        alpha_a =settings.alpha_a;
        alpha_b = settings.alpha_b;
        [alpha, weights(end, :), C(end, :)] = sample_alpha( weights, C, alpha, phi, tau, alpha_a, alpha_b);
        
    end
    
    
    
    % Sample w_{t\ast} (included in alpha)
    
    

    % Sample weights
    [weights, rate(:, i)] = sample_weights(weights, C, M, epsilon, alpha, tau, phi, settings,issimple);
    
    logweights = log(weights(1:(K-1), :));
    
    if i<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:i), 2) - 0.6));
    end
    
    
    %***********
    % Additional moves for irreducibilitiy  
    
    [weights, C] = sample_weights_add(C, weights, M, alpha, phi, tau);
    logweights = log(weights(1:(Kplus-1), :));


    
    % Sample correlation
    if settings.sample_correlation && N>1
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

stats.alpha_mean = mean(alpha_st);
stats.alpha_std = std(alpha_st);
