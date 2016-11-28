clear all
close all


addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'

addpath '../../inference/utils/'



%
%
% seed=10;
% randn('seed', seed);
% rand('seed', seed);




alpha = 10; sigma = 0.1; tau = 10; % Parameters gamma process
phi = 10;                       % tunes dependence in dependent gamma process
rho = 0.001;                      % death rate for latent interactions
T = 4;                         % Number of time steps



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


settings.estimate_alpha=0;
settings.estimate_sigma=0;
settings.estimate_tau=0;
settings.rw_alpha = 0;
settings.rw_std(1) = 1;
settings.rw_std(2) = 1;
settings.hyper_alpha(1) = 1;
settings.hyper_alpha(2) = 1;
settings.hyper_tau(1)=1;
settings.hyper_tau(2)=1;
settings.hyper_sigma(1)=.5;
settings.hyper_sigma(2)=1;


[Z, w, c, KT,  N_new, N_old, N, M, indchain, indcnz] = ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings);
%tweights=w(:, 1:K(T));test_C

indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

wrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tweights = [tweights'; wrem' ];
K = size(tweights,1)-1;

tcrem =  sum(c(:, ~indlog),2);

tc=[c(:, indlog)';  tcrem'];

m = M(indchain, :);

%indlognz = false(size(indcnz)) ;
%indlognz(indcnz) = true;
indlognz = logical(indcnz);
wrem_J = sum(w.*indlognz',2)';
tJ = sum(indlognz);

indlogrep = repmat(indlog, T,1); % Tx Nall
wrem_rest =   sum(w.*(~indlognz' &  ~ indlogrep),2)';


tcrem
% Init C
N_Gibbs=8000;
N_burn = N_Gibbs*0.3;
thin = 1;
N_samples = (N_Gibbs-N_burn)/thin;

counts = tc;


rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(T,1); % Leapfrog stepsize




% Initialization
wrem =wrem';
crem =  poissrnd(phi*wrem);
nt= [0 crem(1:T-1)];


sum_w = sum(tweights(1:end-1, :));
for t=1:T
    fi = phi;
    if t==1
        fi = 0;
        
    else
        
        if nt(t) ==0
            J(t) =0;
        else
            
            [~, logC] = genfact(nt(t),sigma); % computes the log of the generalized factorial coefficients
            
            logtemp = logC(end, :) + (1:nt(t)).*log(alpha/sigma*(fi+tau)^sigma);
            temp = exp(logtemp - max(logtemp));
            J(t) = find(sum(temp)*rand<cumsum(temp), 1);
        end
        
    end
    
end

%w_rem = weights(end,:);
J_samples =  zeros(N_samples, T);
crem_samples =  zeros(N_samples, T);


for t=1:N_Gibbs
    if mod(t, 300)==0
        t
    end
    
    [ J crem] = ggp_sample_J_crem(crem, wrem, wrem_J, J, alpha, sigma, tau, phi);
    
    
    if (t>N_burn && rem((t-N_burn),thin)==0)
        indd = ((t-N_burn)/thin);
        
        J_samples(indd, :) = J;
        crem_samples(indd, :) = crem;
        
    end
    
    
    
    
    
end

for i=1:T
%     figure
%     % for k=1:nchains
%     plot(1:N_samples,  crem_samples(:, i));
%     hold on
%     %end
%     plot(1:N_samples, tcrem(i)*ones(N_samples, 1), '--g', 'linewidth', 3);
%     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
%     legend boxoff
%     xlabel('MCMC iterations', 'fontsize', 16);
%     ystr = sprintf('crem for location t=%d', i);
%     ylabel(ystr, 'fontsize', 16);
%     box off
%     xlim([0, N_Gibbs])
%     
    figure
    hist(crem_samples(:,i), 30)
    hold on
    plot(tcrem(i), 0, 'dg', 'markerfacecolor', 'g')
    %xlabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
    xstr = sprintf('crem at location t=%d', i);
    xlabel(xstr, 'fontsize', 16);
    ylabel('Number of MCMC samples', 'fontsize', 16);
end

% keyboard
% 
% 
% for i=1:T
% %     figure
% %     % for k=1:nchains
% %     plot(1:N_samples,  J_samples(:, i));
% %     hold on
% %     %end
% %     plot(1:N_samples, tJ(i)*ones(N_samples, 1), '--g', 'linewidth', 3);
% %     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
% %     legend boxoff
% %     xlabel('MCMC iterations', 'fontsize', 16);
% %     ystr = sprintf('J for location t=%d', i);
% %     ylabel(ystr, 'fontsize', 16);
% %     box off
% %     xlim([0, N_Gibbs])
% %     
%     figure
%     hist(J_samples(:,i), 30)
%     hold on
%     plot(tJ(i), 0, 'dg', 'markerfacecolor', 'g')
%     %xlabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
%     xstr = sprintf('J at location t=%d', i);
%     xlabel(xstr, 'fontsize', 16);
%     ylabel('Number of MCMC samples', 'fontsize', 16);
% end
