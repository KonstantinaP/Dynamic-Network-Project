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




talpha = 40; tsigma = 0.5; ttau = 10; % Parameters gamma process
tphi =10;                       % tunes dependence in dependent gamma process
rho = 0.1;                      % death rate for latent interactions
T =1;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-4;

settings.gcontrol =0;
gvar= 1:T;
settings.g_a = .1;
settings.g_b = .1;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;

N_Gibbs=20000;
N_burn = 0.3*N_Gibbs;
thin = 1;
N_samples = (N_Gibbs-N_burn)/thin;


settings.estimate_alpha=1;
settings.estimate_sigma=0;
settings.estimate_tau= 0;
settings.estimate_phi=0;
settings.rw_alpha = 0;
settings.rw_std(1) = .02; %sigma
settings.rw_std(2) = .02; %tau
settings.rw_std(3) = .02; %phi
settings.hyper_alpha(1) = 0;
settings.hyper_alpha(2) = 0;
settings.hyper_tau(1)=0;
settings.hyper_tau(2)=0;
settings.hyper_phi(1)=0;
settings.hyper_phi(2)=0;
settings.hyper_sigma(1)=0;
settings.hyper_sigma(2)=0;
[Z, w, c, KT,  N_new, N_old, N, M, indchain, indcnz]= ggp_dyngraphrnd(talpha, tsigma, ttau, T, tphi, rho, gvar, settings);


indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

twrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tweights = [tweights'; twrem' ];
K = size(tweights,1)-1;

crem =  sum(c(:, ~indlog),2);

tc=[c(:, indlog)';  crem'];


m = M(indchain, :);


indlognz = logical(indcnz);
twrem_J = sum(w.*indlognz',2)';
tJ = sum(indlognz);

indlogrep = repmat(indlog, T,1); % Tx Nall
twrem_rest =   sum(w.*(~indlognz' &  ~ indlogrep),2)';


%keyboard



rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(T,1); % Leapfrog stepsize




% Initialization

counts = tc;


%weights(end,:) =randn(1, T); %random init of the w_t_*
if settings.estimate_alpha
    alpha = 1;
else
    alpha =talpha;
end

if settings.estimate_sigma
    
    sigma=rand;
else
    sigma=tsigma;
end

if settings.estimate_tau
    tau=rand;
else
    tau=ttau;
end

if settings.estimate_phi
    phi=rand;
else
    phi=tphi;
end
%J=10*ones(1, T);

c_rem = counts(end, :);
co = [0 c_rem(1:T-1)];
nt= co;
J = zeros(1, T);

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

w_rem_rest = zeros(1, T);
%wprop_rem_J = zeros(1, T);
for t=1:T
    bt =phi;
    if t==1
        bt = 0;
    end
    w_rem_rest(t) = GGPsumrnd(alpha, sigma, tau+bt);
    
end
w_rem_J = gamrnd(nt-sigma.*J, 1/(tau + phi));
w_rem = w_rem_rest + w_rem_J;


wK = tweights(1:K, :);
%w_rem = weights(end,:);
wst_samples =  zeros(N_samples, T);
alpha_samples =  zeros(1, N_samples);
sigma_samples =  zeros(1, N_samples);
tau_samples =  zeros(1, N_samples);
phi_samples =  zeros(1, N_samples);
J_samples =  zeros(N_samples, T);
wrem_J_samples = zeros(N_samples, T);
wrem_rest_samples = zeros(N_samples, T);


for t=1:N_Gibbs
    if mod(t, 300)==0
        t
    end
    
    
    
    [w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, rate2] = ggp_sample_hyperparameters(wK, counts, w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, settings);
    %     wts
    
    
    if t<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:t), 2) - 0.6));
    end
    
    if (t>N_burn && rem((t-N_burn),thin)==0)
        indd = ((t-N_burn)/thin);
        acceptance_rate(indd) = rate2;
        wst_samples(indd, :) = w_rem;
        alpha_samples(indd) =  alpha;
        sigma_samples(indd) =  sigma;
        tau_samples(indd) =  tau;
                phi_samples(indd) =  phi;

         J_samples(indd, :) = J;
         wrem_J_samples(indd, :) = w_rem_J;
         wrem_rest_samples(indd, :) = w_rem_rest;
    end
    
    
    
    
    
end


for t=1:T
    [~, indt{t}] = sort(m(:,t), 'descend');
end
thin=1;
Na =size(m,1);

% 
% 
% %% plots for the w_rem
% figure
% for t=1:T
%     plot([t, t], quantile(wst_samples(:, t),[.025,.975]), 'r', ...
%         'linewidth', 3);
%     hold on
%     plot(t, tweights(end, t), 'xg', 'linewidth', 2)
%     
% end
% xlim([0.1,min(T, 50)+.5])
% legend('95% credible intervals', 'True value')
% legend boxoff
% box off
% xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
% ystr = sprintf('Sociability parameter w rem for all t');
% ylabel(ystr, 'fontsize', 16)
% 
% 
% for i=1:T
%     figure
%     % for k=1:nchains
%     plot(thin:thin:N_samples,  wst_samples(:, i), '-r');
%     hold on
%     %end
%     plot(thin:thin:N_samples, twrem(i)*ones(N_samples, 1), '--g', 'linewidth', 3);
%     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
%     legend boxoff
%     
%     xstr = sprintf('MCMC iterations w rem at location t=%d', i);
%         xlabel(xstr, 'fontsize', 16);
%     %ylabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
%     box off
%     xlim([0, N_samples])
%     
%     figure
%     hist(wst_samples(:,i))
%     hold on
%     plot(twrem(i), 0, 'dg', 'markerfacecolor', 'g')
%      xstr = sprintf('histogram for w rem samples at location t=%d', i);
%         xlabel(xstr, 'fontsize', 16);
%     
%     xlabel(xstr, 'fontsize', 16);
%     ylabel('Number of MCMC samples', 'fontsize', 16);
%     
% end


%% plot wrem_rest
figure
for t=1:T
    plot([t, t], quantile(wrem_rest_samples(:, t),[.025,.975]), 'r', ...
        'linewidth', 3);
    hold on
    plot(t, twrem_rest(t), 'xg', 'linewidth', 2)
    
end
xlim([0.1,min(T, 50)+.5])
legend('95% credible intervals', 'True value')
legend boxoff
box off
xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
ystr = sprintf('w rem rest for all t');
ylabel(ystr, 'fontsize', 16)



%% plot wrem_J
% figure
% for t=1:T
%     plot([t, t], quantile(wrem_J_samples(:, t),[.025,.975]), 'r', ...
%         'linewidth', 3);
%     hold on
%     plot(t, twrem_J(t), 'xg', 'linewidth', 2)
%     
% end
% xlim([0.1,min(T, 50)+.5])
% legend('95% credible intervals', 'True value')
% legend boxoff
% box off
% xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
% ystr = sprintf(' w rem J for all t');
% ylabel(ystr, 'fontsize', 16)


% %% plot for J
% figure
% for t=1:T
%     plot([t, t], quantile(J_samples(:, t),[.025,.975]), 'r', ...
%         'linewidth', 3);
%     hold on
%     plot(t, tJ(t), 'xg', 'linewidth', 2)
%     
% end
% xlim([0.1,min(T, 50)+.5])
% legend('95% credible intervals', 'True value')
% legend boxoff
% box off
% xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
% ystr = sprintf(' J for all t');
% ylabel(ystr, 'fontsize', 16)



%% plots for the hypers

if settings.estimate_alpha
    figure
    plot(thin:thin:N_samples, alpha_samples);
    
    hold on
    plot(thin:thin:N_samples, talpha*ones(N_samples, 1), 'r', 'linewidth', 3);
    plot(thin:thin:N_samples, cumsum(alpha_samples)./(thin:thin:N_samples), 'y-');
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel('\alpha', 'fontsize', 16);
    box off
    %xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
    
    
    
    
end

if settings.estimate_sigma
    figure
    plot(thin:thin:N_samples, sigma_samples);
    hold on
    plot(thin:thin:N_samples, tsigma*ones(N_samples, 1),'r', 'linewidth', 3);
    plot(thin:thin:N_samples, cumsum(sigma_samples)./(thin:thin:N_samples), 'g-');
    
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel('\sigma', 'fontsize', 16);
    box off
    %xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
    %
end

if settings.estimate_tau
    
    figure
    plot(thin:thin:N_samples, tau_samples);
    hold on
    plot(thin:thin:N_samples, ttau*ones(N_samples, 1), 'r', 'linewidth', 3);
    plot(thin:thin:N_samples, cumsum(tau_samples)./(thin:thin:N_samples), 'g-');
    
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel('\tau', 'fontsize', 16);
    box off
end

if settings.estimate_phi
    
    figure
    plot(thin:thin:N_samples, phi_samples);
    hold on
    plot(thin:thin:N_samples, tphi*ones(N_samples, 1), 'r', 'linewidth', 3);
    plot(thin:thin:N_samples, cumsum(phi_samples)./(thin:thin:N_samples), 'g-');
    
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel('\phi', 'fontsize', 16);
    box off
end

figure;
hist(acceptance_rate)