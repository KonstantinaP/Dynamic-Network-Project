clear all
close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
% create the Pitt-Walker model

% seed=4;
% rand('seed',seed);
% randn('seed',seed);

true_alpha=20;

sigma =0;
tau =1;
T=10;
phi=10;
settings.rho = 1; % decay rate. Controls the dependency on the interactions at the previous time step
settings.dt=1;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
settings.alpha_a = .1;
settings.alpha_b = .1;

N_Gibbs= 400;


at=zeros(1,N_Gibbs);
for s=1:N_Gibbs
    at(s) =  gamrnd(settings.alpha_a , 1/settings.alpha_b);
end
figure
%histfit(at,400, 'gamma')
hist(at, 30)
% keyboard
% 
% figure
% for s=1:N_Gibbs
% at(s) =  gamrnd(1 , 100);
% end
% hist(at, 30)



settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-4;


[Z, w, c, KT, N_new, N_old, N, M, indchain]= dyngraphrnd(true_alpha, sigma, tau, T, phi, settings);

wrem =  sum(w(:, ~indchain),2); % need to compute wrem so that we can use it as it is for the inference on weights
tweights = [w(:, indchain)'; wrem'];
crem =  sum(c(:, ~indchain), 2);
tcounts=[c(:, indchain)';  crem'];
m = M(indchain, :);


alpha_samples=zeros(1, N_Gibbs);

%initialize

nchains=3;
arate=0;
for ch=1:nchains
    alpha=gamrnd(settings.alpha_a, 1/settings.alpha_b);
%     if ch==2
%         alpha = true_alpha;
%     end

    alpha_samples(ch, 1)=alpha;
    for t=2:N_Gibbs
        if mod(t, 200)==0
            t
        end
        
        if rem(t,2)==0 % alternate between random walk and gamma proposals for alpha
%                   if t>N_Gibbs/2
            rw_alpha = true;
        else
            rw_alpha = false;
        end
%            rw_alpha= false;
        accept = 0;
        [alpha, accept]= sample_alpha(tweights, tcounts, alpha, phi, tau, settings.alpha_a, settings.alpha_b, rw_alpha);
        
        arate= arate+accept;
        alpha_samples(ch, t) = alpha;
    end
    
    
    
end
arate
arate/N_Gibbs

col = {'k', 'r', 'b', 'y'};
thin=1;
figure
for k=1:nchains
    
    plot(thin:thin:N_Gibbs, alpha_samples(k, :), col{k});
   
    
    hold on
    plot(thin:thin:N_Gibbs, true_alpha*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
    legend boxoff
    xlabel('MCMC iterations', 'fontsize', 16);
    ylabel('\alpha', 'fontsize', 16);
    box off
    %xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
    %
end

figure
hist(alpha_samples(:), 30)
hold on
plot(true_alpha, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\alpha', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

%



