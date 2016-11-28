
clear all
close all
addpath '../'
addpath '../GGPinference/'
addpath '../GGPsimulation/'
addpath '../../inference/utils/'


%seed=10;
%rand('seed',seed);
%randn('seed',seed);

%% Generate a path from the process.
talpha = 400; tsigma = 0.1; ttau = 2; % Parameters gamma process
tphi =2;                       % tunes dependence in dependent gamma process
trho = 0.1;                      % death rate for latent interactions
T =1;                         % Number of time steps
tgvar=ones(1,T);
settings.dt=1;
settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-5;

settings.gcontrol =0;
gvar= 1:T;
settings.g_a = .1;
settings.g_b = .1;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;

N_Gibbs=2000;
N_burn = 0.3*N_Gibbs;
thin = 1;
N_samples = (N_Gibbs-N_burn)/thin;

settings.typegraph = 'simple';
settings.estimate_alpha=0;
settings.estimate_sigma=1;
settings.estimate_tau= 1;
settings.estimate_phi=0;
settings.estimate_rho=0;
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

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;


[Z, w, c, KT,  N_new, N_old, Nall, M, indchain, indcnz]= ggp_dyngraphrnd(talpha, tsigma, ttau, T, tphi, trho, tgvar, settings);




indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

twrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; twrem' ];

tcrem =  sum(c(:, ~indlog),2);
tc=[c(:, indlog)';  tcrem'];

tm = M(indchain, :);


indlognz = logical(indcnz);
twrem_J = sum(w.*indlognz',2)';
tJ = sum(indlognz);

indlogrep = repmat(indlog, T,1); % Tx Nall
twrem_rest =   sum(w.*(~indlognz' &  ~ indlogrep),2)';


for t=1:T
    %    make it symmetric
    temp = squeeze(N_new(t, indchain, indchain)) + squeeze(N_new(t, indchain, indchain))'- diag(diag(squeeze(N_new(t, indchain, indchain))));
    %    make it upper triangualr for the sampler to work correctly
    tn_new{t} =sparse( triu(temp, 1));
    
    %    make it symmetric
    tempo = squeeze(N_old(t, indchain, indchain)) + squeeze(N_old(t, indchain, indchain))'- diag(diag(squeeze(N_old(t, indchain, indchain))));
    %    make it upper triangualr for the sampler to work correctly
    tmnold(:, t) = sum(tempo,1)';
    
    tN(t,:,:) = squeeze(Nall(t,indchain, indchain)) + squeeze(Nall(t,indchain, indchain))' - diag(diag(squeeze(Nall(t,indchain, indchain))));
    tmnall(:, t) =  sum(squeeze(tN(t, :, :)), 1)';
    
end




Z = Z(:, indlog, indlog); % this is symmetric. Need to make it upper triangular for the sampler to work correctly.

ind=cell(1, T);
trZ=cell(1, T);
linidxs = cell(1,T);
issimple=1;
for t=1:T
    G= squeeze(Z(t, :, :));
    
    if issimple % If no self-loops
        G2 = triu((G+G')>0, 1);% G2 upper triangular
        
    else
        G2 = triu((G+G')>0);
    end
    linidxs{t} = find(G2);
    [ind1, ind2] = find(G2); % the indices refer to the upper triangle
    
    
    ind{t} =[ind1 ind2];
    
    
    % [nind1, nind2] = find(temp + G2 == 0);
    % nind{t} = [ids(nind1)' ids(nind2)'];
    trZ{t} = sparse(G2);
end


%% Initialise structures for posterior sampling

[kk T] = size(tw);
rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(kk-1)^(1/4).*ones(T,1); % Leap-frog stepsize

if settings.estimate_alpha
    alpha = 1;
else
    alpha =talpha;
end

if settings.estimate_sigma
    
    sigma=0.9;
else
    sigma=tsigma;
end

if settings.estimate_tau
    tau=1;
else
    tau=ttau;
end

if settings.estimate_phi
    phi=1;
else
    phi=tphi;
end
    

% gvar = gamrnd(settings.g_a, 1/settings.g_b, 1, T);
gvar= tgvar;


if settings.estimate_rho
    rho = gamrnd(settings.rho_a, 1/settings.rho_b);
else
    rho = trho;
end

weights_st = cell(1, T);

weights = rand(size(tw)); % random initialization of the weights

%initialise counts
counts = poissrnd(phi.*weights);
c_rem = counts(end, :);
co = [0 c_rem(1:T-1)];
nt= co;
J = zeros(1, T);

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

%% Initialize interaction counts
Nnew = cell(1, T);
Nold = cell(1, T);

dt=1;
k = kk-1;
nal = cell(1,T);
for t=1:T
    logw = log(weights(:,t));
    Nnew{t} = sparse(k,k);
    iid1=ind{t}(:,1);
    iid2=ind{t}(:,2);
    lograte_poi = log(2) + log(gvar(t))+logw(iid1) + logw(iid2);
    lograte_poi(iid1==iid2) = log(gvar(t))+2*logw(iid1(iid1==iid2));
    
    Nnew{t}(linidxs{t}) =  poissrnd(exp(lograte_poi));
    if t==1
        Nold{t} = sparse(k,k);
    else
        %     if t>1
        Nold{t} = sparse(k,k);
        
        Nold{t}(linidxs{t}) = binornd(nal{t-1}(linidxs{t}), exp(-rho*dt) );
        % end/
    end
    nal{t} = Nnew{t}+ Nold{t};
    
end
 


wst_samples =  zeros(N_samples, T);
alpha_samples =  zeros(1, N_samples);
sigma_samples =  zeros(1, N_samples);
tau_samples =  zeros(1, N_samples);
phi_samples =  zeros(1, N_samples);
J_samples =  zeros(N_samples, T);
wrem_J_samples = zeros(N_samples, T);
wrem_rest_samples = zeros(N_samples, T);


for t = 1:N_Gibbs
    %     gvar=tgvar;
    if mod(t, 100)==0
        iter=t;
        iter
    end
    
    
    % Sample new interaction counts
    for pp=1:T
        
        id=ind{pp};
        ind1=id(:,1);
        ind2=id(:,2);
        logw=log(weights(1:end-1, pp)); % should be K-1 in the rows
        [new_inter, old_inter, Mn] = ggp_sample_interaction_counts(pp, trZ{pp}, logw, Nnew, Nold, ind1, ind2, T, gvar(pp), rho,  settings);
        
        Nnew{pp} = new_inter; %upper triangular (full info)
        Nold{pp} = old_inter; %upper triangular (full info)
        Ntot{pp} = new_inter + old_inter; % upper triangular (full info)
        
        mnnew(:, pp) = Mn; % matrix mm should be of size K-1 x T
        
        % additional structures for debugging
        mnold(:, pp) = sum(Nold{pp},1)' + sum(Nold{pp}, 2);% - diag(squeeze(N_new(t, :, :)));
        mnall(:, pp) = sum(Ntot{pp},1)' + sum(Ntot{pp}, 2);
        
        
        mnold_st{pp}(t, :) = mnold(:, pp);
        mnnew_st{pp}(t, :) = mnnew(:, pp);
        mnall_st{pp}(t, :) = mnall(:,pp);
    end  
    
    

    
    
    
    
    
    %
    [weights,  rate(:, t)] = ggp_sample_weights(weights, counts, mnnew, epsilon, alpha, sigma, tau, phi, gvar,  settings);
                            
    counts = ggp_sample_C(counts, weights, phi, alpha, sigma, tau);
        
    [weights counts] = ggp_sample_weights_add(counts, weights, mnnew, alpha, sigma, tau, phi, gvar);
     
    [ J crem] = ggp_sample_J_crem(c_rem, w_rem, w_rem_J, J, alpha, sigma, tau, phi);
    counts(end, :)=crem;

    [w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, rate2] = ggp_sample_hyperparameters(weights(1:end-1, :), counts, w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, settings);
weights(end, :) = w_rem;
%     keyboard
    %%%%
    
%     rw_alpha=0;
%     if rem(t,2)==0
%         rw_alpha=1;
%     end
    
    
    if settings.gcontrol
        [gvar] = sample_gvar_mh(weights, tn_new, alpha, tau, phi, gvar, settings.g_a, settings.g_b);
        %     [gvar] = slice_sample_g(weights, gvar, tn_new, settings.g_a, settings.g_b);
    end
    
    
    
    if settings.estimate_rho
        [rho] = slice_sample_rho(rho, tn_old, tn_new, settings.rho_a, settings.rho_b, settings.dt);
        
    end
    
    
    
    
    if (t>N_burn && rem((t-N_burn),thin)==0)
        indd = ((t-N_burn)/thin);
        for k=1:T
            weights_st{k}(indd, :) = weights(:, k);
        end
        
        for k=1:T
            counts_st{k}(indd, :) = counts(:, k);
        end
        wst_samples(indd, :) = weights(end, :);
        cst_samples(indd, :) =counts(end, :);
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
    [~, indt{t}] = sort(tm(:,t), 'descend');
end


Na =size(mnnew,1);



%% Plot the Weights

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


% plot the remaining mass
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

%keyboard
%
%
%
% Plot the Counts

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
% Plot alpha 


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
    
    figure
hist(alpha_samples, 30)
hold on
plot(talpha, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\alpha', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

    
    
end





%% Plot phi
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
    
    
    figure
hist(phi_samples, 30)
hold on
plot(tphi, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\phi', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

end

% plot tau
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
    
        figure
hist(tau_samples, 30)
hold on
plot(ttau, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\tau', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

end

% plot sigma
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
        figure
hist(sigma_samples, 30)
hold on
plot(tsigma, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\sigma', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

    %
end


for t=1:T
    figure
    for k=1:min(size(tw(1:end-1, :), 1), 50)
        plot([k, k],...
            quantile(mnall_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
            'linewidth', 3);
        hold on
        plot(k, tmnall(indt{t}(k), t), 'xg', 'linewidth', 2)
    end
    xlim([0.1,min(Na, 50)+.5])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
    ylabel('interaction counts mnall parameter', 'fontsize', 16)
end
%
