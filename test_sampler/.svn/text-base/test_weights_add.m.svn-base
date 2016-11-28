close all
clear all
addpath '../'
addpath '../inference/'
addpath '../inference/utils/'
%seed=55; %alpha=3;
% seed=2;
% rand('seed',seed);
% randn('seed',seed);


%% Generate a path from the process.
true_alpha = 10; sigma = 0; tau = 1; % Parameters gamma process
true_phi = 50;                       % tunes dependence in dependent gamma process
true_rho=0; % death rate for latent interactions
sample_rho =0;
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
true_gvar=ones(1,T);
% Number of time steps
[Z, w, c, KT,  N_new, N_old, Nall, M, indchain] = dyngraphrnd(true_alpha, sigma, tau, T, true_phi, true_rho, true_gvar, settings);

indlog = false(1, length(indchain)) ;
indlog(indchain) = true;

twrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; twrem' ];

crem =  sum(c(:, ~indlog),2);
tc=[c(:, indlog)';  crem'];

tm = M(indchain, :);


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
%wts(end, :) = wrem; % keep the wrem the same as in the truth; zeros(1, size(weights,2))];

% initialise phi
phi=gamrnd(settings.phi_a, 1/settings.phi_b);
% phi = true_phi;
%initialise counts
counts = poissrnd(phi.*weights);

[K T] = size(weights);
rate=zeros(T, N_Gibbs);
epsilon = settings.leapfrog.epsilon/(K-1)^(1/4).*ones(T,1); % Leap-frog stepsize

alpha=true_alpha;
%alpha=0.1;

% gvar = gamrnd(settings.g_a, 1/settings.g_b, 1, T);
gvar= true_gvar;


if sample_rho
    rho = gamrnd(settings.rho_a, 1/settings.rho_b);
else
    rho = true_rho;
end

%% Initialize interaction counts
Nnew = cell(1, T);
Nold = cell(1, T);

dt=1;
k = K-1;
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
% keyboard

N_samples = (N_Gibbs-N_burn)/thin;

for t = 1:N_Gibbs
    %     gvar=tgvar;
    if mod(t, 100)==0
        iter=t;
        iter
    end
    
    
    % Sample new interaction counts
    % Sample new interaction counts
    for pp=1:T
        
        id=ind{pp};
        ind1=id(:,1);
        ind2=id(:,2);
        logw=log(weights(1:end-1, pp)); % should be K-1 in the rows
        [new_inter, old_inter, Mn] = update_interaction_counts_ii(pp, trZ{pp}, logw, Nnew, Nold, ind1, ind2, T, gvar(pp), rho,  settings);
        
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
    [weights,  rate(:, t)] = sample_weights(weights, counts, mnnew, epsilon, alpha, tau, phi, gvar,  settings);
    
    if t<settings.leapfrog.nadapt % Adapt the stepsize
        epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:t), 2) - 0.6));
    end
    
    
    %
    
    counts = sample_C(counts, weights, phi, alpha, tau);
    
    [weights counts] = sample_weights_add(counts, weights, mnnew, alpha, phi, tau, gvar);
    
    
    
    [weights] = sample_Wst(weights, counts, alpha, phi, tau, gvar);
    
    rw_alpha=0;
    if rem(t,2)==0
        rw_alpha=1;
    end
    
    [alpha]= sample_alpha(weights, counts, alpha, phi, tau, settings.alpha_a, settings.alpha_b, rw_alpha);
    
    if settings.gcontrol
        [gvar] = sample_gvar_mh(weights, tn_new, alpha, tau, phi, gvar, settings.g_a, settings.g_b);
        %     [gvar] = slice_sample_g(weights, gvar, tn_new, settings.g_a, settings.g_b);
    end
    
    [phi ] = sample_phi(phi, settings.phi_a, settings.phi_b, weights, alpha, tau)   ;
    %         phi = slice_sample_phi(phi, phi_a, phi_b, tw, alpha, tau);
    
    if sample_rho
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
        
        gvar_st(indd, :) = gvar;
        alpha_st(indd) = alpha;
        phi_st(indd) = phi;
    end
    
end


for t=1:T
    %     [~, indt{t}] = sort(tm(:,t), 'descend');
    indt{t} = 1:K-1;
end



Na =size(mnnew,1);

%
% % gvar
figure
for t=1:T
    hold on
    
    plot([t, t],...
        quantile(gvar_st(:, t),[.025,.975]), 'r', ...
        'linewidth', 3);
    
    plot(t, true_gvar(t), 'xg', 'linewidth', 2)
    %     xlim(-1:T])
    legend('95% credible intervals', 'True value')
    legend boxoff
    box off
    
    ylabel('gamma parameter', 'fontsize', 16)
end
% %

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
figure
plot(thin:thin:N_samples, alpha_st(:));
hold on

hold on
plot(thin:thin:N_samples, true_alpha*ones(N_samples, 1), '--g', 'linewidth', 3);
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\alpha', 'fontsize', 16);
box off
%xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
%

figure
hist(alpha_st(:), 30)
hold on
plot(true_alpha, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\alpha', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);





%% phi

figure
hold on
plot(thin:thin:N_samples,  phi_st);


%end
plot(thin:thin:N_samples, true_phi*ones(N_samples, 1), '--g', 'linewidth', 3);
legend({'Chain 1','Chain 2',  'Chain 3', 'True'}, 'fontsize', 16, 'location', 'Best')
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\phi', 'fontsize', 16);
box off
xlim([0, N_samples])

figure
hist(phi_st, 30)
hold on
plot(true_phi, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\phi', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);


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
    ylabel('interaction counts parameter', 'fontsize', 16)
end
%
