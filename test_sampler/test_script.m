clear all
close all

addpath '../'
addpath '../inference/'
addpath '../inference/utils/'

seed=11;
rand('seed',seed);
randn('seed',seed);


true_alpha=5;
sigma =0;
tau =1;
T =10;
true_phi= 20;
rho =1;
settings.dt=1;

settings.leapfrog.epsilon = 0.1;
settings.leapfrog.L = 10;
settings.leapfrog.nadapt=0;
settings.alpha_a = 5    ;
settings.alpha_b = 1 ;
settings.phi_a = 5;
settings.phi_b = 1;
settings.rho = rho; % decay rate. Controls the dependency on the interactions at the previous time step
settings.typegraph = 'simple';

settings.sample_correlation = 0;

settings.fromggprnd=1;
settings.onlychain=0;
settings.threshold=1e-5;
[Z, w, c, KT, N_new, N_old, N, M, indchainm]= dyngraphrnd(true_alpha, sigma, tau, T, true_phi,  settings);
% output Z symmetric

tm  = M(indchainm, :);
z=Z(:, indchainm, indchainm);

indlog = false(1, length(indchainm)) ;
indlog(indchainm) = true;

twrem =  sum(w(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =w(:, indlog);

tc=c(:, indlog);
tcrem = sum(c(:, ~indlog), 2);
tw =[tw'; twrem'];
tc = [tc'; tcrem'];

N_Gibbs= 3000;
rate=zeros(T, N_Gibbs);
[K T] = size(tw);
k=K-1;
epsilon = settings.leapfrog.epsilon/(k-1)^(1/4).*ones(T,1); % Leapfrog stepsize


% make the N matrices symmetric
indlinks=indchainm;
for t=1:T
    
    tNnew(t,:,:) = squeeze(N_new(t, indlinks, indlinks)) + squeeze(N_new(t, indlinks, indlinks))' - diag(diag(squeeze(N_new(t, indlinks, indlinks))));
    tmnnew(:, t) = tm(:, t);
    
    tNold(t,:,:) = squeeze(N_old(t, indlinks, indlinks)) + squeeze(N_old(t, indlinks, indlinks))' - diag(diag(squeeze(N_old(t, indlinks, indlinks))));
    tmnold(:, t) = sum(squeeze(tNold( t, :, :)),1)';
    
    tN(t,:,:) = squeeze(N(t,indlinks, indlinks)) + squeeze(N(t,indlinks, indlinks))' - diag(diag(squeeze(N(t,indlinks, indlinks))));
    tmnall(:, t) =  sum(squeeze(tN(t, :, :)), 1);
end

if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end

ind=cell(1, T);
trZ=zeros(size(z));
for t=1:T
    G= squeeze(z(t, :, :));
   
    if issimple % If no self-loops
        G2 = triu((G+G')>0, 1);% G2 upper triangular
        temp = tril(ones(size(G2)));
    else
        G2 = triu((G+G')>0);
        temp = tril(ones(size(G2)), -1);
    end
     ids = find(sum(G2,1));
    [ind1, ind2] = find(G2); % the indices refer to the upper triangle
    
 
    ind{t} =[ind1 ind2];
    
    
   % [nind1, nind2] = find(temp + G2 == 0);
   % nind{t} = [ids(nind1)' ids(nind2)'];
    trZ(t, :, :) = G2;
end


K=size(tw,1);



%% Initialize interaction counts
Nnew = zeros(K-1, K-1, T);
Nold = zeros(K-1, K-1, T);

for t=1:T
    
    % Init Nnew
    rr = tw(1:end-1, t)*tw(1:end-1, t)';
    
    nnew = poissrnd(rr)+1; % I added one to test
    nnew = nnew + nnew' -diag(diag(nnew));
    
    Nnew(:, :, t) = triu(nnew, 1)*(issimple) + triu(nnew)*(~issimple); % Nnew is upper triangular
    
    % initialize Nold
    if t==1
        deltat=0;
        nold = zeros(size(nnew));
    else
        deltat= settings.dt;
        pi = exp(-settings.rho*deltat);
        nold = binornd(Nnew(:, :, t-1)+Nold(:, :, t-1), pi.*ones(size(nnew)));
    end
    
    Nold(:, :, t) =triu(nold, 1)*(issimple) + triu(nold)*(~issimple);
end
% keyboard

phi_samples=zeros(1, N_Gibbs);
phi=gamrnd(settings.phi_a, 1/settings.phi_b);
% phi=true_phi;
weights = rand(size(tw));
counts = poissrnd(phi.*(weights));
arate=0;
alpha_samples=zeros(1, N_Gibbs);
alpha=gamrnd(settings.alpha_a, 1/settings.alpha_b);
% alpha=true_alpha;

for t=1:N_Gibbs
    if mod(t, 100)==0
        iter=t;
        iter
    end
    
    
%      % Sample new interaction counts
%     for tim=1:T
%         id=ind{tim};
%         ind1=id(:,1);
%         ind2=id(:,2);
%         logw=log(weights(1:end-1, tim)); % should be K-1 in the rows
%         [new_inter, old_inter, Mn] = update_interaction_counts_ii(tim, trZ(tim, :, :), logw, Nnew, Nold, ind1, ind2, T, settings);
%         Nnew(:,:, tim) = new_inter; %upper triangular (full info)
%         Nold(:,:, tim) = old_inter; %upper triangular (full info)
%         Ntot(:, :, tim) = new_inter + old_inter; % upper triangular (full info)
%         mnnew(:, tim) = Mn; % matrix mm should be of size K-1 x T
%         
%        % additional structures for debugging
%        mnold(:, tim) = sum(squeeze(Nold( :, :, tim)),1)' + sum(squeeze(Nold(:, :, tim)), 2);% - diag(squeeze(N_new(t, :, :)));
%        mnall(:, tim) = sum(Ntot(:, :, tim),1)' + sum(Ntot(:, :, tim), 2);
%        
%             
%             mnold_st{tim}(t, :) = mnold(:, tim);
%             mnnew_st{tim}(t, :) = mnnew(:, tim);
%         mnall_st{tim}(t, :) = mnall(:,tim);
%           
%     end


 mnnew=tm;
    
    [weights,  rate(:, t)] = sample_weights(weights, counts, mnnew, epsilon, alpha, tau, phi, settings);
         if t<settings.leapfrog.nadapt % Adapt the stepsize
            epsilon = exp(log(epsilon) + .01*(mean(rate(:,1:t), 2) - 0.6));
        end
    
        
    counts = sample_C(counts, weights, phi, alpha, tau);
    
    [weights counts] = sample_weights_add(counts, weights, mnnew, alpha, phi, tau);

        cst_samples(t, :) =counts(end, :);

      
    
    
      for k=1:T
        weights_st{k}(t, :) = weights(:, k);
    end
    
    for k=1:T
        counts_st{k}(t, :) = counts(:, k);
    end
    
          
    [weights] = sample_Wst(weights, counts, alpha, phi, tau);

    wst_samples(t, :) = weights(end, :);

    
    % Sample parameter alpha
    if rem(t,2)==0 % alternate between random walk and gamma proposals for alpha
        %                   if t>N_Gibbs/2
        rw_alpha = true;
    else
        rw_alpha = false;
    end
    rw_alpha=false;
    accept = 0;
    [alpha, accept]= sample_alpha(weights, counts, alpha, phi, tau, settings.alpha_a, settings.alpha_b, rw_alpha);
    
%     alpha = true_alpha;
    alpha_samples(t) = alpha;
    
    % Sample correlation phi
%     weights=tw;
   [phi ] = sample_phi(phi, settings.phi_a, settings.phi_b, weights, alpha, tau, accept)   ;
%    phi=true_phi;
    phi_samples(t)=phi;
    
    
  
    %
    %
    
    
    
 
   
    
end




for t=1:T
    [~, indt{t}] = sort(tm(:,t), 'descend');
end
thin=1;
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



for i=1:T
    figure
    % for k=1:nchains
    plot(thin:thin:N_Gibbs,  wst_samples(:, i), '-r');
    hold on
    %end
    plot(thin:thin:N_Gibbs, twrem(i)*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
    legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
    legend boxoff
    xlabel('MCMC iterations w rem', 'fontsize', 16);
    %ylabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
    box off
    xlim([0, N_Gibbs])
    
    figure
    hist(wst_samples(:,i))
    hold on
    plot(twrem(i), 0, 'dg', 'markerfacecolor', 'g')
    xlabel('w rem', 'fontsize', 16);
    ylabel('Number of MCMC samples', 'fontsize', 16);
    
end


% figure
% for i=1:T
%
%     plot([i, i], quantile(wst_samples(:, i),[.025,.975]), 'r', ...
%         'linewidth', 3);
%     hold on
%     plot(i, twrem(i), 'xg', 'linewidth', 2)
%
%     xlim([0.5, T+0.5])
%     legend('95% credible intervals', 'True value')
%     legend boxoff
%     box off
%     xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
%     ylabel('w rem', 'fontsize', 16)
% end
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

% %
% for i=1:T
%     figure
%     plot(thin:thin:N_Gibbs,  cst_samples(:, i), '-r');
%     hold on
%
%     plot(thin:thin:N_Gibbs, tcrem(i)*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
%     legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
%     legend boxoff
%     xlabel('MCMC iterations c rem', 'fontsize', 16);
%     %ylabel([namesvar{3} ' time ' num2str(i)], 'fontsize', 16);
%     box off
%     xlim([0, N_Gibbs])
%
%     figure
%     hist(cst_samples(:,i))
%     hold on
%     plot(tcrem(i), 0, 'dg', 'markerfacecolor', 'g')
%     xlabel('crem ' , 'fontsize', 16);
%     xlabel('c rem', 'fontsize', 16);
%     ylabel('Number of MCMC samples', 'fontsize', 16);
%
%
%
%
%
% end



% %

%% Alpha plots

figure
plot(thin:thin:N_Gibbs, alpha_samples(:));
hold on

hold on
plot(thin:thin:N_Gibbs, true_alpha*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\alpha', 'fontsize', 16);
box off
%xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])
%

figure
hist(alpha_samples(:), 30)
hold on
plot(true_alpha, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\alpha', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);

% 
% %% interactions Nall parameter
% for t=1:T
%     figure
%     for k=1:min(size(tw(1:end-1, :), 1), 50)
%         plot([k, k],...
%             quantile(mnall_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
%             'linewidth', 3);
%         hold on
%         plot(k, tmnall(indt{t}(k), t), 'xg', 'linewidth', 2)
%     end
%     xlim([0.1,min(Na, 50)+.5])
%     legend('95% credible intervals', 'True value')
%     legend boxoff
%     box off
%     xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
%     ylabel('interactions counts parameter (Nall)', 'fontsize', 16)
% end


%% Phi plots
figure 

    hold on
    plot(thin:thin:N_Gibbs,  phi_samples);
plot(thin:thin:N_Gibbs, true_phi*ones(N_Gibbs, 1), '--g', 'linewidth', 3);
legend({'Chain 1', 'True'}, 'fontsize', 16, 'location', 'Best')
legend boxoff
xlabel('MCMC iterations', 'fontsize', 16);
ylabel('\phi', 'fontsize', 16);
box off
xlim([0, N_Gibbs])

figure
hist(phi_samples, 30)
hold on
plot(true_phi, 0, 'dg', 'markerfacecolor', 'g')
xlabel('\phi', 'fontsize', 16);
ylabel('Number of MCMC samples', 'fontsize', 16);




