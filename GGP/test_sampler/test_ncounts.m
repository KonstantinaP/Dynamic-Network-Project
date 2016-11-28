% script that test the update of the nnew and nold interactions

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
alpha =10; sigma = 0; tau = 1; % Parameters gamma process
phi = 10;                       % tunes dependence in dependent gamma process
rho = 0;                      % death rate for latent interactions
T = 4;                         % Number of time steps
settings.dt=1;
settings.fromggprnd=1;
settings.typegraph = 'simple';
settings.onlychain=0;
settings.threshold=1e-5; 
settings.gcontrol=0;
settings.g_a=.1; 
settings.g_b=.1;
tgvar =ones(1,T);
% tgvar =gamrnd(settings.g_a, 1/settings.g_b, 1, T);


[Z, w, c, K, N_new, N_old, N, M, indlinks, indcnz]= ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, tgvar, settings);
% Z is symmetric (not in triangular form)
%N_new, N_old, N are all non-symmetric. Z is symmetric (resulted from
%taking the symmetric N)


% Keep only the nodes that appear active (at least one link) at some point
% in T. These are indicated in indlinks.
Nall = size(w,2);
tw=w;
indlog = false(1, K(T)) ;
indlog(indlinks) = true;
wrem =  sum(tw(:, ~indlog),2); % need to compute wrem so that we can use it as it is for the inference on weights
tw =tw(:, indlog); % keep only the nodes that are active at least one time over T.

%Increase the number of nodes by one to account for the unobserved ones.
tw = [tw'; wrem' ];
counts=[c(:, indlog)';  zeros(1, T)];

M = M(indlinks, :);
Z = Z(:, indlinks, indlinks);
% make the N matrices symmetric
for t=1:T
    
    tNnew(t,:,:) = squeeze(N_new(t, indlinks, indlinks)) + squeeze(N_new(t, indlinks, indlinks))' - diag(diag(squeeze(N_new(t, indlinks, indlinks))));
    tmnnew(:, t) = M(:, t);
    
    tNold(t,:,:) = squeeze(N_old(t, indlinks, indlinks)) + squeeze(N_old(t, indlinks, indlinks))' - diag(diag(squeeze(N_old(t, indlinks, indlinks))));
    tmnold(:, t) = sum(squeeze(tNold( t, :, :)),1)';
    
    tN(t,:,:) = squeeze(N(t,indlinks, indlinks)) + squeeze(N(t,indlinks, indlinks))' - diag(diag(squeeze(N(t,indlinks, indlinks))));
    tmnall(:, t) =  sum(squeeze(tN(t, :, :)), 1)';
end

indchain = indlinks;
for t=1:T
%    make it symmetric
    temp = squeeze(N_new(t, indchain, indchain)) + squeeze(N_new(t, indchain, indchain))'- diag(diag(squeeze(N_new(t, indchain, indchain)))); 
%    make it upper triangualr for the sampler to work correctly
    tn_new{t} =sparse( triu(temp, 1));
end



if strcmp(settings.typegraph, 'simple')
    issimple = true;
else
    issimple =false;
end

ind=cell(1, T);
trZ=cell(1, T);
linidxs = cell(1,T);
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

K=size(tw,1);



%% Initialize interaction counts
Nnew = cell(1, T);
Nold = cell(1, T);



k = size(tw,1)-1;
for t=1:T

    Nnew{t} = sparse(k,k);
    Nnew{t}(linidxs{t}) =1;
    Nold{t} = sparse(k,k);
    if t>1
    Nold{t}(linidxs{t}) = 1;
    end
    
    
end

if settings.gcontrol

gvar = gamrnd(settings.g_a, 1/settings.g_b, 1, T);
else
    gvar=tgvar;
end
%% Sampler
N_Gibbs=1000;
tic

for i=1:N_Gibbs
    if rem(i, 100)==0
        fprintf('i=%d\n', i)        
    end   
    
    % Sample new interaction counts
    for pp=1:T
         
        id=ind{pp};
        ind1=id(:,1);
        ind2=id(:,2);
        logw=log(tw(1:end-1, t)); % should be K-1 in the rows
        [new_inter, old_inter, Mn] = ggp_sample_interaction_counts(pp, trZ{pp}, logw, Nnew, Nold, ind1, ind2, t, gvar(pp), rho,  settings);
      
        Nnew{pp} = new_inter; %upper triangular (full info)
        Nold{pp} = old_inter; %upper triangular (full info)
        Ntot{pp} = new_inter + old_inter; % upper triangular (full info)
      
        mnnew(:, pp) = Mn; % matrix mm should be of size K-1 x T
        
       
        
   
        
       % additional structures for debugging
       mnold(:, pp) = sum(Nold{pp},1)' + sum(Nold{pp}, 2);% - diag(squeeze(N_new(t, :, :)));
       mnall(:,pp) = sum(Ntot{pp},1)' + sum(Ntot{pp}, 2);
       
            
            mnold_st{pp}(i, :) = mnold(:, pp);
            mnnew_st{pp}(i, :) = mnnew(:, pp);
        mnall_st{pp}(i, :) = mnall(:,pp);
          
    end
    
    
       
    
    
    
    if settings.gcontrol
        
    [gvar] = sample_gvar_mh(tw, Nnew, alpha, tau, phi, gvar, settings.g_a, settings.g_b);
    end
%     
            gvar_st(i, :) = gvar;
        
end



%Ntot here is upper triangular. 
toc


for t=1:T
    [~, indt{t}] = sort(M(:,t), 'descend');
end
Na =size(M,1);


% figure
% for t=1:T
%       hold on
%     
%         plot([t, t],...
%             quantile(gvar_st(:, t),[.025,.975]), 'r', ...
%             'linewidth', 3);
%      
%         plot(t, tgvar(t), 'xg', 'linewidth', 2)
% %     xlim(-1:T])
%     legend('95% credible intervals', 'True value')
%     legend boxoff
%     box off
%     
%     ylabel('gamma parameter', 'fontsize', 16)
% end
% % 
% 

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
    ylabel('counts parameter', 'fontsize', 16)
end

% for t=1:T
%     figure
%     for k=1:min(size(tw(1:end-1, :), 1), 50)
%         plot([k, k],...
%             quantile(mnold_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
%             'linewidth', 3);
%         hold on
%         plot(k, tmnold(indt{t}(k), t), 'xg', 'linewidth', 2)
%     end
%     xlim([0.1,min(Na, 50)+.5])
%     legend('95% credible intervals', 'True value')
%     legend boxoff
%     box off
%     xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
%     ylabel('counts parameter', 'fontsize', 16)
% end
% 
% for t=1:T
%     figure
%     for k=1:min(size(tw(1:end-1, :), 1), 50)
%         plot([k, k],...
%             quantile(mnnew_st{t}(:,indt{t}(k)),[.025,.975]), 'r', ...
%             'linewidth', 3);
%         hold on
%         plot(k, tmnnew(indt{t}(k), t), 'xg', 'linewidth', 2)
%     end
%     xlim([0.1,min(Na, 50)+.5])
%     legend('95% credible intervals', 'True value')
%     legend boxoff
%     box off
%     xlabel('Index of node (sorted by dec. degree)', 'fontsize', 16)
%     ylabel('counts parameter', 'fontsize', 16)
% end

