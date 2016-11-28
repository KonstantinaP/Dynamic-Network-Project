function [Z, w, c, K, Knew, N_new, N_old, N, M, indchainmT, indchainmL, indchainzT, indlinks ]  = dyngraphrnd_urn(alpha, sigma, tau, T, phi, rho, dt, settings)

indmax = 1000; % so that to allocate memory
c = zeros(T+1, indmax);
w = zeros(T+1, indmax);

%% Sample the latent C_1,C_2,...
K = zeros(T, 1);
w1_all = gamrnd(alpha, 1/tau);
c1_all = poissrnd(phi*w1_all);
if c1_all>0
    [~, m, K(1)] = crprnd(alpha, c1_all);
    for i=1:K(1)
        c(1, i) = m(i);
    end
else
    K(1) = 0;
end


for t=2:T+1
    
    %% Sample counts for existing atoms
    ind = c(t-1,:)>0;
    w(t, ind) = gamrnd(c(t-1,ind), 1/(tau+phi));
    c(t,ind) = poissrnd(phi*w(t, ind));
    
    % Sample counts for new atoms
    w_rest = gamrnd(alpha, 1/(tau+phi));
    c_rest = poissrnd(phi * w_rest);
    if c_rest>0
        [~, m, K_new] = crprnd(alpha, c_rest);
        for i=1:K_new
            c(t, K(t-1)+i) = m(i);
        end
        K(t) = K(t-1) + K_new;
    else
        K(t) = K(t-1);
    end
end



if T==1
    c= c(1, 1:K(1));
    w=w(2, 1:K(1));
    K=K(1);
else
    
    c = c(2:T, 1:K(T));
    w = w(2:T+1, 1:K(T));
    K= K(1:T);
    
end

if settings.onlychain
    M=[];
    N_new=[];
    N_old=[];
    N=[];
    Z=[];
    Knew=0;
    indchain=0;
    return;
end


%% Sample the network given the C's
Knew = zeros(T, 1); % Number of nodes which only appear at time t (for which c_ti=c_t-1i=0)
N_new = zeros(T, indmax, indmax);
N_old = zeros(T, indmax, indmax);
N = zeros(T, indmax, indmax);
Z = zeros(T, indmax, indmax);
% t=1
w1_all = gamrnd(alpha + sum(c(1,:)), 1/(tau+phi));
wrem(1) = w1_all;
d1_all = poissrnd(w1_all^2);
if d1_all>0
    [partition] = crp2rnd(alpha, 2*d1_all, c(1,:));
    for i=1:2:2*d1_all-1
        N_new(1,partition(i), partition(i+1)) = N_new(1,partition(i), partition(i+1)) + 1;
    end
    Knew(1) = length(unique(partition(partition>K(T)))); % Nodes which are only present at time t
    w(1:end, end+1:end+Knew(1))=NaN;
    % c(1:end, end+1:end+Knew(1))=NaN;
else
    Knew(1) = 0;
end

N(1,:,:) = N_new(1,:,:);
Z(1,:,:) = (squeeze(N(1,:,:))+ squeeze(N(1,:,:))')>0;

% t= 2,...T

c(isnan(c)) = 0;
for t=2:T
    % Sample new interactions
    
    temp=0;
    if t<T
        temp = c(t-1,:)+c(t,:);
        fi = 2*phi;
    else
        temp = c(t-1,:);
        fi=phi;
    end
    wt_all = gamrnd(alpha + sum(temp), 1/(tau+fi));
    wrem(t) = wt_all;
    dt_all = poissrnd(wt_all^2);
    if dt_all>0
        [partition] = crp2rnd(alpha, 2*dt_all, temp );
        partition(partition>K(T)) = partition(partition>K(T)) + sum(Knew(1:t-1));
        Knew(t) = length(unique(partition(partition>K(T))));
        for i=1:2:2*dt_all-1
            N_new(t,partition(i), partition(i+1)) = N_new(t,partition(i), partition(i+1)) + 1;
        end
        w(t:end, end+1:end+Knew(t))=NaN;
        %c(t:end, end+1:end+Knew(t))=NaN;
    else
        Knew(t) = 0;
    end
    % Sample old interactions
    N_old(t,:,:) = binornd(N(t-1,:,:), exp(-rho*dt) );
    
    
    
    % Aggregate old + new
    N(t,:,:) = N_new(t,:,:) + N_old(t,:,:);
    
    % Obtain undirected graph from directed one
    Z(t,:,:) = (squeeze(N(t,:,:)) + squeeze(N(t,:,:))')>0;
    
end
maxclust = K(T) + sum(Knew);% max(find(sum(Z(end,:,:))>0))
N_new = N_new(:,1:maxclust,1:maxclust); %symmetric
N_old = N_old(:,1:maxclust,1:maxclust); %symmetric
N = N(:,1:maxclust,1:maxclust); %symmetric
Z = Z(:,1:maxclust,1:maxclust); %symmetric



 indchainzT = sum(squeeze(sum(Z(:,1:K(T),1:K(T)),1)),1)>0; % indices of nodes active and created during the chain construction
    indlinks = sum(squeeze(sum(Z(:,:,:),1)),1)>0; 
    
    for t=1:T
        M(:, t) = sum(squeeze(N_new(t, :, :)),1)' + sum(squeeze(N_new(t, :, :)), 2) - diag(squeeze(N_new(t, :, :)));
    end
    
    indchainmL = sum(M, 2)>0;
    
    
      for t=1:T
        mm(:, t) = sum(squeeze(N_new(t, 1:K(T), 1:K(T))),1)' + sum(squeeze(N_new(t,1:K(T) , 1:K(T))), 2) - diag(squeeze(N_new(t, 1:K(T), 1:K(T))));
    end
    indchainmT = sum(mm,2)>0;
    
    


end