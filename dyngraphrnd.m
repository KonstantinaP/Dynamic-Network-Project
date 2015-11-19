function [Z, N, N_new, N_old, c, K, Knew, w] = dyngraphrnd(alpha, sigma, tau, T, phi, rho, dt)

if sigma~=0
    error('Not implemented yet!')
end

if nargin<7
    dt = 1; % discretization step
end

indmax = 1000; % so that to allocate memory
c = zeros(T, indmax);
w = zeros(T, indmax);

%% Sample the latent C_1,C_2,...
K = zeros(T, 1); % K(t): \sum_k=1^t c_{ki}>0 for all i<=K(t), 0 otherwise
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

for t=2:T
    % Sample counts for existing atoms
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
c = c(:, 1:K(T));


%% Sample the network given the C's
Knew = zeros(T, 1); % Number of nodes which only appear at time t (for which c_ti=c_t-1i=0)
N_new = zeros(T, indmax, indmax);
N_old = zeros(T, indmax, indmax);
N = zeros(T, indmax, indmax);
Z = zeros(T, indmax, indmax);
% t=1
w1_all = gamrnd(alpha + sum(c(1,:)), 1/(tau+phi));
d1_all = poissrnd(w1_all^2);
if d1_all>0
    [partition] = crp2rnd(alpha, 2*d1_all, c(1,:));
    for i=1:2:2*d1_all-1
        N_new(1,partition(i), partition(i+1)) = N_new(1,partition(i), partition(i+1)) + 1;
    end
    Knew(1) = length(unique(partition(partition>K(T)))); % Nodes which are only present at time t
else
    Knew(1) = 0;
end
N(1,:,:) = N_new(1,:,:);
Z(1,:,:) = (squeeze(N(1,:,:))+ squeeze(N(1,:,:))')>0;  

% t= 2,...T
for t=2:T
    % Sample new interactions
    wt_all = gamrnd(alpha + sum(c(t-1,:)+c(t,:)), 1/(tau+2*phi));
    dt_all = poissrnd(wt_all^2);
    if dt_all>0
        [partition] = crp2rnd(alpha, 2*dt_all, c(t-1,:) + c(t,:) );
        partition(partition>K(T)) = partition(partition>K(T)) + sum(Knew(1:t-1));
        Knew(t) = length(unique(partition(partition>K(T))));
        for i=1:2:2*dt_all-1
            N_new(t,partition(i), partition(i+1)) = N_new(t,partition(i), partition(i+1)) + 1;
        end
    else
        Knew(t) = 0;
    end
    % Sample old interactions
    N_old(t,:,:) = binornd(N(t-1,:,:), exp(-rho*dt) );
    % Aggregate old + new
    N(t,:,:) = N_new(t,:,:) + N_old(t,:,:);
    % Obtain undirected graph from directed one
    Z(t,:,:) = (squeeze(N(t,:,:))+ squeeze(N(t,:,:))')>0;        
end
maxclust = K(T)+sum(Knew);% max(find(sum(Z(end,:,:))>0))
N_new = N_new(:,1:maxclust,1:maxclust);
N_old = N_old(:,1:maxclust,1:maxclust);
N = N(:,1:maxclust,1:maxclust);
Z = Z(:,1:maxclust,1:maxclust);
end


