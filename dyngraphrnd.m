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

w1_all = gamrnd(alpha, 1/tau);
c1_all = poissrnd(phi*w1_all);
if c1_all>0
    [~, m, K(1)] = crprnd(alpha, c1_all);
    for i=1:K(1)
        c(1, i) = m(i);
        theta(i) = alpha*rand;
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
            theta(K(t-1)+i) = alpha*rand;
        end
        K(t) = K(t-1) + K_new;   
    else
        K(t) = K(t-1);
    end
end
c = c(:, 1:K(T));


%% Sample the network given the C's
N_new = zeros(T, indmax, indmax);
N_old = zeros(T, indmax, indmax);
N = zeros(T, indmax, indmax);
Z = zeros(T, indmax, indmax);
% t=1
w1_all = gamrnd(alpha + sum(c(1,:)), 1/(tau+phi));
d1_all = poissrnd(w1_all^2);
if d1_all>0
    [partition] = crp2rnd(alpha, d1_all, c(1,:));
    for i=1:d1_all-1
        N_new(1,partition(i), partition(i+1)) = N_new(1,partition(i), partition(i+1)) + 1;
    end
    Knew(1) = sum(partition>K(T)); % Nodes which are only present at time t
else
    Knew(1) = 0;
end

N(1,:,:) = N_new(1,:,:);
Z(1,:,:) = (squeeze(N(1,:,:))+ squeeze(N(1,:,:))')>0;  

% t= 2,...T
for t=2:T
    w_all = gamrnd(alpha + sum(c(t-1,:)+c(t,:)), 1/(tau+2*phi));
    d_all = poissrnd(w_all^2);
    if d_all>0
        [partition] = crp2rnd(alpha, d_all, c(t-1,:) + c(t,:) );
        partition(partition>K(T)) = partition(partition>K(T)) + sum(Knew(1:t-1));
        Knew(t) = length(unique(partition(partition>K(T))));
        for i=1:d_all-1
            N_new(t,partition(i), partition(i+1)) = N_new(t,partition(i), partition(i+1)) + 1;
        end
    else
        Knew(t) = 0;
    end
    N_old(t,:,:) = binornd(N(t-1,:,:), exp(-rho*dt) );
    N(t,:,:) = N_new(t,:,:) + N_old(t,:,:);
    Z(t,:,:) = (squeeze(N(t,:,:))+ squeeze(N(t,:,:))')>0;    
end
maxclust = max(find(sum(Z(end,:,:))>0));
N_new = N_new(:,1:maxclust,1:maxclust);
N_old = N_old(:,1:maxclust,1:maxclust);
N = N(:,1:maxclust,1:maxclust);
Z = Z(:,1:maxclust,1:maxclust);
end


