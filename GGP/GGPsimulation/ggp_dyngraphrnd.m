function [Z, w, c, K, N_new, N_old, N, M, indlinks, indcnz] = ggp_dyngraphrnd(alpha, sigma, tau, T, phi, rho, gvar, settings)


dt = settings.dt;
% Function that simulates from the process.
% output Z: symmetric
%        N's: all directed counts, non-symmetric

threshold =  settings.threshold;



binall = [];

% indmax = 1900; % so that to allocate memory
indmax = 10000; % so that to allocate memory
c = zeros(T, indmax);
w = zeros(T, indmax);

% Sample the graph conditional on the weights w

[wvec ] = GGPrnd(alpha, sigma, tau,  threshold);
w(1, 1:length(wvec)) = wvec;



c(1, 1:length(wvec)) = poissrnd(phi*wvec);
K(1) = length(wvec);
for t=2:T
    t
    %% Sample counts for existing atoms
    ind = c(t-1,:)>0;
    w(t, ind) = gamrnd(max(c(t-1,ind)-sigma, 0 ), 1/(tau+phi));
    c(t,ind) = poissrnd(phi*w(t, ind));
    
    % Sample new atoms
    wnew = GGPrnd(alpha, sigma, tau+phi, threshold);
    
    cnew = poissrnd(phi.*wnew);
    knew = length(wnew);
    w(t, K(t-1)+1:K(t-1)+ knew)=wnew;
    c(t, K(t-1)+1:K(t-1)+ knew)=cnew;
    K(t) = K(t-1)+ knew;
end


w=w(:, 1:K(T));
c=c(:, 1:K(T));


% % Sample the network given the C's
if ~settings.gcontrol
    gvar = ones(1, T);
else
    %         gvar = gamrnd(settinga.g_a, 1/settings.g_b, 1, T);
    gvar=gvar;
end

N_new = zeros(T, K(T), K(T));
N_old = zeros(T, K(T), K(T));
N = zeros(T, K(T), K(T));
Z = zeros(T, K(T), K(T));
% t=1
cumsum_w = [0, cumsum(w(1,:))];
W_star = cumsum_w(end);  % Total mass of the GGP
D_star = poissrnd(gvar(1)*(W_star)^2); % Total number of directed edges




temp = W_star * rand(D_star, 2);
[~, bin] = histc(temp, cumsum_w);
for d=1:D_star
    N_new(1,bin(d,1), bin(d,2)) = N_new(1,bin(d,1), bin(d,2))+ 1;
end
binall = [binall bin(:)'];
N(1,:,:) = N_new(1,:,:);
Z(1,:,:) = (squeeze(N(1,:,:))+ squeeze(N(1,:,:))')>0;




for t=2:T
    t
    cumsum_w = [0, cumsum(w(t,:))];
    W_star = cumsum_w(end);  % Total mass of the GGP
    D_star = poissrnd(gvar(t)*(W_star^2)); % Total number of directed edges
    
    temp = W_star * rand(D_star, 2);
    [~, bin] = histc(temp, cumsum_w);
    
    binall = [binall bin(:)'];
    for d=1:D_star
        N_new(t,bin(d,1), bin(d,2)) = N_new(t,bin(d,1), bin(d,2))+ 1;
    end
    % Sample old interactions
    N_old(t,:,:) = binornd(N(t-1,:,:), exp(-rho*dt) );
    
    % Aggregate old + new
    N(t,:,:) = N_new(t,:,:) + N_old(t,:,:);
    
    % Obtain undirected graph from directed one
    Z(t,:,:) = (squeeze(N(t,:,:)) + squeeze(N(t,:,:))')>0;
    
%     fprintf('network for location t=%d created \n', t )
    
    
end

%     maxclust = K(T);% max(find(sum(Z(end,:,:))>0))
%     N_new = N_new(:,1:maxclust,1:maxclust); % directed counts, non-symmetric
%     N_old = N_old(:,1:maxclust,1:maxclust); % directed counts, non-symmetric
%     N = N(:,1:maxclust,1:maxclust); % directed counts, non-symmetric
%     Z = Z(:,1:maxclust,1:maxclust); % symmetric
%


for t=1:T
    M(:, t) = sum(squeeze(N_new(t, :, :)),1)' + sum(squeeze(N_new(t, :, :)), 2) - diag(squeeze(N_new(t, :, :)));
end
%     keyboard
indlinks = sum(M, 2)>0; % indices of the nodes that participate in links


[Nall T]=size(M);
indcnz = (sum(M,2)==0 & sum(c(1:end-1,:),1)'>0);

indcnz = zeros(Nall, T-1); % Nall x T-1

for t=1:T-1
    indcnz(:, t) = c(t,:)'>0 & (sum(M,2)==0);
end

 indcnz = [ zeros(Nall,1) indcnz];% Nall x T
 

end





