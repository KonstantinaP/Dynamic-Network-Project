function [lambda, V] = sample_weights_add(lambda, V, sr2, ak, rho, b)

% sample lambda and V at indices where there are no observations
% required to have a correct MCMC sampler

[K, N] = size(sr2);

rho = max(unique(rho)); % only works if rho is the same for all !!!
x = zeros(N, 1);
x(N) = sr2(end, end);
% pause
for t=N-1:-1:1
    x(t) = sr2(end, t) + rho*x(t+1)/(1+rho+x(t+1));
end

for t=1:N-1
    ind = find(sum(ak(1:K-1, t+1:end), 2)==0);
    V(ind, t) = poissrnd((1+rho)/(1+rho+x(t))*rho*lambda(ind,t));
    lambda(ind, t+1) = gamrnd(V(ind,t), 1/(b+rho+x(t+1)));
end
