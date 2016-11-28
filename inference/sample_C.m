function C = sample_C(C, weights, phi, alpha, tau)


% Function that updats the C_{tk} for k=1, ..., K  and c_{t\ast}and t=1, ..., N.
% The last row of C, i.e. C(end,:) is $c_{t\ast}$ for t=1, ..., N.
% Last column will be for last time point N. No evidence for that. So just
% sample from prior.

% *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***

[K, N] = size(weights);

a = zeros(K, N-1);
a(end, :) = alpha; % used for sampling c_{t\ast}

nb_MH = 1; % Nb of MH iterations

c = C(:, 1:end-1);
%cc= C(:, 2:end);
for nn=1:nb_MH
   Cnew = tpoissrnd(phi.*(weights(:, 1:N-1))); % proposal from zero-truncarted Poisson distribution
%     Cnew = poissrnd(phi.*(weights(:, 1:N-1))); % proposal from  Poisson distribution
    u = rand(K, N-1);
    logaccept = (-c+Cnew).* log((tau+phi).*weights(:, 2:N))...
        + gammaln(a+c)  - gammaln(a+Cnew);

    accept = exp(logaccept);
    
    c(u<accept & logical(weights(:, 2:N)>0) ) = Cnew(u<accept & logical(weights(:, 2:N)>0));
% c(u<accept  ) = Cnew(u<accept);
end

% % *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***
term =phi.*weights(:, 1:N-1).*(phi+tau);
p1 = -log(1 + term);
p2 = log(term) -log(1 + term);

d(:, :, 1) = exp(p1);
d(:, :, 2) = exp(p2);

c(weights(:, 2:N)==0) = 0;
c(weights(:, 2:N)==0 & rand(size(weights(:, 2:N)))> squeeze(d(:, :,1) )) = 1;


C(:, 1:end-1) = c;
C(:, end) = poissrnd(phi.*(weights(:, N)));

