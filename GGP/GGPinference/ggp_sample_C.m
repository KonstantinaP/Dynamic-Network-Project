function C = ggp_sample_C(C, weights, phi, alpha, sigma, tau)


% Function that updats the C_{tk} for k=1, ..., K (exclude c_rem)  and t=1, ..., N.
% The last row of C, i.e. C(end,:) is $c_{t\ast}$ for t=1, ..., T.
% Last column will be for last time point N. No evidence for that. So just
% sample from prior.

% *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***

[K, T] = size(weights);
%K = K-1;
nb_MH = 1; % Nb of MH iterations


c = C(1:K-1, 1:end-1);
w = weights(1:K-1, :);
% for nn=1:nb_MH
%     maxo = max(c-sigma, 0);
%    Cnew = tpoissrnd(phi.*(w(:, 1:T-1))); % proposal from zero-truncarted Poisson distribution
%        maxnew = max(Cnew-sigma, 0);
% 
%     u = rand(K-1, T-1);
%     logaccept = (-maxo+maxnew).* log((tau+phi).*w(:, 2:T))...
%         + gammaln(maxo)  - gammaln(maxnew);
% 
%     accept = exp(logaccept);
%     
%     c(u<accept & logical(w(c 0:, 2:T)>0) ) = Cnew(u<accept & logical(w(:, 2:T)>0));
% end



for nn=1:nb_MH
    maxo = max(c-sigma, 0);
   Cnew = poissrnd(phi.*(w(:, 1:T-1))); % proposal from zero-truncarted Poisson distribution
       maxnew = max(Cnew-sigma, 0);

    u = rand(K-1, T-1);
    logaccept = (maxnew - maxo).* log(tau+phi) + (Cnew - c).*log(w(:, 2:T))...
        + gammaln(max(c-sigma, 1-sigma))  - gammaln(max(Cnew-sigma, 1-sigma));

    accept = exp(logaccept);
    
    c(u<accept & logical(w(:, 2:T)>0) ) = Cnew(u<accept & logical(w(:, 2:T)>0));
    
%  keyboard
end


% % *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***
%term =phi.*w(:, 1:T-1).*(phi+tau);
%p1 = -log(1 + term);
%p2 = log(term) -log(1 + term);

%d(:, :, 1) = exp(p1);
%d(:, :, 2) = exp(p2);

c(w(:, 2:T)==0) = 0;
%c(w(:, 2:T)==0 & rand(size(w(:, 2:T)))> squeeze(d(:, :,1) )) = 1;


C(1:K-1, 1:T-1) = c;
C(1:K-1, end) = poissrnd(phi.*(weights(1:K-1, T)));

