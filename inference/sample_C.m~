function C = sample_C(C, weights, phi, alpha, tau)

% Function that updats the C_{tk} for k=1, ..., K  and c_{t\ast}and t=1, ..., N.

% The last row of C, i.e. C(end,:) is $c_{t\ast}$ for t=1, ..., N.

% *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***

[K, N] = size(weights);

a = zeros(K, N-1);
a(end, :) = alpha; % used for sampling c_{t\ast}

nb_MH = 1; % Nb of MH iterations
for nn=1:nb_MH
    Cnew = tpoissrnd(phi.*(weights(:, 1:N-1)));
    
    u = rand(K, N-1);
    logaccept = (-C+Cnew).* log((tau+phi).*weights(:, 2:N))...
        + gammaln(a+C)  - gammaln(a+Cnew);
        
    accept = exp(logaccept);
    C(u<accept) = Cnew(u<accept);
    
%     if sum(sum(isnan(C)))
%         keyboard
%     end
    
%         Cnew = max(0, C + 2*(rand>.5) -1);% gamrnd(phi, 1./(weights(:, 1:N-1)));
%     u = rand(K, N-1);
%     logaccept = (-C+Cnew).* log((b+phi).*phi.*weights(:, 2:N).*weights(:, 1:N-1))...
%         + gammaln(a+C) + gammaln(C+1) - gammaln(a+Cnew) - gammaln(Cnew+1);
%         
%     accept = exp(logaccept);
%     V(u<accept) = Cnew(u<accept);
end