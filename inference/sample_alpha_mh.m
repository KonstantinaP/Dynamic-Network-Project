function [alpha, acct ] = sample_alpha_mh(weights, C, alpha, phi, tau, alpha_a, alpha_b)
%
%
% [K N] = size(weights);
%
% cc=[0 sum(C(:, 1:N-1), 1)];%+ sum(C,1);
% fi = ones(1, N)*(phi+tau);
% fi(1) = tau  ;
% lambda = .5;
% alphaprop = alpha.*exp(lambda*randn); % log normal distribution
% logaccept = sum((alphaprop - alpha)*log(fi) + (alphaprop - alpha)*log(sum(weights,1)) + gammaln(alpha + cc) - gammaln(alphaprop + cc));
% logaccept = logaccept  - alpha_b*(alphaprop - alpha) + (alpha_a )*(log(alphaprop) - log(alpha)) ;

[K N] = size(weights);
mask = (weights(1:end-1, 1:N-1)==0 & weights(1:end-1, 2:N)~=0);

lambda = .5;
alphaprop = alpha.*exp(lambda*randn); % log normal distribution

wvec = zeros(1, N);
aex = ones(1, N).*alpha;
aexprop = ones(1, N).*alphaprop;

wvec(1) = sum(weights(:, 1));
wvec(2:end) = weights(end, 2:N)  + sum(weights(1:end-1, 2:N).*mask, 1 );

aex(2:end) = aex(2:end) + C(end, 1:N-1) ;
aexprop(2:end) = aexprop(2:end) + C(end, 1:N-1) ;

fi = ones(1, N)*(phi+tau);
fi(1) = tau;
logaccept = sum((alphaprop - alpha)*log(fi) + (alphaprop - alpha)*log(wvec) + gammaln(aex) - gammaln(aexprop));
logaccept = logaccept  - alpha_b*(alphaprop - alpha) + (alpha_a )*(log(alphaprop) - log(alpha)) ;





acct = 0;


if rand<exp(logaccept)
    acct=1;
    alpha=alphaprop;
end


end