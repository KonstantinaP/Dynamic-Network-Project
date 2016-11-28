function [G, status] = gtGGPsumrnd(alpha, sigma, tau, a, b)

%gtGGPsumrnd samples from the distribution of the total mass of a generalized gamma process
% [G, status] = gtGGPsumrnd(alpha, sigma, tau, a, b)
%
%   It generates a realization of the random variable G with pdf 
%       p(x) \propto x^a * exp(-b*x) * g_{alpha,sigma,tau}(x)
%   where g_{alpha,sigma,tau}(x) is the pdf of the random variable S 
%   with Laplace transform
%   E[e^{-(t*S)}] = exp(-alpha/sigma * [(t+tau)^sigma - tau^sigma])
% -------------------------------------------------------------------------
% INPUTS
%   - alpha: strictly positive scalar
%   - sigma: real in (-Inf, 1)
%   - tau: positive scalar; strictly positive if sigma<=0
%   - a: non-negative scalar
%   - b: non-negative scalar
% 
% OUTPUTS
%   - G: positive scalar
% -------------------------------------------------------------------------
% EXAMPLE
% alpha = 100; sigma = 0.5; tau = 1; a = 10; b = 1;
% S = gtGGPsumrnd(alpha, sigma, tau, a, b);
% -------------------------------------------------------------------------
% See also GTGGPSUMPDF, GGPSUMRND

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% February 2016
%--------------------------------------------------------------------------

% Conditions on the parameters
if sigma>=1 
    error('sigma must be in (-\infty,1)');
elseif tau<0
    error('tau must be non-negative');
elseif tau==0 && sigma~=0
    error('tau must be strictly positive if sigma=0');
elseif alpha<=0
    error('alpha must be strictly positive');
elseif a<0
    error('a must be non-negative');
elseif b<0
    error('b must be non-negative');
end

status = 1;

if (a==0)
    G = GGPsumrnd(alpha, sigma, tau + b);
    return;
end

if sigma<-10^-8  % Compound Poisson case
    % G is distributed from a Poisson mixture of gamma variables
    error('Case sigma<0 not implemented yet')
elseif sigma<1e-8 % Gamma process case
    G = gamrnd(alpha + a, 1/(tau + b));    
elseif sigma==0.5 % Inverse Gaussian case
    nu = a - 1/2;
    delta = sqrt(2) * alpha;
    gamma = sqrt(2) * sqrt(tau + b);
    G = gigaussrnd(nu, delta, gamma);
else 
    % Sample using the trick of Stefano
    J = sample_nbclust(alpha, sigma, tau, a, b);
    W = GGPsumrnd(alpha, sigma, tau + b);
    V = gamrnd(a - J*sigma, 1/(b+tau));
    G = W + V;
end

end


function [x, proba] = sample_nbclust(alpha, sigma, tau, n, b)

[~, logC] = genfact(n,sigma); % computes the log of the generalized factorial coefficients

logtemp = logC(end, :) + (1:n).*log(alpha/sigma*(b+tau)^sigma);
temp = exp(logtemp - max(logtemp));
x = find(sum(temp)*rand<cumsum(temp), 1); % sample from the discrete distribution
proba = temp/sum(temp);
end

