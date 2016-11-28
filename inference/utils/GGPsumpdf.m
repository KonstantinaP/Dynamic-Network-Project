function [pdf, logpdf] = GGPsumpdf(x0,alpha, sigma, tau)

%GGPsumpdf evaluates the pdf of the total mass of a generalized gamma process
% [pdf, logpdf] = gtGGPsumpdf(x, alpha, sigma, tau, a, b)
%
%   It returns the pdf p(x) such that 
%       p(x) \propto x^a * exp(-b*x) * g_{alpha,sigma,tau}(x)
%   where g_{alpha,sigma,tau}(x) is the pdf of the random variable S 
%   with Laplace transform
%   E[e^{-(t*S)}] = exp(-alpha/sigma * [(t+tau)^sigma - tau^sigma])
% -------------------------------------------------------------------------
% INPUTS
%   - x: vector of length n
%   - alpha: strictly positive scalar
%   - sigma: real in (-Inf, 1)
%   - tau: positive scalar; strictly positive if sigma<=0
%   - a: non-negative scalar
%   - b: non-negative scalar
% 
% OUTPUTS
%   - pdf: vector of length n
%   - logpdf: vector of length n
% -------------------------------------------------------------------------
% EXAMPLE
% x = 1:10; alpha = 1; sigma = 0.5; tau = 1; a = 10; b = 1;
% [pdf, logpdf] = gtGGPsumpdf(x, alpha, sigma, tau, a, b);
% -------------------------------------------------------------------------
% See also GTGGPSUMRND, GGPSUMRND, GGPSUMPDF

% Copyright (C) Francois Caron, University of Oxford
% caron@stats.ox.ac.uk
% February 2016
%--------------------------------------------------------------------------

if sigma<-10^-8  % Compound Poisson case
    % G is distributed from a Poisson mixture of gamma variables
    error('Case sigma<0 not implemented yet')
elseif sigma<1e-8
     logpdf = loggampdf(x0, alpha, tau);
     pdf = exp(logpdf);
elseif sigma==0.5
    mu = alpha/sqrt(tau);
    lambda = 2*alpha^2;
    [pdf, logpdf] = igausspdf(x0, mu, lambda);
else % exponentially tilted stable distribution
    [~, logstable] =  stablepdf2((-1/sigma)*log(alpha/sigma)+log(x0), sigma) ; 
    logpdf = -1/sigma * log(alpha/sigma) ... 
        + alpha/sigma *tau^sigma - tau*x0...
        + logstable;
    pdf = exp(logpdf);
end

end

function out = loggampdf(x, a, b)
    out = (a-1).*log(x) - b.*x +a.*log(b) - gammaln(a);
end