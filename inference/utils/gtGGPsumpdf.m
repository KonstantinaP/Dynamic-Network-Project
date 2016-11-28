function [pdf, logpdf] = gtGGPsumpdf(x, alpha, sigma, tau, a, b)

%gtGGPsumpdf evaluates the gamma tilted pdf of the total mass of a generalized gamma process
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

if (a==0)
    [pdf, logpdf] = GGPtotalmasspdf(x, alpha, sigma, tau + b);
    return;
end

if sigma<-10^-8  % Compound Poisson case
    % G is distributed from a Poisson mixture of gamma variables
    error('Case sigma<0 not implemented yet')
elseif sigma<1e-8 % gamma process case
     logpdf = loggampdf(x, alpha + a, tau + b);
     pdf = exp(logpdf);
elseif sigma==0.5 % generalized inverse gaussian case
    nu = a - 1/2;
    delta = sqrt(2) * alpha;
    gamma = sqrt(2) * sqrt(tau + b);
    [pdf, logpdf] = gigausspdf(x, nu, delta, gamma);
else
    [~, logpdf1] = GGPsumpdf(x,alpha, sigma, tau +b);
    logtemp = a*log(x) + a*log(b+tau);
    [~, logC] = genfact(a,sigma); % computes the log of the generalized factorial coefficients
    logtemp2 = logC(end, :) + (1:a).*log(alpha/sigma*(b+tau)^sigma);
    maxlogtemp2 = max(logtemp2);
    logtemp2 = log(sum(exp(logtemp2 - maxlogtemp2))) + maxlogtemp2;
    logpdf = logpdf1 + logtemp - logtemp2;
    pdf = exp(logpdf);    
end
end

function out = loggampdf(x, a, b)

    out = (a-1).*log(x) - b.*x +a.*log(b) - gammaln(a);
end


