function [pdf logpdf] = gigausspdf(x, nu, delta, gamma)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [pdf logpdf] = gigausspdf(x, nu, delta, gamma)
% 
% Compute pdf of a generalized inverse Gaussian distribution
% See the article 
% "Non-Gaussian Ornstein-Uhlenbeck-based models and some of their uses in financial economics" 
% by Barndorff-Nielsen and Shephard, JRSSB 2001
%
% pdf =  (gamma/delta)^nu / (2*besselk(nu, delta*gamma) * x^(nu-1)
%           *exp[-1/2* (delta^2/x + gamma^2*x)]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by Francois Caron
% INRIA Bordeaux Sud Ouest
% University of Bordeaux, France
% December 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (delta == 0) % gamma distribution
    alpha = gamma^2/2;
    logpdf = nu*log(alpha) -gammaln(nu) + (nu-1)*log(x) -alpha*x;
elseif (gamma == 0) % inverse gamma distribution
    alpha = -nu;
    beta = delta^2/2;
    logpdf = alpha * log(beta) - gammaln(alpha) -(alpha+1)*log(x) -beta./x;
else
%     [bes, ierr] = besselk(nu, delta.*gamma);
%     if ierr>1
%         warning('Problem in computation of besselk function, ierr=%d', ierr);
%     end
    bes = besselk(nu, delta.*gamma);    
    if bes ==0
        logbes = log(besselk(nu, delta.*gamma, 1)) - delta.*gamma;
    else
        logbes = log(bes);
    end
    logpdf = nu.*(log(gamma)-log(delta)) - log(2) - logbes...
        + (nu-1).*log(x) -.5*(delta.^2./x + gamma.^2.*x);
end
pdf = exp(logpdf);