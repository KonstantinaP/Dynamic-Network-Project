function [pdf logpdf] = igausspdf(x, mu, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [pdf logpdf] = invGausspdf(x, mu, lambda)
% 
% Compute pdf of an inverse Gaussian distribution
% See the book 
% "The inverse gaussian distribution: theory, methodology, and applications" 
% by Raj Chhikara and Leroy Folks, 1989
%
% pdf =  sqrt(lambda / (2 * pi * x^3)) 
%           * exp[-lambda * (x-mu)^2 / (2*mu^2*x)]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by Francois Caron
% INRIA Bordeaux Sud Ouest
% University of Bordeaux, France
% December 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logpdf = .5*log(lambda/(2*pi)) - 3/2*log(x) ...
    - lambda*(x-mu).^2./(2*mu^2.*x);
pdf = exp(logpdf);