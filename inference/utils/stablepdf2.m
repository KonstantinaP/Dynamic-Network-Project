function [pdf, logpdf] = stablepdf2(logx, alpha)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pdf of the exponentially tilted stable distribution 
% with Laplace transform (in z)
% exp(-z^alpha)
% Can be expressed with Zolotarev's integral representation
% uses the log(x) as entry for numerical stability
%
% References:
% Luc Devroye. Random variate generation for exponentially and polynomially
% tilted stable distributions. ACM Transactions on Modeling and Computer
% Simulation, vol. 19(4), 2009.
% page 18:11
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% François Caron
% INRIA Bordeaux Sud-Ouest
% May 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check parameters
if alpha<=0 || alpha>=1
    error('alpha must be in ]0,1[');
end

pas = 1e-4;
x0 = pas:pas:pi;
[A, logB] = meshgrid(x0, logx);
[~, logzol] = zolotarev(A, alpha);
logtemp = logzol - exp(logzol - (alpha/(1-alpha))*logB);
maxi = max(logtemp, [], 2);
maxirep = repmat(maxi, 1, length(x0));
logintegral = log(pas) +log(sum(exp(logtemp - maxirep ), 2)) + maxi;
logpdf = log(alpha) - log(1-alpha) -1/(1-alpha)*logx - log(pi)...
    + logintegral';

pdf = exp(logpdf);

end

function [out, logout] = zolotarev(u, sigma)
% Zolotarev function, cf (Devroye, 2009)
logout = (1/(1-sigma)) * log((sin(sigma*u)).^sigma .* (sin((1-sigma)*u)).^(1-sigma) ./ sin(u));
out = exp(logout);
% out = ((sin(sigma*u)).^sigma .* (sin((1-sigma)*u)).^(1-sigma) ./ sin(u)).^(1/(1-sigma));

end