function X = gigaussrnd(nu, delta, gamma, M, N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = gigaussrnd(nu, delta, gamma, M, N)
% 
% Sample from a generalized inverse Gaussian distribution
% See the article 
% "Non-Gaussian Ornstein-Uhlenbeck-based models and some of their uses in financial economics" 
% by Barndorff-Nielsen and Shephard, JRSSB 2001
%
% pdf =  (gamma/delta)^nu / (2*besselk(nu, delta*gamma) * x^(nu-1)
%           *exp[-1/2* (delta^2/x + gamma^2*x)]
%
% Requires the randraw.m function by Alex Bar Guy  &  Alexander Podgaetsky
% and the Statistics toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by Francois Caron
% INRIA Bordeaux Sud Ouest
% University of Bordeaux, France
% December 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
    [M, N] = size(nu);
end

if (delta == 0) % gamma distribution
    X = gamrnd(nu, 2./gamma.^2, M, N);
elseif (gamma == 0) % inverse gamma distribution
    alpha = -nu;
    beta = delta.^2/2;
    X = 1./gamrnd(alpha, 1./beta, M, N);
elseif (nu == -1/2) % inverse Gaussian distribution
    lambda = delta.^2;
    mu = delta./gamma;
    X = igaussrnd(mu, lambda, M, N);
elseif (nu==1/2) % (inverse) inverse Gaussian distribution
    lambda = gamma.^2;
    mu = gamma./delta;
    X = 1./igaussrnd(mu, lambda, M, N);
else
    if nargin<4
        for i=1:M
            for j=1:N
                X(i,j) = randraw('gig', [nu(i,j), delta(i,j)^2, gamma(i,j)^2]);
            end
        end
    else
        X = randraw('gig', [nu, delta^2, gamma^2], M, N);
    end
end