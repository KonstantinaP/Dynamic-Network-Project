function X = igaussrnd(mu, lambda, M, N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = igaussrnd(mu, lambda, M, N)
% Sample data from an inverse Gaussian distribution of pdf
% pdf =  sqrt(lambda / (2 * pi * x^3)) 
%           * exp[-lambda * (x-mu)^2 / (2*mu^2*x)]
%
% See the book 
% "The inverse gaussian distribution: theory, methodology, and applications" 
% by Raj Chhikara and Leroy Folks, 1989
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by Francois Caron
% INRIA Bordeaux Sud Ouest
% University of Bordeaux, France
% December 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<3
    [M,N] = size(mu);
    if isequal(size(lambda),[1,1])
        lambda = repmat(lambda, M, N);
    elseif isequal(size(lambda),[M,N])==0
        error('mu and lambda should be arrays of the same size')
    end
end

% Sample chi2(1)
Y = chi2rnd(1, M, N);

if isequal([M,N],size(mu))
    X1 = mu./(2*lambda).*(2*lambda + mu.*Y -sqrt(4*lambda.*mu.*Y + mu.^2.*Y.^2));
    X2 = mu.^2 ./ X1;
else
    X1 = mu/(2*lambda)*(2*lambda + mu*Y -sqrt(4*lambda*mu*Y + mu^2*Y.^2));
    X2 = mu^2 ./ X1;
end

U = rand(M,N);
P = mu./(mu + X1);
C = (U < P);
X = C .* X1 + (ones(M, N) - C) .* X2;

