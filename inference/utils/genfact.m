function [C, logC] = genfact(n , s)

%genfact computes the generalized factorial coefficients
% [C, logC] = genfact(n , s)
%
%   It computes the generalized factorial coefficients C(n,k;s), such that 
%   (st)_n = \sum_{k=0}^n C(n,k;s) (t)_k
%   where (t)_n = t * (t+1) * ... * (t + n-1)_n is the ascending factorial
%
%   C(n,k; s) verifies C(n,0;s)=0, C(1,1;s)=s and
%   C(n,k;s) = s * C(n-1,k-1;s) + (n-1-ks) * C(n-1, k;s)
% -------------------------------------------------------------------------
% INPUTS
%   - n: positive integer
%   - s: scalar in (0,1)
% 
% OUTPUTS
%   - C: n*n matrix of generalized factorial coefficients C(n,k;s)
%   - logC: log of the generalized factorial coefficients
% -------------------------------------------------------------------------
% EXAMPLE
% [C,logC] = genfact(10,0.1)
% -------------------------------------------------------------------------

% REFERENCE
% S. Favaro, B. Nipoti, YW Teh. Random variate generaion for Laguerre-type
% exponentially tilted alpha-stable distributions. Electronic Journal of
% Statistics, vol. 9(2015).
%
% Copyright (C) François Caron, University of Oxford
% caron@stats.ox.ac.uk
% Modified from Stefano Favaro, University of Torino
% February 2016
%--------------------------------------------------------------------------

if n<1
    error('n must be a positive integer');
end
if (s<0 || s>1)
    error('s must be in (0,1)');
end

logC = -inf*ones(n, n);
logC(1,1)=log(s);
for i=2:n    
    logC(i,1)=log(i - 1 - s) + logC(i-1,1);    
    for j=2:i       
        if i==j
            logC(i,j)=log(s) + logC(i-1,j-1);     
        else
            a = log(s) + logC(i-1,j-1);
            b = log(i - 1 - j*s) + logC(i-1,j);
            logC(i,j) = max(a,b) + log(1 + exp(min(a,b)-max(a,b)));
        end
    end    
end
C = exp(logC);