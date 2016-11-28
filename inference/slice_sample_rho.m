function [rho] = slice_sample_rho(rho, nold, nnew, rho_a, rho_b, dt)
% Function that implements the slice sampler for parameter rho


[T] = length(nnew);
for t=2:T

    
    n = nnew{t-1} + nold{t-1};
    ll_term = @(q) compute_lpdf(q, nnew{t}, n, nold{t}, rho_a, rho_b, dt);
        f = @(y) ll_term(exp(y)) + rho_a*log(rho_b) - rho_b*exp(y)+ rho_a*y - gammaln(rho_a);


    
    res= slicesample(log(rho), 1,'logpdf', f, 'thin', 1,'burnin', 0);
    rho=exp(res);
end

    function out = compute_lpdf(r,nn, nmt, no, rho_a, rho_b, dt)
        
       out = sum(sum(-r*dt.*no +(nmt - no).*log(1 - exp(-r*dt))));
        
        
        %
        
    end





end
