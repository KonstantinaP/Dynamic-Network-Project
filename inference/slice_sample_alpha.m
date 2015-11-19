function [alpha] = slice_sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b)

% Function that implements the sliece sampler for hyperparameter alpha


ll_term = @(q) compute_lpdf(q,  weights, C, phi, tau);


f = @(y) ll_term(exp(y)) + alpha_a*log(alpha_b) - alpha_b*exp(y)+alpha_a*y -gammaln(alpha_a);


res= slicesample(log(alpha), 1,'logpdf', f, 'thin', 1,'burnin', 0);

alpha=exp(res);



    function out = compute_lpdf(q, weights, C,  phi, tau)
      a=q;
   
       
    [K N] = size(weights);
   
     out = (a + C(end,1:N-1)).* log(tau+phi) - (phi+tau).*weights(end, 2:N)...
         + (a + C(end, 1:N-1)-1).*log(weights(end, 2:N))-gammaln(a+C(end, 1:N-1));
     if ~isfinite(out)
         keyboard
     end
     out = sum(out) + a*log(tau) -weights(end,1)*tau+(a-1)*log(weights(end,1)) -gammaln(a);

        
  

 
        
    end


    


end
