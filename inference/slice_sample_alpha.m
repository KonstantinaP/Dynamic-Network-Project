function [alpha] = slice_sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b)

% Function that implements the sliece sampler for hyperparameter alpha


ll_term = @(q) compute_lpdf(q,  weights, C, alpha, phi, tau);


f = @(y) ll_term(exp(y)) + alpha_a*log(alpha_b) - alpha_b*exp(y)+alpha_a*y -gammaln(alpha_a);


res= slicesample(log(alpha), 1,'logpdf', f, 'thin', 1,'burnin', 0);

alpha=exp(res);



    function res = compute_lpdf(q, weights, C, alpha, phi, tau)
      a=q;

       
    [K N] = size(weights);
     res = (a + C(end,1:N-1)).* log(tau+phi) - (phi+tau).*weights(end, 2:N)...
         + (a + C(end, 1:N-1)-1).*log(weights(end, 2:N))-gammaln(alpha+C(end, 1:N-1));
     
     res = sum(res) + a*log(tau) -weights(end,1)*tau+(a-1)*log(weights(end,1)) -gammaln(a)
    
        

  

        
%       if ~isfinite(res) || isnan(res)
%           
%                 %disp('Warning, res is -inf or NaN in sampling alpha'); 
%                 
%                 res=minres-1e8;
%                 
%             end
%             minres=min(res,minres);
        % assert(isfinite(res));
        
        
    end


    


end
