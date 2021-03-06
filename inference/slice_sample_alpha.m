function [alpha] = slice_sample_alpha(weights, C, alpha, phi, tau, alpha_a, alpha_b)
%sumw = sum(weights,1);
% Function that implements the slice sampler for hyperparameter alpha

ll_term = @(q) compute_lpdf(q,  weights, C, phi, tau);

f = @(y) ll_term(exp(y)) + alpha_a*log(alpha_b) - alpha_b*exp(y)+alpha_a*y -gammaln(alpha_a);


res= slicesample(log(alpha), 1,'logpdf', f, 'thin', 1,'burnin', 0);
alpha=exp(res);

    function out = compute_lpdf(a, weights, C, phi, tau)
%         
%         [K N] = size(weights);
%         cc=[0 sum(C(:, 1:N-1), 1)];
%         fi = ones(1, N)*(phi+tau);
%         fi(1) = tau;% + phi ;
%         out = (a + cc).* log(fi) - (fi).*sum(weights,1)...
%             + (a + cc-1).*log(sum(weights,1))-gammaln(a+ cc);
%         % a
%         out=sum(out);
%         %          keyboard
%         if ~isfinite(out)
%             keyboard
%         end
%         
%         
        
        [K N] = size(weights);
        mask = (weights(1:end-1, 1:N-1)==0 & weights(1:end-1, 2:N)~=0);
        
        wvec = zeros(1, N);
        aex = ones(1, N).*a;        
        wvec(1) = sum(weights(:, 1));
        wvec(2:end) = weights(end, 2:N)  + sum(weights(1:end-1, 2:N).*mask, 1 );
        aex(2:end) = aex(2:end) + C(end, 1:N-1) ;
        fi = ones(1, N)*(phi+tau);
        fi(1) = tau;% + phi ;
        out = (aex ).* log(fi) - (fi).*wvec...
            + (aex - 1).*log(wvec)-gammaln(aex);
        out = sum(out);
        
          if ~isfinite(out)
            keyboard
        end
        
        
%         
        
    end





end
