function [tau] = slice_sample_tau(weights, counts, alpha, phi, tau, tau_a, tau_b)
%sumw = sum(weights,1);
% Function that implements the slice sampler for hyperparameter alpha

ll_term = @(q) compute_lpdf(q,  weights, counts, phi, alpha);

f = @(y) ll_term(exp(y)) + tau_a*log(tau_b) - tau_b*exp(y)+tau_a*y -gammaln(tau_a);


res= slicesample(log(tau), 1,'logpdf', f, 'thin', 1,'burnin', 0);
tau=exp(res);

    function out = compute_lpdf(tt, w, c, phi, alpha)
        
        
        
        
        
        [K N] = size(w);
        
        wfall = sum(w(:,1),1);
        aterm = alpha*log(tt) - tt*wfall + (alpha-1)*log(wfall) -gammaln(alpha);
        
        mask = c(1:K-1, 1:N-1)==0;
        
        w(end, 2:N) = w(end, 2:N) + sum(w(1:K-1, 2:N).*mask,1);
        
        
        a_ext = zeros(K, N-1);
        a_ext(end, :) = alpha; % used for sampling c_{t\ast}
        
        cc = c(:, 1:N-1);
        
        out = (a_ext + cc).*log(tt+phi) -(tt+phi).*w(:, 2:N) ...
            + (a_ext + cc-1).*log(w(:, 2:N)) - gammaln(a_ext + cc);
        
        mask = [mask ; logical(false(1, N-1))];
        out(mask)=0;
        
        out=sum(sum(out)) + aterm;
        
        %          keyboard
        if ~isfinite(out)
            keyboard
        end
        
        
        
        
        
    end





end
