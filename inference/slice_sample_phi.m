function [phi] = slice_sample_phi(phi, phi_a, phi_b, weights, alpha, tau)

% Function that implements the sliece sampler for hyperparameter phi
% We assume gamma prior over phi

ll_term = @(q) compute_lpdf(q,  weights, alpha, tau);


f = @(y) ll_term(exp(y)) + phi_a*log(phi_b) - phi_b*exp(y)+phi_a*y -gammaln(phi_a);


res= slicesample(log(phi), 1,'logpdf', f, 'thin', 1,'burnin', 0);

phi=exp(res);



    function res = compute_lpdf(q, weights, alpha, tau)
        res=0;
        ph=q;
        [K, N] = size(weights);
        a = zeros(K, N-1);
        a(end, :) = alpha;
        
        temp = 2*sqrt((tau+ph).*ph.*weights(:, 1:N-1).*weights(:, 2:N));
        
        res= log(besseli(a-1,temp,1)) - ph.*(weights(:, 1:N-1)+weights(:, 2:N))...
            + ((a+1)./2)*log(tau+ph)  -(a-1)/2 .*log(ph) + temp;
        
        
        res(~isfinite(res)) = 0;
        
        mask0 = (weights(:, 2:N)==0);
        res(mask0) = -ph .* weights(mask0);
        
        
        res=sum(sum(res));
        
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
