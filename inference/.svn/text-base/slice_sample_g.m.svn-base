function [gvar] = slice_sample_g(weights, gvar, nnew, g_a, g_b)
%sumw = sum(weights,1);
% Function that implements the slice sampler for parameter gvar


[K T] = size(weights);
for t=1:T

    ll_term = @(q) compute_lpdf(q,  weights(1:K-1,t), nnew{t});
        f = @(y) ll_term(exp(y)) + g_a*log(g_b) - g_b*exp(y)+ g_a*y - gammaln(g_a);


    g=gvar(t);
    res= slicesample(log(g), 1,'logpdf', f, 'thin', 1,'burnin', 0);
    gvar(t)=exp(res);
end

    function out = compute_lpdf(g, wts, n)
        [id1 id2] = find(triu(ones(size(n)),1));
        idx = sub2ind(size(n), id1, id2);
        logw=log(wts);
        temp = log(2.*g)+logw(id1)+logw(id2);
        temp(id1==id2) = log(g)+2*logw(id1(id1==id2));
temp(temp==-inf)=0;
        out = sum(n(idx).*temp - exp(temp));
        
        
        %
        
    end





end
