%% Update of the weights 
function [weights, rate] = sample_weights(weights, C, M, epsilon, alpha, tau, phi, settings,issimple)

[K, T] = size(weights);
rate = zeros(1, T);


counts= C(1:K-1, :);
% Update weights
for t=1:T
    
    w = weights(1:K-1, t);
    w_rem = weights(K, t);
    logw= log(w);
    logweights=logw;
    
    sum_w = sum(w);
sumall_w = sum_w + w_rem;


    eps = epsilon(t);
 
    m = M(:, t);
    logwprop = logw;
    p = randn(K-1, 1);
    
    grad1 = grad_U(t, M, w, w_rem, counts, tau, phi);
    
    pprop = p - eps.* grad1/2;
    for lp=1:settings.leapfrog.L
        %             wprop = exp(log(wprop) + epsilon*pprop);%wprop = exp(log(wprop) + epsilon*pprop./m_leapfrog);
        logwprop = logwprop + eps.*pprop;
        if lp~=settings.leapfrog.L            
            pprop = pprop  - eps.* grad_U(t, M, exp(logwprop), w_rem, counts, tau, phi);
        end
    end
    wprop = exp(logwprop);    
    pprop = pprop - (eps/2).*grad_U(t, M, wprop, w_rem, counts, tau, phi);
    
    sum_wprop = sum(wprop);
    sumall_wprop = sum_wprop + w_rem;
    
    % ratio (in log) ******
    % Cases
    %***
    
    if T==1
        temp1 = - sumall_wprop^2 + sumall_w^2 ...
            + sum((m -1).*(logwprop - logw) )...
            - (tau)*(sum_wprop - sum_w);
        
    else
        switch t
            case 1
                temp1 = - sumall_wprop^2 + sumall_w^2 ...
                    + sum((m +counts(:, t) -1).*(logwprop - logw) )...
                    - (tau+ phi)*(sum_wprop - sum_w);
                
            case T
                temp1 = - sumall_wprop^2 + sumall_w^2 ...
                    + sum((m + counts(:, t-1) -1).*(logwprop - logw) )...
                    - (tau + phi)*(sum_wprop - sum_w);
            otherwise
                temp1 = - sumall_wprop^2 + sumall_w^2 ...
                    + sum((m +counts(:, t) +counts(:, t-1) -1).*(logwprop - logw) )...
                    - (tau+ 2*phi)*(sum_wprop - sum_w);
                
        end
    end
    
    logaccept = temp1 -.5*sum(pprop.^2-p.^2) -sum(logw) + sum(logwprop);
    
    %%%%*********
%     
%     if issimple % If simple graph, do not take into account self-connections
%         logaccept = logaccept+ sum(wprop.^2) - sum(w.^2); % TODO: not so sure about the commented out term
%     end
    
    if isnan(logaccept)
        logaccept = -Inf;
    end
    
    if log(rand)<logaccept
        
        
        w = wprop;
        logw = logwprop;
        
    end
    
    rate(t) = min(1, exp(logaccept));
    
    weights(1:K-1, t)=w;
    logweights(1:K-1, t)= logw;
end
%%%%%%%%%%%%
%%%%%%%%%%%%%%%




%% Gradient
function [out] = grad_U(t, M, wvec, w_rem, counts, tau, phi)
    N = size(M,2);
if N ==1
    out( :, 1)= -(M(:, 1)) +wvec.*(2*sum(wvec)+2*w_rem + tau);
else
    switch t
        case 1
            
            out= -(M(:, 1) + counts( :,  1)) +wvec.*(2*sum(wvec)+2*w_rem +tau + phi);
        case N
            out= -(M( :, N) + counts( :, N-1 )) +wvec.*(2*sum(wvec)+2*w_rem +tau + phi);
        otherwise
            temp =tau+2*sum(wvec)+2*w_rem +2*phi;
            out = - (M(:, t) + counts(:, t-1) + counts(:, t))+ wvec.*temp;
    end
    
end

end

end
