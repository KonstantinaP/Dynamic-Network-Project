%% Update of the weights 
function [iweights, rate] = sample_weights(iweights, iC, iM, epsilon, alpha, tau, phi, gvec, settings)

[K, T] = size(iweights);
rate = zeros(1, T);

% exclude the rem counts from computaiton. Not needed
% Update weights
for t=1:T
    g=gvec(t);
    counts= iC(1:K-1, :); 

    weights = iweights(1:K-1, t);
    ids = weights~=0;
    w = weights(ids);
    w_rem = iweights(K, t);
    
    counts = counts(ids, :);
    
    
    logw= log(w);
    
%     logweights=logw;
    
    sum_w = sum(w);
    sumall_w = (sum_w + w_rem);


    eps = epsilon(t);
 
    m = iM(ids, t);
    logwprop = logw;
    p = randn(length(w), 1);
    
    grad1 = grad_U(t, m, w, w_rem, counts, tau, phi, g);
    
    pprop = p - eps.* grad1/2;
    for lp=1:settings.leapfrog.L
        %             wprop = exp(log(wprop) + epsilon*pprop);%wprop = exp(log(wprop) + epsilon*pprop./m_leapfrog);
        logwprop = logwprop + eps.*pprop;
        if lp~=settings.leapfrog.L            
            pprop = pprop  - eps.* grad_U(t, m, exp(logwprop), w_rem, counts, tau, phi, g);
        end
    end
    wprop = exp(logwprop);    
    pprop = pprop - (eps/2).*grad_U(t, m, wprop, w_rem, counts, tau, phi, g);
    
    sum_wprop = sum(wprop);
    sumall_wprop = (sum_wprop + w_rem);
    
    % ratio (in log) ******
    % Cases
    %***
    
    if T==1
        temp1 = - g*(sumall_wprop^2) + g*(sumall_w^2) ...
            + sum((m -1).*(logwprop - logw) )...
            - (tau)*(sum_wprop - sum_w);
        
    else
        switch t
            case 1
                temp1 = - g*(sumall_wprop^2) + g*(sumall_w^2) ...
                    + sum((m +counts(:, t) -1).*(logwprop - logw) )...
                    - (tau+ phi)*(sum_wprop - sum_w);
                
            %case T
             %   temp1 = - sumall_wprop^2 + sumall_w^2 ...
              %      + sum((m + counts(:, t-1) -1).*(logwprop - logw) )...
               %     - (tau + phi)*(sum_wprop - sum_w);
            otherwise
                temp1 = - g*(sumall_wprop^2) + g*(sumall_w^2) ...
                    + sum((m +counts(:, t) +counts(:, t-1) -1).*(logwprop - logw) )...
                    - (tau+ 2*phi)*(sum_wprop - sum_w);
                
        end
    end
    
    logaccept = temp1 -.5*sum(pprop.^2-p.^2) -sum(logw) + sum(logwprop);
    
    %%%%*********
% %     
%     if issimple % If simple graph, do not take into account self-connections
%         logaccept = logaccept+ sum(wprop.^2) - sum(w.^2); % TODO: not so sure about the commented out term
%     end
%     

%     if isnan(logaccept)
%         logaccept = -Inf;
%                 logaccept = 0;
% 
%     end
 

%     wprop(w==fix)= 0;
%     w(w==fix)= 0;
%     keyboard
    if log(rand)<logaccept
        
        
        w = wprop;
        logw = logwprop;
        
    end
    
    
    
    rate(t) = min(1, exp(logaccept));
    ids = [ids; logical(0)];
    iweights(ids, t)=w;
    
%     logweights(1:K-1, t)= logw;
end
%%%%%%%%%%%%
%%%%%%%%%%%%%%%




%% Gradient
function [out] = grad_U(t, mm, wvec, w_rem, counts, tau, phi, gv)
%     N = size(M,2);
    
% if N ==1
%     out( :, 1)= -(M(:, 1)) +wvec.*(2*sum(wvec)+2*w_rem + tau);
% else
    switch t
        case 1
            
            out= -(mm + counts( :,  t)) +wvec.*(2*gv*sum(wvec)+2*gv*w_rem +tau + phi);
       % case N
        %    out= -(M( :, N) + counts( :, N-1 )) +wvec.*(2*sum(wvec)+2*w_rem +tau + phi);
        otherwise
            temp =tau+2*gv*sum(wvec)+2*gv*w_rem +2*phi;
            out = - (mm + counts(:, t-1) + counts(:, t))+ wvec.*temp;
    end
    
% end

end

end
