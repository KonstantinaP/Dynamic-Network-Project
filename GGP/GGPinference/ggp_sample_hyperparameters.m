%% Update of the hyperparameters
function [w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, rate2] = ggp_sample_hyperparameters(w, C, w_rem, w_rem_J, w_rem_rest, J, alpha, sigma, tau, phi, settings)
% Input:
%       w: the matrix of the KXT weights excluding the w_rem masses.
%       settings.rw_std vector of length 2. Contains the std hyper for the
%       sigma and tau correspondingly.
%

[K T] = size(w);
c_rem = C(end, :);
co = [0 c_rem(1:T-1)];
nt= co; %+ c_rem;

nbMH=1;
for nn=1:nbMH
    sum_w = sum(w(1:end, :));
    sumall_w = sum_w + w_rem;
    % Sample (alpha,sigma,tau,w*) from the proposal distribution
    if settings.estimate_sigma
        zz =(log(sigma/(1-sigma)) + settings.rw_std(1)*randn);
        sigmaprop = 1/(1 + exp(-zz));
        
        if sigmaprop<0 || sigmaprop>1
            keyboard
        end
        
    else
        sigmaprop = sigma;
    end
    if settings.estimate_tau
        tauprop = exp(log(tau) + settings.rw_std(2)*randn);
    else
        tauprop = tau;
    end
    if settings.estimate_phi
        phiprop = exp(log(phi) + settings.rw_std(3)*randn);
    else
        phiprop = phi;
    end
    
    
    if settings.estimate_alpha
        
        
        if ~settings.rw_alpha % gamma proposal
            
            fi = 2*phiprop*ones(1,T);
            fi(1) = phiprop;
            bb = sum(GGPpsi( (2.*sum_w + 2.*w_rem + fi), 1, sigmaprop, tauprop));
            
           % bb = (tau^sigmaprop - tauprop^sigmaprop*sum(GGPpsi( (2.*sum_w + 2.*w_rem + fi)/tauprop, 1, sigmaprop, 1)))/sigmaprop
            alphaprop = gamrnd(K, 1/( bb));
        else
            alphaprop = alpha.*exp(.02*randn);
        end
        logalphaprop = log(alphaprop);
    else
        alphaprop = alpha;
    end
    
    Jprop = zeros(1, T);
    for t=1:T
        fi = phiprop;
        if t==1
            fi = 0;
        end
        
        if nt(t)==0
            Jprop(t) =0;
        else
            Jprop(t) = sample_nbclust(alphaprop, sigmaprop, tauprop, nt(t), fi);
        end
    end
    
    wprop_rem_rest = zeros(1, T);
    %wprop_rem_J = zeros(1, T);
    for t=1:T
        bt =2*sum_w(t) + 2*w_rem(t)+ 2*phiprop;
        if t==1
            bt = 2*sum_w(t) + 2*w_rem(t)+ phiprop;
        end
        ff =GGPsumrnd(alphaprop, sigmaprop, tauprop+bt);
        if ff==0
            
            keyboard
        end
        wprop_rem_rest(t) = ff;
        
    end
    wprop_rem_J = gamrnd(nt-sigmaprop.*Jprop, 1/(tauprop + phiprop));
    wprop_rem = wprop_rem_rest + wprop_rem_J;
    
    % Compute the acceptance probability
    % First, compute the fixed terms, i.e. the terms of the ratio without
    % that do not correspond to the prior or proposal of \alpha, \sigma and
    % \tau.
    
    sumall_wprop = sum_w + wprop_rem; % vector of length T
    
    %%%new
    caterm1 = K*(log(alphaprop) - log(alpha));
    
    fi =ones(1,T)*2*phi;
    fi(1) = phi;
    
    fip =ones(1,T)*2*phiprop;
    fip(1) = phiprop;
    
    caterm2 = +sum(GGPpsi((2*sum_w + 2*wprop_rem + fi), alpha, sigma,tau))...
        -sum(GGPpsi((2*sum_w + 2*w_rem + fip), alphaprop, sigmaprop,tauprop));
    
    
    t1 = K*(gammaln(1-sigma) - gammaln(1-sigmaprop));
    
    
    tc  = C(1:end-1, :);
    tc = [zeros(K,1) tc(:, 1:T-1)];
    nzwindii = find(tc>0) ;
    maxxii = max( tc(nzwindii) - sigma, 0 );
    maxxpii = max( tc(nzwindii) - sigmaprop, 0 );
    t2 = sum(gammaln(maxxii) - gammaln(maxxpii));
    
    
    t3 = -(tauprop - tau +2*phiprop-2*phi)* sum(sum_w(2:T)) -(tauprop - tau +phiprop-phi)*(sum_w(1)) ;
    %t3 = -(tauprop - tau +2*phiprop-2*phi)* sum(sum_w);
    
    nzwind = find(w>0);
    maxx = max( tc(nzwind) - sigma, 0 );
    maxxp = max( tc(nzwind) - sigmaprop, 0 );
    t4 = (sigma - sigmaprop)*sum(log(w(nzwind)));
    
    
    
    %t5 = sum(maxxp)*log(tauprop+phi) - sum(maxx)*log(tau+phi);
    t5 = sum(maxxp)*log(tauprop+phiprop) - sum(maxx)*log(tau+phi);
    
    
    if phi~= 0
        t6 = sum(sum(C))*(log(phiprop)-log(phi));
    else
        t6 =(log(phiprop^sum(sum(C))) -log(phi^sum(sum(C))));
    end
    %     if T==1
    %         t6=0;
    %     end
    
    
    t7 = sum(- sumall_wprop.^2 + sumall_w.^2);
    
    
    t8 =sum((2*sumall_w).*wprop_rem_rest -(2*sumall_wprop).*w_rem_rest  );
    
    
    
    %t9 = - phi*sum(wprop_rem_J - w_rem_J);
    t9 = -phiprop*sum(wprop_rem_J)+ phi*sum(w_rem_J);
    
    
    t10 = sum(c_rem.*(log(wprop_rem) - log(w_rem))); %
    
    
    
    fi = phi*ones(1, T);
    fi(1) = 0;
    fip =ones(1,T)*phiprop;
    fip(1) = 0;
    %fi=phi;
    t11 = sum ( GGPpsi(fip, alphaprop, sigmaprop, tauprop) ...
        - GGPpsi(fi, alpha, sigma, tau));
    
    %t10=0;
    
    
    logaccept = t1 + t2+t3+t4+t5+t6+t7+t8+t9+t10 + t11 + caterm1 + caterm2 ;
    
    dbg = logaccept;
    keyboard
    if settings.estimate_alpha
        if ~settings.rw_alpha
            
            
            fi = 2*phi*ones(1,T);
            fi(1) = phi;
            fip = 2*phiprop*ones(1,T);
            fip(1) = phiprop;
            
            
            nom =  log( sum(GGPpsi((2*sum_w + 2*wprop_rem + fi), 1, sigma,tau))) ;
            
            den =  log(sum( GGPpsi((2*sum_w + 2*w_rem + fip), 1, sigmaprop, tauprop)) );
            
            % logaccept
            
            logaccept = logaccept ...
                + K*(nom - den) - caterm1 - caterm2 ...
                +settings.hyper_alpha(1)*( log(alphaprop) - log(alpha))...
                - settings.hyper_alpha(2) * (alphaprop - alpha);
            %exp(logaccept)
            
            % keyboard
            
            
        else % need to look at this AGAIN
            logaccept = logaccept ...
                - exp(logalphaprop + sigmaprop*log(tauprop))* GGPpsi((2*sum_w + 2*w_rem)/tauprop, 1, sigmaprop, 1) ...
                + exp(logalpha + sigma*log(tau)) * GGPpsi((2*sum_wprop + 2*wprop_rem)/tau, 1, sigma, 1)...
                + K*(logalphaprop - logalpha);
        end
        % %
        %         if settings.hyper_alpha(1)>0
        %             logaccept = logaccept + settings.hyper_alpha(1)*( log(alphaprop) - log(alpha));
        %         endp
        %         if settings.hyper_alpha(2)>0
        %             logaccept = logaccept - settings.hyper_alpha(2) * (alphaprop - alpha);
        %         end
        
    end
    
    
    if settings.estimate_tau
        logaccept = logaccept ...
            + settings.hyper_tau(1)*( log(tauprop) - log(tau)) - settings.hyper_tau(2) * (tauprop - tau);
    end
    
    
    if settings.estimate_phi
        logaccept = logaccept ...
            + settings.hyper_phi(1)*( log(phiprop) - log(phi)) - settings.hyper_phi(2) * (phiprop - phi);
    end
    %keyboard
    
    if settings.estimate_sigma
        %         logaccept = logaccept ...
        %             + settings.hyper_sigma(1)*(log(sigmaprop) - log(sigma)) ...
        %             +settings.hyper_sigma(1)*( log(1 - sigma) - log(1-sigmaprop)) ...
        %             -settings.hyper_sigma(2)*((sigmaprop/(1-sigmaprop))-(sigma/(1-sigma)) );
        %
        logaccept = logaccept ...
            + settings.hyper_sigma(1)*(log(sigmaprop) - log(sigma)) ...
            +settings.hyper_sigma(2)*(log(1-sigmaprop) - log(1-sigma));
        
    end
    
    
  % keyboard
    
    if isnan(logaccept)
        keyboard
    end
    % Accept step
    
    if log(rand)<logaccept
        %disp('accept')
        w_rem = wprop_rem;
        alpha = alphaprop;
        sigma = sigmaprop;
        tau = tauprop;
        phi = phiprop;
        w_rem_rest = wprop_rem_rest;
        w_rem_J = wprop_rem_J;
        J = Jprop;
        
   
    end
end
assert(w_rem_J(1) == 0 );
assert(J(1) == 0 );
rate2 = min(1, exp(logaccept));
%keyboard
end



function [x, proba] = sample_nbclust(alpha, sigma, tau, n, b)

[~, logC] = genfact(n,sigma); % computes the log of the generalized factorial coefficients

logtemp = logC(end, :) + (1:n).*log(alpha/sigma*(b+tau)^sigma);
temp = exp(logtemp - max(logtemp));
x = find(sum(temp)*rand<cumsum(temp), 1); % sample from the discrete distribution
proba = temp/sum(temp);

end
