%% Function that jointly samples J and c_rem
function [ J c_rem] = ggp_sample_J_crem(c_rem, w_rem, w_rem_J, J, alpha, sigma, tau, phi)

% this works for 0<sigma<1 (not =0)
T = length(c_rem);
nt = [0 c_rem(1:T-1)];
c=c_rem(1:T-1);
j = J(2:T); % J is of T length with J(1) =0  always

nbMH =1;
for nn=1:nbMH 
  cnew = tpoissrnd(phi.*(w_rem(1:T-1))); % proposal from zero-truncarted Poisson distribution
  jnew = zeros(1, T-1);
   for t=1:T-1
        fi = phi;
        if t==1
            fi = 0;
            jnew(t) = 0;
        end
        
        jnew(t) = sample_nbclust(alpha, sigma, tau, cnew(t), fi);
%         keyboard
        
   end
    logaccept =  (cnew - c_rem(1:T-1) - sigma.*(jnew - J(2:T))).*log(tau+phi) ...
    +(cnew - c_rem(1:T-1) - sigma.*(jnew - J(2:T))).*log(w_rem_J(2:T))...
    + gammaln(c_rem(1:T-1) -sigma.*J(2:T))-gammaln(cnew - sigma.*jnew);

  u = log(rand(1, T-1));
  

    c(u<logaccept & logical(w_rem_J(2:T)>0) ) = cnew(u<logaccept & logical(w_rem_J(2:T)>0));
    j(u<logaccept & logical(w_rem_J(2:T)>0)) = jnew(u<logaccept & logical(w_rem_J(2:T)>0));
end

% % *** HERE I NEED TO TAKE INTO ACCOUNT THE FACT THAT WEIGHTS CAN BE ZERO ***
%term =phi.*w_rem(1:T-1).*(phi+tau);
%p1 = -log(1 + term);
%p2 = log(term) -log(1 + term);

%d(1,:) = exp(p1);
%d(2,:) = exp(p2);

c(w_rem_J(2:T)==0) = 0;
j(w_rem_J(2:T)==0) = 0;
%c(w_rem_J(2:T)==0 & (rand(size(w_rem_J(2:T)))> d(1,:)) ) = 1;


c_rem(1:end-1) = c;
c_rem( end) = poissrnd(phi.*(w_rem(T)));

J(2:T) = j;
J(1) =0;



end





function [x, proba] = sample_nbclust(alpha, sigma, tau, n, b)

[~, logC] = genfact(n,sigma); % computes the log of the generalized factorial coefficients

logtemp = logC(end, :) + (1:n).*log(alpha/sigma*(b+tau)^sigma);
temp = exp(logtemp - max(logtemp));
x = find(sum(temp)*rand<cumsum(temp), 1); % sample from the discrete distribution
proba = temp/sum(temp);
end


