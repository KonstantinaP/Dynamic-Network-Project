function [rho] = ggp_sample_rho(rho, nold, nnew, settings)
% Function that implements the slice sampler for parameter rho
dt = settings.dt;
rho_a = settings.hyper_rho(1);
rho_b= settings.hyper_rho(2);

[T] = length(nnew);
%for t=2:T

for t=1:T
    n{t} = nnew{t} + nold{t};
end

% if rand<0.5
ll_term = @(q) compute_lpdf(q, n, nold, rho_a, rho_b, dt);
f = @(y) ll_term(exp(y)) + rho_a*log(rho_b) - rho_b*exp(y)+ rho_a*y;



res= slicesample(log(rho), 1,'logpdf', f, 'thin', 1,'burnin', 0);
rho=exp(res);
%end



%%%%

% else
%     rhoprop = exp(log(rho) + settings.rw_std(4)*randn);
%
%
%     logaccept =0;
%
%     for t=2:T
%         logaccept =  logaccept + -dt*(rhoprop - rho).*nold{t} +(n{t-1} - nold{t})*(log(1 - exp(-rhoprop*dt)) - log(1 -exp(-rho*dt)));
%     end
%
%     logaccept = logaccept ...
%         + settings.hyper_rho(1)*( log(rhoprop) - log(rho)) - settings.hyper_rho(2) * (rhoprop - rho);
%
%     if log(rand)<logaccept
%         %disp('accept')
%         rho = rhoprop;
%
%
%     end
% end

    function out = compute_lpdf(r, nmt, no, rho_a, rho_b, dt)
        
        w =length(nmt);
        %keyboard
        out=0;
                for t=2:w
                    out = out + sum(sum(-r*dt.*no{t} +(nmt{t-1} - no{t}).*log(1 - exp(-r*dt))));
        %             keyboard
                end
                
                
%                 noal =[];
%         nal=[];
%         for t=2:w
%             
%             noal = [noal no{t}(:)];
%             nal = [nal nmt{t}(:)];
%         end
%         noall= noal(nal>0);
%         dif = nal(nal>0) - noall;
%         out
%         out2 = sum(sum(-r*dt.*noall +dif.*log(1 - exp(-r*dt))))
%         keyboard
    end
end
