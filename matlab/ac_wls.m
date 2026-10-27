function [x, J, converged, iterations] = ac_wls(z, Ybus, sigma)
% Blind, fixed-flat initialization; never initialized from true/fake states.
% Optimize log(Vm) to keep magnitudes positive; no normal-range voltage bounds.
u = zeros(27,1);
converged = false;
for iterations = 1:80
    x = [exp(u(1:14)); u(15:27)];
    [h, H] = measurement_model(x, Ybus);
    r = z - h;
    A = H; A(:,1:14) = H(:,1:14) .* x(1:14).';
    J = sum(r.^2) / sigma^2;
    gradient = A.' * r;
    if norm(gradient, inf) < 1e-9
        converged = true;
        break;
    end
    step = A \ r;
    if ~all(isfinite(step)), break; end
    if norm(step, inf) < 1e-10
        converged = true;
        break;
    end
    step = step * min(1, 0.5/max(abs(step)));
    accepted = false;
    for ls = 0:24
        candidate = u + step * 2^(-ls);
        if any(abs(candidate(1:14)) > 20), continue; end
        xc = [exp(candidate(1:14)); candidate(15:27)];
        rc = z - measurement_model(xc, Ybus);
        Jc = sum(rc.^2) / sigma^2;
        if isfinite(Jc) && Jc < J
            accepted = true; u = candidate; break;
        end
    end
    if ~accepted
        % Numerical stagnation is accepted only with a small first-order residual.
        converged = norm(gradient, inf) < 1e-7;
        break;
    end
end
x = [exp(u(1:14)); u(15:27)];
r = z - measurement_model(x, Ybus);
J = sum(r.^2) / sigma^2;
converged = logical(converged && isfinite(J));
end
