function [h, H] = measurement_model(x, Ybus)
% x=[Vm1..14;theta2..14], radians; all net injections, pu.
v = x(1:14); theta = [0; x(15:27)];
V = v .* exp(1i*theta);
S = V .* conj(Ybus * V);
h = [real(S); imag(S)];
if nargout > 1
    [dVa, dVm] = dSbus_dV(Ybus, V);
    H = full([real(dVm), real(dVa(:,2:14)); imag(dVm), imag(dVa(:,2:14))]);
end
end
