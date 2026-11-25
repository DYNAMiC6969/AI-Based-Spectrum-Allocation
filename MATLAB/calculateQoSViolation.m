function violation = calculateQoSViolation(rate, rMin)
% CALCULATEQOSVIOLATION  Binary QoS violation indicator.
%
%   violation = 1 if rate < rMin, else 0
%
% Inputs:
%   rate - achieved data rate (e.g. from calculateDataRate.m)
%   rMin - minimum acceptable rate threshold (our simulation assumption)
%
% Output:
%   violation - logical/binary indicator (1 = QoS violated)
%
% NOTE: no reward penalty is assigned here. Reward design, including how
% heavily a QoS violation is weighted, is explicitly deferred to Stage 10.

    violation = rate < rMin;

end
