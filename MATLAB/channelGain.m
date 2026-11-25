function h = channelGain(distance, beta, alpha)
% CHANNELGAIN  Distance-dependent path-loss channel gain.
%
%   h = beta * distance^(-alpha)
%
% Inputs:
%   distance - transmitter-receiver distance (meters); scalar or matrix
%   beta     - path-loss constant (our simulation assumption)
%   alpha    - path-loss exponent (our simulation assumption, typ. 2-4)
%
% Output:
%   h        - channel gain, same size as 'distance'
%
% NOTE: distance must be > 0 for a finite result. This function does not
% itself guard against distance = 0; callers should ensure TX and RX
% positions are distinct (true for all node pairs in this project).

    h = beta .* (distance .^ (-alpha));

end
