function sinr = calculateSINR(txPower, H, channelAssignment, pairIndex, noisePower)
% CALCULATESINR  Signal-to-Interference-plus-Noise Ratio for D2D pair
% 'pairIndex', given the current channel assignment.
%
% Inputs:
%   txPower           - [1xN] transmit power of each D2D transmitter
%   H                 - [NxN] D2D channel-gain matrix
%   channelAssignment - [1xN] channel index assigned to each D2D pair
%   pairIndex         - index j of the D2D pair whose SINR is computed
%   noisePower        - receiver noise power N
%
% Output:
%   sinr              - linear (not dB) SINR for D2D pair 'pairIndex'

    desiredSignal = txPower(pairIndex) * H(pairIndex, pairIndex);

    interference = calculateInterference(txPower, H, channelAssignment, pairIndex);

    sinr = desiredSignal / (noisePower + interference);

end
