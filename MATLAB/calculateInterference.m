function interference = calculateInterference(txPower, H, channelAssignment, pairIndex)
% CALCULATEINTERFERENCE  Same-channel D2D-to-D2D interference at the
% receiver of D2D pair 'pairIndex'.
%
% A D2D transmitter k interferes with D2D receiver j only when both are
% assigned the same channel (a_k = a_j), k ~= j.
%
% Inputs:
%   txPower           - [1xN] transmit power of each D2D transmitter
%   H                 - [NxN] D2D channel-gain matrix, H(k,j) = gain
%                       from transmitter k to receiver j
%   channelAssignment - [1xN] channel index assigned to each D2D pair
%   pairIndex         - index j of the D2D pair whose receiver
%                       interference is being calculated
%
% Output:
%   interference      - total interference power at receiver 'pairIndex'
%                        (does NOT include noise)
%
% NOTE: this models cross-D2D interference only. Cellular-user
% interference is explicitly out of scope for this stage.

    interference = 0;
    numD2D = length(channelAssignment);

    for k = 1:numD2D
        if k ~= pairIndex && ...
           channelAssignment(k) == channelAssignment(pairIndex)

            interference = interference + txPower(k) * H(k, pairIndex);
        end
    end

end
