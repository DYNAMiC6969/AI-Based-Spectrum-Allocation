function interference = calculateCellularInterference(txPowerD2D, h_D2D_BS, d2dChannelAssignment, cuChannel, cuIndex)
% CALCULATECELLULARINTERFERENCE  D2D-to-BS interference affecting the
% uplink reception of CU 'cuIndex'.
%
% UPLINK MODEL (per professor-provided thesis): the cellular receiver is
% the BASE STATION, not the CU. This computes interference AT THE BS,
% caused by D2D transmitters that share CU 'cuIndex's assigned channel.
% This REPLACES the earlier (incorrect, downlink) version of this file,
% which computed interference at the CU receiver instead.
%
% Inputs:
%   txPowerD2D         - [1xJ] transmit power of each D2D transmitter
%   h_D2D_BS           - [Jx1] channel gain, D2D transmitter j -> BS
%   d2dChannelAssignment - [1xJ] channel index assigned to each D2D pair
%   cuChannel          - [1xI] channel index (uplink resource) assigned
%                        to each CU
%   cuIndex            - index i of the CU whose uplink is being evaluated
%
% Output:
%   interference       - total D2D-to-BS interference power affecting
%                         CU 'cuIndex's uplink (does NOT include noise)

    interference = 0;
    numD2D = length(d2dChannelAssignment);

    for j = 1:numD2D
        if d2dChannelAssignment(j) == cuChannel(cuIndex)
            interference = interference + txPowerD2D(j) * h_D2D_BS(j);
        end
    end

end
