function sinr = calculateCellularSINR(P_CU, h_CU_BS, txPowerD2D, h_D2D_BS, ...
                                       d2dChannelAssignment, cuChannel, cuIndex, noisePower)
% CALCULATECELLULARSINR  Uplink SINR for CU 'cuIndex', per the
% professor-provided thesis's formulation (Eq. 3.7/3.9): CU transmits to
% the BS; D2D transmitters interfere at the BS if they share the CU's
% channel.
%
% This REPLACES the earlier (incorrect, downlink) version of this file.
%
% Inputs:
%   P_CU               - [1xI] transmit power of each cellular user
%   h_CU_BS            - [Ix1] channel gain, CU i -> BS
%   txPowerD2D         - [1xJ] transmit power of each D2D transmitter
%   h_D2D_BS           - [Jx1] channel gain, D2D transmitter j -> BS
%   d2dChannelAssignment - [1xJ] channel index assigned to each D2D pair
%   cuChannel          - [1xI] uplink channel assigned to each CU
%   cuIndex            - index i of the CU being evaluated
%   noisePower         - receiver (BS) noise power N
%
% Output:
%   sinr               - linear (not dB) uplink SINR for CU 'cuIndex'

    desiredSignal = P_CU(cuIndex) * h_CU_BS(cuIndex);

    interference = calculateCellularInterference(txPowerD2D, h_D2D_BS, ...
                        d2dChannelAssignment, cuChannel, cuIndex);

    sinr = desiredSignal / (noisePower + interference);

end
