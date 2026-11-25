%% Stage 9 (CORRECTED) - Cellular Uplink QoS and D2D-to-BS Interference
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% *** MODELING DIRECTION: UPLINK (per professor-provided thesis) ***
% This REPLACES the earlier downlink version of Stage 9. The cellular
% link modeled here is CU -> BS (uplink), matching the thesis's Eq.
% 3.7/3.9 and consistent with Stage 5. D2D transmitters interfere AT THE
% BS, not at the CU receiver.
%
% SCOPE: cellular uplink desired signal, D2D-to-BS interference,
% cellular uplink SINR/rate, QoS violation check, plus the existing
% D2D-to-D2D model. Does NOT implement the reward function or
% Q-learning.
%
% NOTE: P_CU (cellular transmit power) is a NEW simulation assumption
% introduced at this stage (not previously established). R_min is
% carried over from the earlier (now-superseded) Stage 9 draft, since a
% minimum-rate threshold is direction-agnostic; it is flagged here as a
% simulation assumption, not a thesis-derived value.

clear; clc; close all;

%% 1. Load fixed network positions (Stage 6) and D2D channel gains (Stage 7)
net = generatePositions();

beta  = 1;    % path-loss constant (assumption, same as Stages 7-8)
alpha = 3;    % path-loss exponent (assumption, same as Stages 7-8)

numD2D = size(net.D2D_TX, 1);   % = 2
numCU  = size(net.CU, 1);       % = 3

% D2D-to-D2D channel gain matrix H_D2D (Stage 7) - unchanged
D_d2d = zeros(numD2D, numD2D);
for i = 1:numD2D
    for j = 1:numD2D
        D_d2d(i,j) = calculateDistance(net.D2D_TX(i,:), net.D2D_RX(j,:));
    end
end
H_D2D = channelGain(D_d2d, beta, alpha);

%% 2. Cellular UPLINK channel-gain vectors (NEW / corrected for Stage 9)

% CU-to-BS gain vector (3x1): h_CU_BS(i) = gain from CU i to BS
d_CU_BS = zeros(numCU, 1);
for i = 1:numCU
    d_CU_BS(i) = calculateDistance(net.CU(i,:), net.BS);
end
h_CU_BS = channelGain(d_CU_BS, beta, alpha);

% D2D-transmitter-to-BS gain vector (2x1): h_D2D_BS(j) = gain from D2D
% transmitter j to BS
d_D2D_BS = zeros(numD2D, 1);
for j = 1:numD2D
    d_D2D_BS(j) = calculateDistance(net.D2D_TX(j,:), net.BS);
end
h_D2D_BS = channelGain(d_D2D_BS, beta, alpha);

disp('--- CU-to-BS Channel Gain Vector h_CU_BS (3x1) ---');
disp(h_CU_BS);
disp('--- D2D-Transmitter-to-BS Channel Gain Vector h_D2D_BS (2x1) ---');
disp(h_D2D_BS);

%% 3. Simulation parameters (our assumptions)
txPowerD2D = [1 1];      % D2D transmit power (same as Stages 8-9 draft)
P_CU       = [5 5 5];    % cellular transmit power (NEW assumption, chosen
                         % so all CUs clear R_min absent D2D interference)
noisePower = 1e-6;       % receiver (BS) noise power N (same as before)
W          = 1;          % bandwidth = 1 -> rates in bits/s/Hz (same as before)

% Cellular resource assignment: OUR initial simulation assumption
% (unchanged from the earlier draft)
cuChannel = [1 2 3];     % CU1->Ch1, CU2->Ch2, CU3->Ch3

% Minimum acceptable cellular rate per CU (carried over from the earlier
% Stage 9 draft as a simulation assumption; direction-agnostic)
R_min = [5 5 5];         % bits/s/Hz

%% 4. Define two D2D channel allocations to compare (same as earlier draft)
allocX = [1 7];   % D2D pair 1 reuses CU1's channel (1); pair 2 uses channel 7
allocY = [4 7];   % neither D2D pair uses any CU's channel

allocations = {allocX, allocY};
allocNames  = {'Allocation X: D2D A=[1,7] (pair 1 reuses CU1 channel)', ...
               'Allocation Y: D2D A=[4,7] (no D2D pair on a CU channel)'};

%% 5. Evaluate each allocation: full D2D + cellular (uplink) environment output
for a = 1:length(allocations)

    A = allocations{a};
    fprintf('\n===== %s =====\n', allocNames{a});

    %% -- D2D performance (unchanged model) --
    d2dInterference = zeros(1, numD2D);
    d2dSINR         = zeros(1, numD2D);
    d2dRate         = zeros(1, numD2D);

    for j = 1:numD2D
        d2dInterference(j) = calculateInterference(txPowerD2D, H_D2D, A, j);
        d2dSINR(j)          = calculateSINR(txPowerD2D, H_D2D, A, j, noisePower);
        d2dRate(j)          = calculateDataRate(d2dSINR(j), W);
    end
    aggregateD2DThroughput = sum(d2dRate);

    disp('--- D2D Performance ---');
    for j = 1:numD2D
        fprintf('D2D Pair %d: Interference=%.6g, SINR=%.4f, Rate=%.4f bits/s/Hz\n', ...
            j, d2dInterference(j), d2dSINR(j), d2dRate(j));
    end
    fprintf('Aggregate D2D Throughput = %.4f bits/s/Hz\n', aggregateD2DThroughput);

    %% -- Cellular UPLINK performance and QoS --
    cuInterference = zeros(1, numCU);   % D2D-to-BS interference per CU
    cuSINR         = zeros(1, numCU);
    cuRate         = zeros(1, numCU);
    cuViolation    = zeros(1, numCU);

    for i = 1:numCU
        cuInterference(i) = calculateCellularInterference(txPowerD2D, h_D2D_BS, A, cuChannel, i);
        cuSINR(i)          = calculateCellularSINR(P_CU, h_CU_BS, txPowerD2D, h_D2D_BS, ...
                                                    A, cuChannel, i, noisePower);
        cuRate(i)          = calculateDataRate(cuSINR(i), W);
        cuViolation(i)      = calculateQoSViolation(cuRate(i), R_min(i));
    end
    N_QoS = sum(cuViolation);

    disp('--- Cellular Uplink Performance and QoS ---');
    for i = 1:numCU
        fprintf('CU%d (channel %d): D2D-to-BS Interference=%.6g, SINR=%.4f, Rate=%.4f bits/s/Hz, R_min=%.1f, Violation=%d\n', ...
            i, cuChannel(i), cuInterference(i), cuSINR(i), cuRate(i), R_min(i), cuViolation(i));
    end
    fprintf('Total QoS Violations N_QoS = %d\n', N_QoS);

end
