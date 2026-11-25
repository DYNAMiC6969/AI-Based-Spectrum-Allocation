%% Stage 10 - Reward Function Design and Manual Validation
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% SCOPE OF THIS STAGE:
% Computes the reward for two manually chosen channel allocations, using
% the D2D model (Stages 6-8) and the CORRECTED cellular UPLINK model
% (Stage 9: CU -> BS, D2D interferes at the BS). Does NOT implement the
% Q-table, epsilon-greedy, training loop, or Q-learning update — those
% are Stage 11.
%
% NOTE: referenceRate and lambdaQoS are PROJECT DESIGN PARAMETERS,
% explicitly not taken from the thesis (see Stage 10 document, Sections
% 5 and 7, for how each is chosen and why).

clear; clc; close all;

%% 1. Load fixed network positions and all channel-gain matrices
net = generatePositions();

beta  = 1;    % path-loss constant (assumption, same as prior stages)
alpha = 3;    % path-loss exponent (assumption, same as prior stages)

numD2D = size(net.D2D_TX, 1);   % = 2
numCU  = size(net.CU, 1);       % = 3

% D2D-to-D2D channel gain matrix (Stage 7)
D_d2d = zeros(numD2D, numD2D);
for i = 1:numD2D
    for j = 1:numD2D
        D_d2d(i,j) = calculateDistance(net.D2D_TX(i,:), net.D2D_RX(j,:));
    end
end
H_D2D = channelGain(D_d2d, beta, alpha);

% CU-to-BS gain vector (Stage 9, corrected uplink model)
d_CU_BS = zeros(numCU, 1);
for i = 1:numCU
    d_CU_BS(i) = calculateDistance(net.CU(i,:), net.BS);
end
h_CU_BS = channelGain(d_CU_BS, beta, alpha);

% D2D-transmitter-to-BS gain vector (Stage 9, corrected uplink model)
d_D2D_BS = zeros(numD2D, 1);
for j = 1:numD2D
    d_D2D_BS(j) = calculateDistance(net.D2D_TX(j,:), net.BS);
end
h_D2D_BS = channelGain(d_D2D_BS, beta, alpha);

%% 2. Simulation parameters (carried over from Stages 8-9)
txPowerD2D = [1 1];
P_CU       = [5 5 5];
noisePower = 1e-6;
W          = 1;           % rates in bits/s/Hz
cuChannel  = [1 2 3];     % CU1->Ch1, CU2->Ch2, CU3->Ch3
R_min      = [5 5 5];     % bits/s/Hz

%% 3. Stage 10 reward-design parameters (NEW project design choices)

% referenceRate: the total D2D rate achievable with ZERO D2D-to-D2D
% interference (i.e. both D2D pairs on distinct, non-interfering
% channels), computed once from this network's own geometry. Using the
% network's own best case as the normalization anchor means
% normalizedD2DRate = 1.0 represents "no D2D self-interference", not an
% arbitrary external number.
noInterferenceSINR = (txPowerD2D .* diag(H_D2D)') ./ noisePower;
noInterferenceRates = W .* log2(1 + noInterferenceSINR);
referenceRate = sum(noInterferenceRates);

% lambdaQoS: weight per QoS violation. Chosen as 1 for this initial
% implementation (a violation costs exactly one full unit of normalized
% D2D throughput) - a simple, easily explainable starting point, and a
% project design parameter subject to tuning in later stages.
lambdaQoS = 1;

fprintf('Reference D2D rate (no D2D-D2D interference) = %.4f bits/s/Hz\n', referenceRate);
fprintf('lambdaQoS = %.2f\n', lambdaQoS);

%% 4. Define the two test channel allocations (as specified for Stage 10)
allocA = [3 3];   % both D2D pairs on channel 3 (= CU3's channel)
allocB = [3 7];   % D2D pair 1 on channel 3 (= CU3's channel), pair 2 on channel 7

allocations = {allocA, allocB};
allocNames  = {'Allocation A: D2D A=[3,3]', 'Allocation B: D2D A=[3,7]'};

%% 5. Evaluate reward for each allocation
for a = 1:length(allocations)

    A = allocations{a};
    fprintf('\n===== %s =====\n', allocNames{a});

    %% -- D2D performance --
    d2dRates = zeros(1, numD2D);
    for j = 1:numD2D
        sinr_j = calculateSINR(txPowerD2D, H_D2D, A, j, noisePower);
        d2dRates(j) = calculateDataRate(sinr_j, W);
    end

    disp('D2D rates (bits/s/Hz):');
    disp(d2dRates);

    %% -- Cellular uplink performance --
    cuRates = zeros(1, numCU);
    for i = 1:numCU
        sinr_i = calculateCellularSINR(P_CU, h_CU_BS, txPowerD2D, h_D2D_BS, ...
                                        A, cuChannel, i, noisePower);
        cuRates(i) = calculateDataRate(sinr_i, W);
    end

    disp('Cellular uplink rates (bits/s/Hz):');
    disp(cuRates);

    %% -- Reward --
    [reward, totalD2DRate, normalizedD2DRate, numberOfQoSViolations] = ...
        calculateReward(d2dRates, cuRates, R_min, referenceRate, lambdaQoS);

    fprintf('Total D2D rate         = %.4f bits/s/Hz\n', totalD2DRate);
    fprintf('Normalized D2D rate    = %.4f\n', normalizedD2DRate);
    fprintf('Number of QoS violations = %d\n', numberOfQoSViolations);
    fprintf('REWARD                 = %.4f\n', reward);

end
