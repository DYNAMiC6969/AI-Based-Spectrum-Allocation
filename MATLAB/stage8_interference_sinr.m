%% Stage 8 - Interference and SINR (D2D-to-D2D only)
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% SCOPE OF THIS STAGE:
% This script implements cross-D2D interference, SINR, and data rate,
% and verifies that channel allocation actually changes these values by
% comparing two manual allocations. It does NOT implement Q-learning,
% the Q-table, epsilon-greedy, the reward function, or any cellular-user
% interference model — those are later stages.
%
% NOTE: beta, alpha, txPower, and noisePower values below are OUR
% simulation assumptions, not values taken from the professor-provided
% thesis.

clear; clc; close all;

%% 1. Load fixed network positions (Stage 6) and channel-gain matrix (Stage 7)
net = generatePositions();

beta  = 1;    % path-loss constant (assumption, same as Stage 7)
alpha = 3;    % path-loss exponent (assumption, same as Stage 7)

numPairs = size(net.D2D_TX, 1);   % = 2

D = zeros(numPairs, numPairs);
for i = 1:numPairs
    for j = 1:numPairs
        D(i,j) = calculateDistance(net.D2D_TX(i,:), net.D2D_RX(j,:));
    end
end

H = channelGain(D, beta, alpha);

disp('--- D2D Channel Gain Matrix H (from Stage 7) ---');
disp(H);

%% 2. Simulation parameters for this stage (our assumptions)
txPower    = [1 1];   % transmit power of D2D_TX1, D2D_TX2
noisePower = 1e-6;    % receiver noise power N
W          = 1;       % channel bandwidth (=1 -> rates below are in bits/s/Hz)

%% 3. Define the two test channel allocations
caseA = [3 3];   % both D2D pairs on the SAME channel -> interference expected
caseB = [3 7];   % D2D pairs on DIFFERENT channels    -> no cross-D2D interference

cases = {caseA, caseB};
caseNames = {'Case A: A = [3,3]', 'Case B: A = [3,7]'};

%% 4. Evaluate each case
for c = 1:length(cases)

    A = cases{c};
    fprintf('\n===== %s =====\n', caseNames{c});

    interference = zeros(1, numPairs);
    sinrLinear    = zeros(1, numPairs);
    sinrDB        = zeros(1, numPairs);
    dataRate      = zeros(1, numPairs);
    desiredSignal = zeros(1, numPairs);

    for j = 1:numPairs
        desiredSignal(j) = txPower(j) * H(j,j);
        interference(j)  = calculateInterference(txPower, H, A, j);
        sinrLinear(j)    = calculateSINR(txPower, H, A, j, noisePower);
        sinrDB(j)        = 10 * log10(sinrLinear(j));
        dataRate(j)      = calculateDataRate(sinrLinear(j), W);
    end

    aggregateThroughput = sum(dataRate);

    for j = 1:numPairs
        fprintf('D2D Pair %d:\n', j);
        fprintf('  Desired signal S_%d      = %.6g\n', j, desiredSignal(j));
        fprintf('  Interference I_%d        = %.6g\n', j, interference(j));
        fprintf('  SINR (linear)            = %.6g\n', sinrLinear(j));
        fprintf('  SINR (dB)                = %.2f dB\n', sinrDB(j));
        fprintf('  Data rate (bits/s/Hz)    = %.4f\n', dataRate(j));
    end

    fprintf('Aggregate D2D throughput (bits/s/Hz) = %.4f\n', aggregateThroughput);
end
