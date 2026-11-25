%% Stage 11 - Tabular Q-Learning for D2D Channel Allocation
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% *** MODEL CONSISTENCY CHECK (per Stage 11 instructions) ***
% Verified before proceeding: calculateCellularInterference.m and
% calculateCellularSINR.m implement the CORRECTED cellular UPLINK model
% (CU -> BS desired signal, D2D TX -> BS interference), matching Stage 9
% and the professor-provided thesis. No inconsistency found in the
% functions reused from prior stages. This script does not modify any
% wireless-environment function; it only adds the RL layer on top.
%
% AGENT SCOPE: the agent controls ONLY D2D pair 1's channel. D2D pair
% 2's channel ("otherD2DChannel") is redrawn uniformly at random every
% time step, simulating its unmodeled, autonomous behavior. This is a
% project simulation assumption, not from the thesis - see the Stage 11
% document, Section 2.
%
% Do NOT use DQN, PPO, Actor-Critic, or neural networks anywhere here.

clear; clc; close all;

%% 1. Load fixed network positions and all channel-gain matrices (Stages 6,7,9)
net = generatePositions();

beta  = 1;    % path-loss constant (assumption, same as prior stages)
alpha = 3;    % path-loss exponent (assumption, same as prior stages)

numD2D = size(net.D2D_TX, 1);   % = 2
numCU  = size(net.CU, 1);       % = 3

D_d2d = zeros(numD2D, numD2D);
for i = 1:numD2D
    for j = 1:numD2D
        D_d2d(i,j) = calculateDistance(net.D2D_TX(i,:), net.D2D_RX(j,:));
    end
end
H_D2D = channelGain(D_d2d, beta, alpha);

d_CU_BS = zeros(numCU, 1);
for i = 1:numCU
    d_CU_BS(i) = calculateDistance(net.CU(i,:), net.BS);
end
h_CU_BS = channelGain(d_CU_BS, beta, alpha);

d_D2D_BS = zeros(numD2D, 1);
for j = 1:numD2D
    d_D2D_BS(j) = calculateDistance(net.D2D_TX(j,:), net.BS);
end
h_D2D_BS = channelGain(d_D2D_BS, beta, alpha);

%% 2. Environment parameters (carried over from Stages 8-10)
envParams.txPowerD2D = [1 1];
envParams.P_CU       = [5 5 5];
envParams.noisePower = 1e-6;
envParams.W          = 1;
envParams.cuChannel  = [1 2 3];
envParams.R_min      = [5 5 5];

noInterferenceSINR = (envParams.txPowerD2D .* diag(H_D2D)') ./ envParams.noisePower;
noInterferenceRates = envParams.W .* log2(1 + noInterferenceSINR);
envParams.referenceRate = sum(noInterferenceRates);   % Stage 10 definition
envParams.lambdaQoS = 1;                              % Stage 10 assumption

fprintf('referenceRate = %.4f bits/s/Hz\n', envParams.referenceRate);

%% 3. Q-learning hyperparameters (simulation hyperparameters, NOT thesis-derived)
hyperParams.alpha        = 0.1;
hyperParams.gamma        = 0.9;
hyperParams.epsilon      = 1.0;
hyperParams.epsilonMin   = 0.05;
hyperParams.epsilonDecay = 0.995;
hyperParams.maxTimeSteps = 50;

%% ============================================================
%% VALIDATION RUN (small number of episodes, checked before full run)
%% ============================================================
fprintf('\n===== VALIDATION RUN (20 episodes) =====\n');
hyperParamsValidation = hyperParams;
hyperParamsValidation.numEpisodes = 20;

[Qval, metricsVal] = trainQLearning(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParamsValidation);

checks = struct();
checks.qTableSizeOK   = isequal(size(Qval), [32 8]);
checks.rewardsFinite  = all(isfinite(metricsVal.episodeReward));
checks.noNaNInQ       = ~any(isnan(Qval(:)));
checks.noInfInQ       = ~any(isinf(Qval(:)));
checks.epsilonDecreasing = all(diff(metricsVal.epsilonHistory) <= 0);
checks.qValuesChanged = any(Qval(:) ~= 0);

fprintf('Q-table is 32x8:                 %s\n', mat2str(checks.qTableSizeOK));
fprintf('All episode rewards finite:      %s\n', mat2str(checks.rewardsFinite));
fprintf('No NaN in Q-table:                %s\n', mat2str(checks.noNaNInQ));
fprintf('No Inf in Q-table:                %s\n', mat2str(checks.noInfInQ));
fprintf('Epsilon non-increasing over time: %s\n', mat2str(checks.epsilonDecreasing));
fprintf('Q-values changed from zero:       %s\n', mat2str(checks.qValuesChanged));

allChecksPassed = checks.qTableSizeOK && checks.rewardsFinite && checks.noNaNInQ && ...
                   checks.noInfInQ && checks.epsilonDecreasing && checks.qValuesChanged;

if ~allChecksPassed
    error('Validation checks FAILED - fix issues before running full training.');
end
disp('All validation checks PASSED.');

%% ============================================================
%% FULL TRAINING RUN
%% ============================================================
fprintf('\n===== FULL TRAINING RUN =====\n');
hyperParams.numEpisodes = 3000;

[Q, metrics] = trainQLearning(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams);

fprintf('Training complete. Final epsilon = %.4f\n', metrics.epsilonHistory(end));
fprintf('Mean episode return (sum over %d steps), last 100 episodes = %.4f\n', hyperParams.maxTimeSteps, mean(metrics.episodeReward(end-99:end)));
fprintf('Mean per-step reward, last 100 episodes = %.4f\n', mean(metrics.averageReward(end-99:end)));
fprintf('Mean QoS violation rate (per CU per step), last 100 episodes = %.4f\n', mean(metrics.QoSViolationRate(end-99:end)));

%% 4. Plots
movAvgWindow = 50;

figure;
plot(movmean(metrics.episodeReward, movAvgWindow));
xlabel('Episode (moving average)'); ylabel('Episode Return (sum over 50 steps)');
title('Episode vs Total Reward'); grid on;
saveas(gcf, 'stage11_plot1_reward.png');

figure;
plot(movmean(metrics.averageD2DThroughput, movAvgWindow));
xlabel('Episode (moving average)'); ylabel('Aggregate D2D Throughput (bits/s/Hz)');
title('Episode vs Aggregate D2D Throughput'); grid on;
saveas(gcf, 'stage11_plot2_d2d_throughput.png');

figure;
plot(movmean(metrics.averageCellularRate, movAvgWindow));
xlabel('Episode (moving average)'); ylabel('Avg Cellular Throughput (bits/s/Hz)');
title('Episode vs Average Cellular Throughput'); grid on;
saveas(gcf, 'stage11_plot3_cellular_throughput.png');

figure;
plot(movmean(metrics.QoSViolationRate, movAvgWindow));
xlabel('Episode (moving average)'); ylabel('QoS Violation Rate (per CU per step)');
title('Episode vs QoS Violation Rate'); grid on;
saveas(gcf, 'stage11_plot4_qos_violations.png');

figure;
plot(metrics.epsilonHistory);
xlabel('Episode'); ylabel('Epsilon');
title('Episode vs Epsilon'); grid on;
saveas(gcf, 'stage11_plot5_epsilon.png');

%% 5. Learned policy and final Q-table
policy = zeros(1, 32);
for s = 1:32
    [~, policy(s)] = max(Q(s, :));
end

disp(' ');
disp('--- Learned Policy (state -> best action/channel) ---');
disp('State | otherD2DChannel | QoSViolationCount | Best Action (Channel)');
for otherCh = 1:8
    for qos = 0:3
        s = getState(otherCh, qos);
        fprintf('%5d | %15d | %18d | %d\n', s, otherCh, qos, policy(s));
    end
end

disp(' ');
disp('--- Final Q-table (32x8) ---');
disp(Q);

%% 6. Save trained Q-table and environment setup for Stage 12 (evaluation)
% This lets Stage 12 load the FROZEN, already-trained Q-table rather than
% retraining the Q-learning algorithm, per the Stage 12 instructions.
save('qlearning_trained.mat', 'Q', 'policy', 'envParams', 'hyperParams', ...
     'H_D2D', 'h_CU_BS', 'h_D2D_BS', 'net');
disp(' ');
disp('Saved trained Q-table and environment setup to qlearning_trained.mat');
