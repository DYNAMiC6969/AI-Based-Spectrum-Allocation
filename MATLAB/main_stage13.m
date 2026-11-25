%% Stage 13 - State and Environment Validation (Model A vs Model B)
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% *** WIRELESS MODEL: UNCHANGED FROM STAGE 9/10/11/12 ***
% Cellular uplink (CU -> BS), D2D-to-D2D, D2D-to-BS interference.
%
% *** TERMINOLOGY (per Stage 13 instructions) ***
% This is tabular Q-learning-based channel selection for a single D2D
% transmitter (D2D pair 1) in a cellular network. D2D pair 2's channel
% is redrawn randomly every step and is NOT controlled by any agent.
% This is NOT multi-agent reinforcement learning.
%
% MODEL A (baseline, from Stage 11/12): state = [otherD2DChannel,
%   QoSViolationCount], 32 states.
% MODEL B (new, this stage): state = [otherD2DChannel, q1,q2,q3],
%   64 states, distinguishing WHICH CU(s) are violated.
%
% Both models are trained and evaluated under IDENTICAL environment,
% reward, hyperparameters, episode/step counts - only the state
% representation differs.

clear; clc; close all;

%% 1. Load fixed network positions and all channel-gain matrices (Stages 6,7,9)
net = generatePositions();

beta  = 1;
alpha = 3;

numD2D = size(net.D2D_TX, 1);
numCU  = size(net.CU, 1);

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

%% 2. Environment parameters (IDENTICAL for both models, unchanged from Stages 8-12)
envParams.txPowerD2D = [1 1];
envParams.P_CU       = [5 5 5];
envParams.noisePower = 1e-6;
envParams.W          = 1;
envParams.cuChannel  = [1 2 3];
envParams.R_min      = [5 5 5];

noInterferenceSINR = (envParams.txPowerD2D .* diag(H_D2D)') ./ envParams.noisePower;
noInterferenceRates = envParams.W .* log2(1 + noInterferenceSINR);
envParams.referenceRate = sum(noInterferenceRates);
envParams.lambdaQoS = 1;

%% 3. Q-learning hyperparameters (IDENTICAL for both models, unchanged from Stage 11/12)
hyperParams.alpha        = 0.1;
hyperParams.gamma        = 0.9;
hyperParams.epsilon      = 1.0;
hyperParams.epsilonMin   = 0.05;
hyperParams.epsilonDecay = 0.995;
hyperParams.maxTimeSteps = 50;
hyperParams.numEpisodes  = 3000;

evaluationSteps   = 1000;   % IDENTICAL for both models (unchanged from Stage 12)
numEvaluationRuns = 10;     % IDENTICAL for both models

%% 4. Train and evaluate both models
fprintf('===== TRAINING AND EVALUATING MODEL A AND MODEL B =====\n');
results = compareStateModels(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams, ...
                              evaluationSteps, numEvaluationRuns);

%% ============================================================
%% VALIDATION CHECKS
%% ============================================================
fprintf('\n===== VALIDATION CHECKS =====\n');

check.modelA_QDims   = isequal(size(results.Q_A), [32 8]);
check.modelB_QDims   = isequal(size(results.Q_B), [64 8]);

allMetricsA = [results.evalA.d2dThroughput, results.evalA.cuThroughput, ...
               results.evalA.qosViolationRate, results.evalA.reward];
allMetricsB = [results.evalB.d2dThroughput, results.evalB.cuThroughput, ...
               results.evalB.qosViolationRate, results.evalB.reward];

check.finiteRewardsA = all(isfinite(results.evalA.reward));
check.finiteRewardsB = all(isfinite(results.evalB.reward));
check.noNaNInfA      = ~any(isnan(allMetricsA)) && ~any(isinf(allMetricsA));
check.noNaNInfB      = ~any(isnan(allMetricsB)) && ~any(isinf(allMetricsB));
check.nonnegRatesA   = all(results.evalA.d2dThroughput >= 0) && all(results.evalA.cuThroughput >= 0);
check.nonnegRatesB   = all(results.evalB.d2dThroughput >= 0) && all(results.evalB.cuThroughput >= 0);
check.qosRangeA      = all(results.evalA.qosViolationRate >= 0 & results.evalA.qosViolationRate <= 1);
check.qosRangeB      = all(results.evalB.qosViolationRate >= 0 & results.evalB.qosViolationRate <= 1);
check.qUnchangedA    = results.evalA.qUnchangedAllRuns;
check.qUnchangedB    = results.evalB.qUnchangedAllRuns;

% State/action range checks via a diagnostic pass (independent of training/eval RNG)
stateRangeA_OK = true; stateRangeB_OK = true; actionRangeOK = true;
for ch = 1:8
    for qv = 0:3
        s = getStateModelA(ch, qv);
        if s < 1 || s > 32, stateRangeA_OK = false; end
    end
    for a = 0:1
      for b = 0:1
        for c = 0:1
            s = getStateModelB(ch, a, b, c);
            if s < 1 || s > 64, stateRangeB_OK = false; end
        end
      end
    end
end
[~, testActionA] = max(results.Q_A(1,:));
[~, testActionB] = max(results.Q_B(1,:));
actionRangeOK = (testActionA >= 1 && testActionA <= 8) && (testActionB >= 1 && testActionB <= 8);

check.stateRangeA_1to32 = stateRangeA_OK;
check.stateRangeB_1to64 = stateRangeB_OK;
check.actionsWithinRange = actionRangeOK;

fprintf('Model A Q-table is 32x8:                 %s\n', mat2str(check.modelA_QDims));
fprintf('Model B Q-table is 64x8:                 %s\n', mat2str(check.modelB_QDims));
fprintf('Model A state indices in 1..32:           %s\n', mat2str(check.stateRangeA_1to32));
fprintf('Model B state indices in 1..64:           %s\n', mat2str(check.stateRangeB_1to64));
fprintf('Actions within 1..8 (both models):        %s\n', mat2str(check.actionsWithinRange));
fprintf('Model A finite rewards:                    %s\n', mat2str(check.finiteRewardsA));
fprintf('Model B finite rewards:                    %s\n', mat2str(check.finiteRewardsB));
fprintf('Model A no NaN/Inf:                         %s\n', mat2str(check.noNaNInfA));
fprintf('Model B no NaN/Inf:                         %s\n', mat2str(check.noNaNInfB));
fprintf('Model A nonnegative data rates:             %s\n', mat2str(check.nonnegRatesA));
fprintf('Model B nonnegative data rates:             %s\n', mat2str(check.nonnegRatesB));
fprintf('Model A QoS violation rate in [0,1]:        %s\n', mat2str(check.qosRangeA));
fprintf('Model B QoS violation rate in [0,1]:        %s\n', mat2str(check.qosRangeB));
fprintf('Model A Q-table unchanged during eval:      %s\n', mat2str(check.qUnchangedA));
fprintf('Model B Q-table unchanged during eval:      %s\n', mat2str(check.qUnchangedB));

allChecksPassed = check.modelA_QDims && check.modelB_QDims && check.stateRangeA_1to32 && ...
    check.stateRangeB_1to64 && check.actionsWithinRange && check.finiteRewardsA && ...
    check.finiteRewardsB && check.noNaNInfA && check.noNaNInfB && check.nonnegRatesA && ...
    check.nonnegRatesB && check.qosRangeA && check.qosRangeB && check.qUnchangedA && check.qUnchangedB;

if ~allChecksPassed
    error('Stage 13 validation FAILED.');
end
disp('All Stage 13 validation checks PASSED.');

%% ============================================================
%% COMPARISON TABLE
%% ============================================================
mA = [mean(results.evalA.d2dThroughput), mean(results.evalA.cuThroughput), ...
      mean(results.evalA.qosViolationRate), mean(results.evalA.reward)];
sA = [std(results.evalA.d2dThroughput), std(results.evalA.cuThroughput), ...
      std(results.evalA.qosViolationRate), std(results.evalA.reward)];
mB = [mean(results.evalB.d2dThroughput), mean(results.evalB.cuThroughput), ...
      mean(results.evalB.qosViolationRate), mean(results.evalB.reward)];
sB = [std(results.evalB.d2dThroughput), std(results.evalB.cuThroughput), ...
      std(results.evalB.qosViolationRate), std(results.evalB.reward)];

metricNames = {'Average D2D Throughput (bits/s/Hz)', 'Average Cellular Throughput (bits/s/Hz)', ...
               'QoS Violation Rate', 'Average Reward'};

fprintf('\n===== COMPARISON TABLE (mean +/- std, %d eval runs) =====\n', numEvaluationRuns);
fprintf('%-42s | %-20s | %-20s\n', 'Metric', 'Model A (32 states)', 'Model B (64 states)');
fprintf('%s\n', repmat('-', 1, 90));
for m = 1:4
    fprintf('%-42s | %8.4f +/- %6.4f | %8.4f +/- %6.4f\n', metricNames{m}, mA(m), sA(m), mB(m), sB(m));
end

fprintf('\nFinal epsilon           - Model A: %.4f | Model B: %.4f\n', ...
    results.trainMetricsA.epsilonHistory(end), results.trainMetricsB.epsilonHistory(end));
fprintf('Q-table dimensions      - Model A: %s | Model B: %s\n', ...
    mat2str(size(results.Q_A)), mat2str(size(results.Q_B)));
fprintf('Visited states           - Model A: %d / %d | Model B: %d / %d\n', ...
    results.coverageA.visitedStates, results.coverageA.numStates, ...
    results.coverageB.visitedStates, results.coverageB.numStates);
fprintf('Visited state-action pairs - Model A: %d / %d | Model B: %d / %d\n', ...
    results.coverageA.visitedStateActionPairs, results.coverageA.numStates*8, ...
    results.coverageB.visitedStateActionPairs, results.coverageB.numStates*8);
fprintf('Fraction of state space explored - Model A: %.4f | Model B: %.4f\n', ...
    results.coverageA.fractionExplored, results.coverageB.fractionExplored);

%% ============================================================
%% PLOTS
%% ============================================================
movAvgWindow = 50;

% 1. Model A training reward
figure;
plot(movmean(results.trainMetricsA.episodeReward, movAvgWindow));
xlabel('Episode'); ylabel('Total Reward (moving avg)');
title('Model A Training Reward'); grid on;
saveas(gcf, 'stage13_plot1_modelA_reward.png');

% 2. Model B training reward
figure;
plot(movmean(results.trainMetricsB.episodeReward, movAvgWindow));
xlabel('Episode'); ylabel('Total Reward (moving avg)');
title('Model B Training Reward'); grid on;
saveas(gcf, 'stage13_plot2_modelB_reward.png');

% 3. D2D throughput comparison
figure;
bar([mA(1) mB(1)]); hold on;
errorbar(1:2, [mA(1) mB(1)], [sA(1) sB(1)], 'k.');
set(gca, 'XTickLabel', {'Model A', 'Model B'});
ylabel('bits/s/Hz'); title('Average D2D Throughput'); grid on; hold off;
saveas(gcf, 'stage13_plot3_d2d_throughput.png');

% 4. Cellular throughput comparison
figure;
bar([mA(2) mB(2)]); hold on;
errorbar(1:2, [mA(2) mB(2)], [sA(2) sB(2)], 'k.');
set(gca, 'XTickLabel', {'Model A', 'Model B'});
ylabel('bits/s/Hz'); title('Average Cellular Throughput'); grid on; hold off;
saveas(gcf, 'stage13_plot4_cellular_throughput.png');

% 5. QoS violation rate comparison
figure;
bar([mA(3) mB(3)]); hold on;
errorbar(1:2, [mA(3) mB(3)], [sA(3) sB(3)], 'k.');
set(gca, 'XTickLabel', {'Model A', 'Model B'});
ylabel('Violation Rate'); title('QoS Violation Rate'); grid on; hold off;
saveas(gcf, 'stage13_plot5_qos_violation_rate.png');

% 6. Average reward comparison
figure;
bar([mA(4) mB(4)]); hold on;
errorbar(1:2, [mA(4) mB(4)], [sA(4) sB(4)], 'k.');
set(gca, 'XTickLabel', {'Model A', 'Model B'});
ylabel('Reward'); title('Average Reward'); grid on; hold off;
saveas(gcf, 'stage13_plot6_avg_reward.png');

% 7. State visitation - Model A
figure;
bar(results.trainMetricsA.visitState);
xlabel('State Index (1-32)'); ylabel('Visit Count');
title('Model A State Visitation'); grid on;
saveas(gcf, 'stage13_plot7_modelA_visitation.png');

% 8. State visitation - Model B
figure;
bar(results.trainMetricsB.visitState);
xlabel('State Index (1-64)'); ylabel('Visit Count');
title('Model B State Visitation'); grid on;
saveas(gcf, 'stage13_plot8_modelB_visitation.png');

% 9. Q-table change vs episode (both models)
figure;
plot(results.trainMetricsA.qChangeHistory); hold on;
plot(results.trainMetricsB.qChangeHistory);
xlabel('Episode'); ylabel('||Q_{current} - Q_{previous}||');
title('Q-table Change vs Episode'); legend('Model A', 'Model B'); grid on; hold off;
saveas(gcf, 'stage13_plot9_qchange.png');

%% ============================================================
%% OBJECTIVE INTERPRETATION
%% ============================================================
fprintf('\n===== OBJECTIVE RESULT SUMMARY =====\n');
for m = 1:4
    fprintf('%s: Model A %.4f vs Model B %.4f (difference B-A: %.4f)\n', ...
        metricNames{m}, mA(m), mB(m), mB(m)-mA(m));
end
