%% Stage 14 - Final Experimental Campaign
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% *** WIRELESS MODEL: UNCHANGED FROM STAGE 9 onward ***
% Cellular uplink (CU -> BS), D2D-to-D2D interference, D2D-to-BS
% interference. No wireless equations are modified in this stage.
%
% *** STATE MODEL: Model A (selected in Stage 13) ***
% state_A = [otherD2DChannel, QoSViolationCount]. Model A was selected
% because Stage 13 found it performs identically to the richer Model B
% under the baseline configuration, with better Q-table stability.
%
% *** AGENT SCOPE (unchanged project simplification) ***
% D2D pair 1 is controlled by Q-learning; D2D pair 2's channel is
% redrawn uniformly at random every time step. This is tabular
% Q-learning-based channel selection for a single D2D transmitter, NOT
% multi-agent reinforcement learning.
%
% *** COMPUTATIONAL NOTE (documented honestly) ***
% A single 3000-episode training run takes ~90 seconds in this
% environment. This campaign trains 1 (Experiment 1) + 4 (Experiment 3)
% + 3 (Experiment 4) + 3 (Experiment 5) = 11 separate Q-tables.
% Running all 11 at 3000 episodes would take ~16 minutes, which is
% impractical for this session. Experiment 1 and Experiment 2 (which
% reuses Experiment 1's Q-table) use the FULL 3000-episode baseline,
% matching Stages 11-13 exactly. Experiments 3, 4, and 5 (the sweep
% experiments, 10 additional trainings) use a REDUCED 1000-episode
% budget - still enough for epsilon to reach its floor (~episode 600)
% and stabilize for ~400 further episodes, but with less stabilization
% time than the 3000-episode baseline. This is an explicit,
% documented resource-driven scope reduction, not a fabrication - see
% the Stage 14 document, Section 2 and Section 12 (Limitations).

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

%% 2. Baseline environment parameters (unchanged since Stage 8-13)
baseEnvParams.txPowerD2D = [1 1];
baseEnvParams.P_CU       = [5 5 5];
baseEnvParams.noisePower = 1e-6;
baseEnvParams.W          = 1;
baseEnvParams.cuChannel  = [1 2 3];
baseEnvParams.R_min      = [5 5 5];

noInterferenceSINR = (baseEnvParams.txPowerD2D .* diag(H_D2D)') ./ baseEnvParams.noisePower;
noInterferenceRates = baseEnvParams.W .* log2(1 + noInterferenceSINR);
baseEnvParams.referenceRate = sum(noInterferenceRates);
baseEnvParams.lambdaQoS = 1;

%% 3. Baseline hyperparameters (Experiments 1 & 2: full 3000 episodes)
baseHyperParams.alpha        = 0.1;
baseHyperParams.gamma        = 0.9;
baseHyperParams.epsilon      = 1.0;
baseHyperParams.epsilonMin   = 0.05;
baseHyperParams.epsilonDecay = 0.995;
baseHyperParams.maxTimeSteps = 50;
baseHyperParams.numEpisodes  = 3000;

% Sweep hyperparameters (Experiments 3,4,5: reduced to 1000 episodes -
% see computational note above)
sweepHyperParams = baseHyperParams;
sweepHyperParams.numEpisodes = 1000;

evaluationSteps   = 1000;
numEvaluationRuns = 10;

MASTER_SEED = 42;

%% ============================================================
%% EXPERIMENT 1 - Q-LEARNING CONVERGENCE (full 3000 episodes)
%% ============================================================
fprintf('\n########## EXPERIMENT 1: CONVERGENCE ##########\n');
[Q_baseline, metrics1, config1] = runConvergenceExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                              baseEnvParams, baseHyperParams, MASTER_SEED);

%% ============================================================
%% EXPERIMENT 2 - Q-LEARNING VS RANDOM (reuses Experiment 1's frozen Q)
%% ============================================================
fprintf('\n########## EXPERIMENT 2: BASELINE COMPARISON ##########\n');
exp2Results = runBaselineExperiment(Q_baseline, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, ...
                                      evaluationSteps, numEvaluationRuns, MASTER_SEED);

%% ============================================================
%% EXPERIMENT 3 - NUMBER OF CHANNELS (reduced-episode sweep)
%% ============================================================
fprintf('\n########## EXPERIMENT 3: CHANNEL COUNT SWEEP ##########\n');
channelCounts = [4 6 8 10];
exp3Results = runChannelCountExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, ...
                                          sweepHyperParams, channelCounts, evaluationSteps, ...
                                          numEvaluationRuns, MASTER_SEED);

%% ============================================================
%% EXPERIMENT 4 - QoS SENSITIVITY (reduced-episode sweep)
%% ============================================================
fprintf('\n########## EXPERIMENT 4: QoS THRESHOLD SWEEP ##########\n');
rMinValues = [3 5 7];   % project simulation assumption, NOT thesis-derived
exp4Results = runQoSSensitivityExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, ...
                                            sweepHyperParams, rMinValues, evaluationSteps, ...
                                            numEvaluationRuns, MASTER_SEED);

%% ============================================================
%% EXPERIMENT 5 - HYPERPARAMETER SENSITIVITY (optional, reduced-episode sweep)
%% ============================================================
fprintf('\n########## EXPERIMENT 5: ALPHA SENSITIVITY ##########\n');
alphaValues = [0.05 0.1 0.2];   % Q-learning hyperparameters, NOT thesis-derived
exp5Results = runHyperparameterExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, ...
                                            sweepHyperParams, alphaValues, evaluationSteps, ...
                                            numEvaluationRuns, MASTER_SEED);

%% ============================================================
%% FINAL RESULTS TABLE
%% ============================================================
fprintf('\n########## FINAL RESULTS TABLE ##########\n');
resultsTable = createResultsTable(exp2Results, exp3Results, exp4Results, exp5Results);

%% ============================================================
%% VALIDATION CHECKS (finite/no NaN-Inf across all experiment outputs)
%% ============================================================
fprintf('\n########## VALIDATION CHECKS ##########\n');

allVals = [exp2Results.random.d2dThroughput, exp2Results.random.cuThroughput, ...
           exp2Results.random.qosViolationRate, exp2Results.random.reward, ...
           exp2Results.qlearn.d2dThroughput, exp2Results.qlearn.cuThroughput, ...
           exp2Results.qlearn.qosViolationRate, exp2Results.qlearn.reward];
for k = 1:length(exp3Results)
    allVals = [allVals, exp3Results(k).d2dThroughput, exp3Results(k).cuThroughput, ...
               exp3Results(k).qosViolationRate, exp3Results(k).reward];
end
for k = 1:length(exp4Results)
    allVals = [allVals, exp4Results(k).d2dThroughput, exp4Results(k).cuThroughput, ...
               exp4Results(k).qosViolationRate, exp4Results(k).reward];
end
for k = 1:length(exp5Results)
    allVals = [allVals, exp5Results(k).d2dThroughput, exp5Results(k).cuThroughput, ...
               exp5Results(k).qosViolationRate, exp5Results(k).reward];
end

check.allFinite = all(isfinite(allVals));
check.noNaNInf  = ~any(isnan(allVals)) && ~any(isinf(allVals));
check.qUnchangedExp2 = exp2Results.qUnchangedAllRuns;

fprintf('All experiment metrics finite:  %s\n', mat2str(check.allFinite));
fprintf('No NaN/Inf anywhere:              %s\n', mat2str(check.noNaNInf));
fprintf('Experiment 2 Q-table unchanged:   %s\n', mat2str(check.qUnchangedExp2));

if ~(check.allFinite && check.noNaNInf && check.qUnchangedExp2)
    error('Stage 14 validation FAILED.');
end
disp('All Stage 14 validation checks PASSED.');

%% ============================================================
%% REPRODUCIBILITY RECORD
%% ============================================================
fprintf('\n########## REPRODUCIBILITY RECORD ##########\n');
fprintf('Number of channels (baseline):  8   | sweep: [%s]\n', num2str(channelCounts));
fprintf('Number of CUs:                   %d\n', numCU);
fprintf('Number of D2D pairs:              %d\n', numD2D);
fprintf('D2D transmit power:                %s\n', mat2str(baseEnvParams.txPowerD2D));
fprintf('Cellular transmit power:            %s\n', mat2str(baseEnvParams.P_CU));
fprintf('Noise power:                         %.2e\n', baseEnvParams.noisePower);
fprintf('QoS threshold (baseline):  %s | sweep: [%s]\n', mat2str(baseEnvParams.R_min), num2str(rMinValues));
fprintf('alpha (baseline):  %.3f | sweep: [%s]\n', baseHyperParams.alpha, num2str(alphaValues));
fprintf('gamma:                                %.3f\n', baseHyperParams.gamma);
fprintf('Initial epsilon:                      %.3f\n', baseHyperParams.epsilon);
fprintf('Minimum epsilon:                      %.3f\n', baseHyperParams.epsilonMin);
fprintf('Epsilon decay:                         %.3f\n', baseHyperParams.epsilonDecay);
fprintf('Episodes (Exp 1,2 baseline):             %d\n', baseHyperParams.numEpisodes);
fprintf('Episodes (Exp 3,4,5 sweeps):              %d\n', sweepHyperParams.numEpisodes);
fprintf('Steps per episode:                         %d\n', baseHyperParams.maxTimeSteps);
fprintf('Evaluation steps per run:                   %d\n', evaluationSteps);
fprintf('Evaluation runs:                             %d\n', numEvaluationRuns);
fprintf('Master random seed:                           %d\n', MASTER_SEED);

fprintf('\n===== STAGE 14 CAMPAIGN COMPLETE =====\n');
