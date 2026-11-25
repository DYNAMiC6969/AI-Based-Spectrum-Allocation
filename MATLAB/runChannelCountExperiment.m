function expResults = runChannelCountExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                   baseEnvParams, sweepHyperParams, ...
                                                   channelCounts, evaluationSteps, numEvaluationRuns, seed)
% RUNCHANNELCOUNTEXPERIMENT  Experiment 3: trains a SEPARATE Q-learning
% model (own state space, own Q-table, own action space) for each value
% in 'channelCounts', then evaluates each frozen policy.
%
% MODEL A state formula (selected in Stage 13): numStates = N * 4.
% Action space: A = 1:N. A Q-table trained for one N is NEVER reused for
% a different N (each training call below creates a fresh Q-table of
% the correct size via trainModelA_N.m).
%
% NOTE: cuChannel=[1,2,3] (an existing project assumption) requires
% N >= 3; all values in Stage 14's suggested channelCounts ({4,6,8,10})
% satisfy this. This function does not check N<3 since it is not a
% configuration this experiment uses.
%
% Inputs:
%   net, H_D2D, h_CU_BS, h_D2D_BS - environment (position/gain data;
%                                   these do NOT depend on channel count)
%   baseEnvParams     - envParams struct (txPowerD2D, P_CU, noisePower,
%                       W, cuChannel, R_min, referenceRate, lambdaQoS) -
%                       IDENTICAL for every channel count tested
%   sweepHyperParams  - hyperParams struct used for EVERY channel count
%                       (same alpha/gamma/epsilon schedule/episodes/steps)
%   channelCounts     - e.g. [4 6 8 10]
%   evaluationSteps, numEvaluationRuns, seed - evaluation configuration
%
% Output 'expResults': struct array, one element per channel count, with
% fields: numChannels, Q, trainMetrics, d2dThroughput, cuThroughput,
% qosViolationRate, reward (each eval field is [1 x numEvaluationRuns]).

    for k = 1:length(channelCounts)

        N = channelCounts(k);
        fprintf('\n[Experiment 3] Training Model A for numChannels = %d ...\n', N);

        rng(seed + N);   % distinct but reproducible seed per configuration
        [Q_N, trainMetrics_N] = trainModelA_N(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                baseEnvParams, sweepHyperParams, N);

        d2dT = zeros(1, numEvaluationRuns);
        cuT  = zeros(1, numEvaluationRuns);
        qosV = zeros(1, numEvaluationRuns);
        rew  = zeros(1, numEvaluationRuns);

        for run = 1:numEvaluationRuns
            seq = randi(N, 1, evaluationSteps);
            [d2dT(run), cuT(run), qosV(run), rew(run), ~] = evaluateModelA_N( ...
                Q_N, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, evaluationSteps, seq, N);
        end

        expResults(k).numChannels       = N;
        expResults(k).Q                 = Q_N;
        expResults(k).trainMetrics      = trainMetrics_N;
        expResults(k).d2dThroughput     = d2dT;
        expResults(k).cuThroughput      = cuT;
        expResults(k).qosViolationRate  = qosV;
        expResults(k).reward            = rew;

        fprintf('[Experiment 3] N=%d: D2D=%.4f, Cellular=%.4f, QoSViolRate=%.4f, Reward=%.4f (means)\n', ...
            N, mean(d2dT), mean(cuT), mean(qosV), mean(rew));

    end

    %% Plots
    N_list     = [expResults.numChannels];
    d2dMeans   = arrayfun(@(r) mean(r.d2dThroughput), expResults);
    d2dStds    = arrayfun(@(r) std(r.d2dThroughput), expResults);
    cuMeans    = arrayfun(@(r) mean(r.cuThroughput), expResults);
    cuStds     = arrayfun(@(r) std(r.cuThroughput), expResults);
    qosMeans   = arrayfun(@(r) mean(r.qosViolationRate), expResults);
    qosStds    = arrayfun(@(r) std(r.qosViolationRate), expResults);
    rewMeans   = arrayfun(@(r) mean(r.reward), expResults);
    rewStds    = arrayfun(@(r) std(r.reward), expResults);

    figure;
    errorbar(N_list, d2dMeans, d2dStds, '-o');
    xlabel('Number of Channels'); ylabel('Average D2D Throughput (bits/s/Hz)');
    title('Experiment 3: Number of Channels vs D2D Throughput'); grid on;
    saveas(gcf, 'stage14_exp3_d2d_throughput.png');

    figure;
    errorbar(N_list, cuMeans, cuStds, '-o');
    xlabel('Number of Channels'); ylabel('Average Cellular Throughput (bits/s/Hz)');
    title('Experiment 3: Number of Channels vs Cellular Throughput'); grid on;
    saveas(gcf, 'stage14_exp3_cellular_throughput.png');

    figure;
    errorbar(N_list, qosMeans, qosStds, '-o');
    xlabel('Number of Channels'); ylabel('QoS Violation Rate');
    title('Experiment 3: Number of Channels vs QoS Violation Rate'); grid on;
    saveas(gcf, 'stage14_exp3_qos_violation.png');

    figure;
    errorbar(N_list, rewMeans, rewStds, '-o');
    xlabel('Number of Channels'); ylabel('Average Reward');
    title('Experiment 3: Number of Channels vs Reward'); grid on;
    saveas(gcf, 'stage14_exp3_reward.png');

end
