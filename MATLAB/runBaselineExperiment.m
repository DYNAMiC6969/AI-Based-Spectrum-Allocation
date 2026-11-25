function results = runBaselineExperiment(Q, H_D2D, h_CU_BS, h_D2D_BS, envParams, ...
                                           evaluationSteps, numEvaluationRuns, seed)
% RUNBASELINEEXPERIMENT  Experiment 2: compares Policy A (random channel
% allocation) against Policy B (the frozen, trained Q-learning policy
% from Experiment 1) at the 8-channel baseline configuration.
%
% Reuses Stage 12's evaluateRandomPolicy.m and evaluateQLearningPolicy.m
% UNMODIFIED (both already implement the 32-state / 8-channel baseline
% exactly). The Q-table passed in is NEVER updated here (verified via
% each function's QUnchanged output, matching Stage 12's methodology).
%
% Inputs:
%   Q                 - [32x8] frozen, trained Q-table (from Experiment 1)
%   H_D2D, h_CU_BS, h_D2D_BS - channel-gain matrices
%   envParams         - environment parameter struct
%   evaluationSteps   - steps per evaluation run
%   numEvaluationRuns - number of independent evaluation runs
%   seed              - RNG seed for reproducibility
%
% Output 'results' struct: random.*, qlearn.* (each a
% [1 x numEvaluationRuns] array for d2dThroughput, cuThroughput,
% qosViolationRate, reward), plus qUnchangedAllRuns.

    rng(seed);

    results.random.d2dThroughput    = zeros(1, numEvaluationRuns);
    results.random.cuThroughput     = zeros(1, numEvaluationRuns);
    results.random.qosViolationRate = zeros(1, numEvaluationRuns);
    results.random.reward           = zeros(1, numEvaluationRuns);

    results.qlearn.d2dThroughput    = zeros(1, numEvaluationRuns);
    results.qlearn.cuThroughput     = zeros(1, numEvaluationRuns);
    results.qlearn.qosViolationRate = zeros(1, numEvaluationRuns);
    results.qlearn.reward           = zeros(1, numEvaluationRuns);

    qUnchangedAllRuns = true;

    for run = 1:numEvaluationRuns

        otherD2DChannelSequence = randi(8, 1, evaluationSteps);

        [rD2D, rCU, rQoS, rRew] = evaluateRandomPolicy( ...
            H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, otherD2DChannelSequence);

        [qD2D, qCU, qQoS, qRew, qUnchanged] = evaluateQLearningPolicy( ...
            Q, H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, otherD2DChannelSequence);

        results.random.d2dThroughput(run)    = rD2D;
        results.random.cuThroughput(run)     = rCU;
        results.random.qosViolationRate(run) = rQoS;
        results.random.reward(run)           = rRew;

        results.qlearn.d2dThroughput(run)    = qD2D;
        results.qlearn.cuThroughput(run)     = qCU;
        results.qlearn.qosViolationRate(run) = qQoS;
        results.qlearn.reward(run)           = qRew;

        qUnchangedAllRuns = qUnchangedAllRuns && qUnchanged;

    end

    results.qUnchangedAllRuns = qUnchangedAllRuns;
    results.seed = seed;
    results.evaluationSteps = evaluationSteps;
    results.numEvaluationRuns = numEvaluationRuns;

    % Comparison table (console) and plots
    rMean = [mean(results.random.d2dThroughput), mean(results.random.cuThroughput), ...
             mean(results.random.qosViolationRate), mean(results.random.reward)];
    rStd  = [std(results.random.d2dThroughput), std(results.random.cuThroughput), ...
             std(results.random.qosViolationRate), std(results.random.reward)];
    qMean = [mean(results.qlearn.d2dThroughput), mean(results.qlearn.cuThroughput), ...
             mean(results.qlearn.qosViolationRate), mean(results.qlearn.reward)];
    qStd  = [std(results.qlearn.d2dThroughput), std(results.qlearn.cuThroughput), ...
             std(results.qlearn.qosViolationRate), std(results.qlearn.reward)];

    metricNames = {'Avg D2D Throughput (bits/s/Hz)', 'Avg Cellular Throughput (bits/s/Hz)', ...
                   'QoS Violation Rate', 'Average Reward'};

    fprintf('\n[Experiment 2] Comparison table (mean +/- std, %d runs):\n', numEvaluationRuns);
    fprintf('%-36s | %-18s | %-18s\n', 'Metric', 'Random', 'Q-Learning');
    for m = 1:4
        fprintf('%-36s | %7.4f +/- %6.4f | %7.4f +/- %6.4f\n', ...
            metricNames{m}, rMean(m), rStd(m), qMean(m), qStd(m));
    end
    fprintf('Q-table unchanged after all evaluation runs: %s\n', mat2str(results.qUnchangedAllRuns));

    plotTitles  = {'Average D2D Throughput', 'Average Cellular Throughput', 'QoS Violation Rate', 'Average Reward'};
    plotYLabels = {'bits/s/Hz', 'bits/s/Hz', 'Violation Rate', 'Reward'};
    plotFiles   = {'stage14_exp2_d2d_throughput.png', 'stage14_exp2_cellular_throughput.png', ...
                   'stage14_exp2_qos_violation.png', 'stage14_exp2_reward.png'};

    for m = 1:4
        figure;
        bar([rMean(m) qMean(m)]); hold on;
        errorbar(1:2, [rMean(m) qMean(m)], [rStd(m) qStd(m)], 'k.');
        set(gca, 'XTickLabel', {'Random', 'Q-Learning'});
        ylabel(plotYLabels{m});
        title(['Experiment 2: ' plotTitles{m} ' (Random vs Q-Learning)']);
        grid on; hold off;
        saveas(gcf, plotFiles{m});
    end

end
