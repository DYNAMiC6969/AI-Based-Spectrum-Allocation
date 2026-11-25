function expResults = runQoSSensitivityExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                     baseEnvParams, sweepHyperParams, ...
                                                     rMinValues, evaluationSteps, numEvaluationRuns, seed)
% RUNQOSSENSITIVITYEXPERIMENT  Experiment 4: investigates the effect of
% the cellular minimum-rate QoS threshold on D2D/cellular performance.
%
% rMinValues is a PROJECT SIMULATION ASSUMPTION (a configurable set of
% threshold values to sweep) - NOT a thesis-derived set of numbers. The
% thesis defines the QoS constraint conceptually (Constraint C1) without
% fixing universal numeric values for this specific simulation.
%
% Fixed at the Stage 11/12 baseline: numChannels=8 (Model A, 32 states).
% Each threshold gets its OWN freshly trained Q-table, since changing
% R_min changes which allocations count as QoS violations and therefore
% changes the reward landscape itself - evaluation-only reuse of one
% Q-table across different R_min values would not reflect what the
% agent would actually learn under each threshold.
%
% Inputs:
%   net, H_D2D, h_CU_BS, h_D2D_BS - environment (unchanged across sweep)
%   baseEnvParams     - envParams struct; R_min is OVERWRITTEN per
%                       iteration, every other field held fixed
%   sweepHyperParams  - hyperParams struct, IDENTICAL for every threshold
%   rMinValues        - e.g. [3 5 7] (bits/s/Hz), applied uniformly to
%                       all 3 CUs for each configuration
%   evaluationSteps, numEvaluationRuns, seed - evaluation configuration
%
% Output 'expResults': struct array, one element per R_min value.

    numChannels = 8;   % fixed baseline channel count for this experiment

    for k = 1:length(rMinValues)

        rMin = rMinValues(k);
        fprintf('\n[Experiment 4] Training Model A for R_min = %.1f (all CUs) ...\n', rMin);

        envParams_k = baseEnvParams;
        envParams_k.R_min = [rMin rMin rMin];

        rng(seed + round(rMin * 100));
        [Q_k, trainMetrics_k] = trainModelA_N(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                envParams_k, sweepHyperParams, numChannels);

        d2dT = zeros(1, numEvaluationRuns);
        cuT  = zeros(1, numEvaluationRuns);
        qosV = zeros(1, numEvaluationRuns);
        rew  = zeros(1, numEvaluationRuns);

        for run = 1:numEvaluationRuns
            seq = randi(numChannels, 1, evaluationSteps);
            [d2dT(run), cuT(run), qosV(run), rew(run), ~] = evaluateModelA_N( ...
                Q_k, H_D2D, h_CU_BS, h_D2D_BS, envParams_k, evaluationSteps, seq, numChannels);
        end

        expResults(k).R_min             = rMin;
        expResults(k).Q                 = Q_k;
        expResults(k).trainMetrics      = trainMetrics_k;
        expResults(k).d2dThroughput     = d2dT;
        expResults(k).cuThroughput      = cuT;
        expResults(k).qosViolationRate  = qosV;
        expResults(k).reward            = rew;

        fprintf('[Experiment 4] R_min=%.1f: D2D=%.4f, Cellular=%.4f, QoSViolRate=%.4f, Reward=%.4f (means)\n', ...
            rMin, mean(d2dT), mean(cuT), mean(qosV), mean(rew));

    end

    %% Plots
    rMinList  = [expResults.R_min];
    d2dMeans  = arrayfun(@(r) mean(r.d2dThroughput), expResults);
    d2dStds   = arrayfun(@(r) std(r.d2dThroughput), expResults);
    qosMeans  = arrayfun(@(r) mean(r.qosViolationRate), expResults);
    qosStds   = arrayfun(@(r) std(r.qosViolationRate), expResults);

    figure;
    errorbar(rMinList, d2dMeans, d2dStds, '-o');
    xlabel('Cellular Minimum Rate Threshold R_{min} (bits/s/Hz)');
    ylabel('Average D2D Throughput (bits/s/Hz)');
    title('Experiment 4: QoS Threshold vs D2D Throughput'); grid on;
    saveas(gcf, 'stage14_exp4_d2d_throughput.png');

    figure;
    errorbar(rMinList, qosMeans, qosStds, '-o');
    xlabel('Cellular Minimum Rate Threshold R_{min} (bits/s/Hz)');
    ylabel('QoS Violation Rate');
    title('Experiment 4: QoS Threshold vs QoS Violation Rate'); grid on;
    saveas(gcf, 'stage14_exp4_qos_violation.png');

end
