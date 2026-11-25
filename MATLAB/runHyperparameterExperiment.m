function expResults = runHyperparameterExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                     baseEnvParams, baseHyperParams, ...
                                                     alphaValues, evaluationSteps, numEvaluationRuns, seed)
% RUNHYPERPARAMETEREXPERIMENT  Experiment 5 (optional): small sensitivity
% sweep over the Q-learning LEARNING RATE (alpha). gamma and all other
% hyperparameters are held fixed at the Stage 11/12 baseline. This is a
% small demonstration sweep, NOT a hyperparameter search - per the
% instruction not to perform a large search.
%
% alphaValues are Q-LEARNING HYPERPARAMETERS - NOT thesis-derived values.
%
% Fixed at the Stage 11/12 baseline: numChannels=8, R_min=[5,5,5].
%
% Inputs:
%   net, H_D2D, h_CU_BS, h_D2D_BS - environment (unchanged across sweep)
%   baseEnvParams     - envParams struct, IDENTICAL for every alpha
%   baseHyperParams   - hyperParams struct; alpha is OVERWRITTEN per
%                       iteration, gamma/epsilon schedule/episodes/steps
%                       held fixed
%   alphaValues       - e.g. [0.05 0.1 0.2]
%   evaluationSteps, numEvaluationRuns, seed - evaluation configuration
%
% Output 'expResults': struct array, one element per alpha value, with
% fields: alpha, Q, trainMetrics, finalMeanReward100 (mean episode
% reward over the last 100 training episodes - a convergence
% indicator), d2dThroughput, cuThroughput, qosViolationRate, reward.

    numChannels = 8;

    for k = 1:length(alphaValues)

        a = alphaValues(k);
        fprintf('\n[Experiment 5] Training Model A for alpha = %.3f ...\n', a);

        hyperParams_k = baseHyperParams;
        hyperParams_k.alpha = a;

        rng(seed + round(a * 1000));
        [Q_k, trainMetrics_k] = trainModelA_N(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                baseEnvParams, hyperParams_k, numChannels);

        finalMeanReward100 = mean(trainMetrics_k.episodeReward(end-99:end));

        d2dT = zeros(1, numEvaluationRuns);
        cuT  = zeros(1, numEvaluationRuns);
        qosV = zeros(1, numEvaluationRuns);
        rew  = zeros(1, numEvaluationRuns);

        for run = 1:numEvaluationRuns
            seq = randi(numChannels, 1, evaluationSteps);
            [d2dT(run), cuT(run), qosV(run), rew(run), ~] = evaluateModelA_N( ...
                Q_k, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, evaluationSteps, seq, numChannels);
        end

        expResults(k).alpha               = a;
        expResults(k).Q                   = Q_k;
        expResults(k).trainMetrics        = trainMetrics_k;
        expResults(k).finalMeanReward100  = finalMeanReward100;
        expResults(k).d2dThroughput       = d2dT;
        expResults(k).cuThroughput        = cuT;
        expResults(k).qosViolationRate    = qosV;
        expResults(k).reward              = rew;

        fprintf('[Experiment 5] alpha=%.3f: TrainMeanReward(last100)=%.4f, EvalD2D=%.4f, EvalQoSViolRate=%.4f, EvalReward=%.4f\n', ...
            a, finalMeanReward100, mean(d2dT), mean(qosV), mean(rew));

    end

    %% Plot: training reward curves for each alpha (convergence behavior)
    figure; hold on;
    movAvgWindow = 50;
    legendEntries = cell(1, length(alphaValues));
    for k = 1:length(alphaValues)
        plot(movmean(expResults(k).trainMetrics.episodeReward, movAvgWindow), 'LineWidth', 1.1);
        legendEntries{k} = sprintf('alpha = %.3f', expResults(k).alpha);
    end
    xlabel('Episode'); ylabel('Total Episode Reward (50-episode moving avg)');
    title('Experiment 5: Training Reward vs Episode for Different alpha');
    legend(legendEntries, 'Location', 'southeast');
    grid on; hold off;
    saveas(gcf, 'stage14_exp5_convergence.png');

    %% Plot: final evaluation reward vs alpha
    aList    = [expResults.alpha];
    rewMeans = arrayfun(@(r) mean(r.reward), expResults);
    rewStds  = arrayfun(@(r) std(r.reward), expResults);

    figure;
    errorbar(aList, rewMeans, rewStds, '-o');
    xlabel('Learning Rate alpha'); ylabel('Average Reward (evaluation)');
    title('Experiment 5: Learning Rate alpha vs Evaluation Reward'); grid on;
    saveas(gcf, 'stage14_exp5_alpha_vs_reward.png');

end
