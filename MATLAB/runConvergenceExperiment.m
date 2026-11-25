function [Q, metrics, config] = runConvergenceExperiment(net, H_D2D, h_CU_BS, h_D2D_BS, ...
                                                            baseEnvParams, baseHyperParams, seed)
% RUNCONVERGENCEEXPERIMENT  Experiment 1: trains the Model A Q-learning
% agent at the baseline configuration (8 channels) and produces the 5
% required convergence plots. Uses trainModelA_N.m with numChannels=8,
% which is mathematically identical to Stage 11/13's trainModelA.m.

    numChannels = 8;
    rng(seed);

    [Q, metrics] = trainModelA_N(net, H_D2D, h_CU_BS, h_D2D_BS, baseEnvParams, baseHyperParams, numChannels);

    config.numChannels    = numChannels;
    config.numCU          = length(baseEnvParams.cuChannel);
    config.numD2D         = 2;
    config.txPowerD2D     = baseEnvParams.txPowerD2D;
    config.P_CU           = baseEnvParams.P_CU;
    config.noisePower     = baseEnvParams.noisePower;
    config.R_min          = baseEnvParams.R_min;
    config.alpha          = baseHyperParams.alpha;
    config.gamma          = baseHyperParams.gamma;
    config.epsilonStart   = baseHyperParams.epsilon;
    config.epsilonMin     = baseHyperParams.epsilonMin;
    config.epsilonDecay   = baseHyperParams.epsilonDecay;
    config.numEpisodes    = baseHyperParams.numEpisodes;
    config.maxTimeSteps   = baseHyperParams.maxTimeSteps;
    config.seed           = seed;

    movAvgWindow = 50;

    figure;
    plot(movmean(metrics.episodeReward, movAvgWindow), 'LineWidth', 1.2);
    xlabel('Episode (50-episode moving average)'); ylabel('Episode Return (sum over 50 steps)');
    title('Experiment 1: Episode vs Reward');
    grid on;
    saveas(gcf, 'stage14_exp1_reward.png');

    figure;
    plot(movmean(metrics.averageD2DThroughput, movAvgWindow), 'LineWidth', 1.2);
    xlabel('Episode (50-episode moving average)'); ylabel('Aggregate D2D Throughput (bits/s/Hz)');
    title('Experiment 1: Episode vs Aggregate D2D Throughput');
    grid on;
    saveas(gcf, 'stage14_exp1_d2d_throughput.png');

    figure;
    plot(movmean(metrics.averageCellularRate, movAvgWindow), 'LineWidth', 1.2);
    xlabel('Episode (50-episode moving average)'); ylabel('Avg Cellular Throughput (bits/s/Hz)');
    title('Experiment 1: Episode vs Cellular Throughput');
    grid on;
    saveas(gcf, 'stage14_exp1_cellular_throughput.png');

    figure;
    plot(movmean(metrics.QoSViolationRate, movAvgWindow), 'LineWidth', 1.2);
    xlabel('Episode (50-episode moving average)'); ylabel('QoS Violation Rate (per CU per step)');
    title('Experiment 1: Episode vs QoS Violation Rate');
    grid on;
    saveas(gcf, 'stage14_exp1_qos_violation.png');

    figure;
    plot(metrics.epsilonHistory, 'LineWidth', 1.2);
    xlabel('Episode'); ylabel('Epsilon');
    title('Experiment 1: Episode vs Epsilon');
    grid on;
    saveas(gcf, 'stage14_exp1_epsilon.png');

    fprintf('[Experiment 1] Training complete. Final epsilon = %.4f, Mean reward (last 100 ep) = %.4f\n', ...
        metrics.epsilonHistory(end), mean(metrics.episodeReward(end-99:end)));

end
