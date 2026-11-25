function resultsTable = createResultsTable(exp2Results, exp3Results, exp4Results, exp5Results)
% CREATERESULTSTABLE  Consolidates all Stage 14 experiment results into
% one final table with columns: Configuration, Policy, Average D2D
% Throughput, Average Cellular Throughput, QoS Violation Rate, Average
% Reward. Every value is read directly from the experiment result
% structs (themselves produced by actual MATLAB/Octave execution) - none
% are manually entered.
%
% Uses MATLAB's table() function when available. Octave (used to verify
% this code) does not implement table() as of this writing, so a
% compatible cell-array fallback is used there, printed in the same
% row/column layout - clearly marked as a compatibility fallback, not a
% different set of results.

    Configuration = {};
    Policy        = {};
    D2DThroughput = [];
    CUThroughput  = [];
    QoSViolRate   = [];
    Reward        = [];

    % --- Experiment 2: baseline Random vs Q-Learning (8 channels) ---
    Configuration{end+1} = '8 channels (baseline)'; Policy{end+1} = 'Random';
    D2DThroughput(end+1) = mean(exp2Results.random.d2dThroughput);
    CUThroughput(end+1)  = mean(exp2Results.random.cuThroughput);
    QoSViolRate(end+1)   = mean(exp2Results.random.qosViolationRate);
    Reward(end+1)        = mean(exp2Results.random.reward);

    Configuration{end+1} = '8 channels (baseline)'; Policy{end+1} = 'Q-Learning';
    D2DThroughput(end+1) = mean(exp2Results.qlearn.d2dThroughput);
    CUThroughput(end+1)  = mean(exp2Results.qlearn.cuThroughput);
    QoSViolRate(end+1)   = mean(exp2Results.qlearn.qosViolationRate);
    Reward(end+1)        = mean(exp2Results.qlearn.reward);

    % --- Experiment 3: channel-count sweep (Q-Learning only) ---
    for k = 1:length(exp3Results)
        Configuration{end+1} = sprintf('%d channels', exp3Results(k).numChannels);
        Policy{end+1}        = 'Q-Learning';
        D2DThroughput(end+1) = mean(exp3Results(k).d2dThroughput);
        CUThroughput(end+1)  = mean(exp3Results(k).cuThroughput);
        QoSViolRate(end+1)   = mean(exp3Results(k).qosViolationRate);
        Reward(end+1)        = mean(exp3Results(k).reward);
    end

    % --- Experiment 4: QoS threshold sweep (Q-Learning only, 8 channels) ---
    for k = 1:length(exp4Results)
        Configuration{end+1} = sprintf('R_min=%.1f (8 ch)', exp4Results(k).R_min);
        Policy{end+1}        = 'Q-Learning';
        D2DThroughput(end+1) = mean(exp4Results(k).d2dThroughput);
        CUThroughput(end+1)  = mean(exp4Results(k).cuThroughput);
        QoSViolRate(end+1)   = mean(exp4Results(k).qosViolationRate);
        Reward(end+1)        = mean(exp4Results(k).reward);
    end

    % --- Experiment 5: hyperparameter (alpha) sweep (Q-Learning only, 8 channels) ---
    if ~isempty(exp5Results)
        for k = 1:length(exp5Results)
            Configuration{end+1} = sprintf('alpha=%.3f (8 ch)', exp5Results(k).alpha);
            Policy{end+1}        = 'Q-Learning';
            D2DThroughput(end+1) = mean(exp5Results(k).d2dThroughput);
            CUThroughput(end+1)  = mean(exp5Results(k).cuThroughput);
            QoSViolRate(end+1)   = mean(exp5Results(k).qosViolationRate);
            Reward(end+1)        = mean(exp5Results(k).reward);
        end
    end

    Configuration = Configuration(:);
    Policy        = Policy(:);
    D2DThroughput = D2DThroughput(:);
    CUThroughput  = CUThroughput(:);
    QoSViolRate   = QoSViolRate(:);
    Reward        = Reward(:);

    try
        resultsTable = table(Configuration, Policy, D2DThroughput, CUThroughput, QoSViolRate, Reward, ...
            'VariableNames', {'Configuration','Policy','AvgD2DThroughput','AvgCellularThroughput','QoSViolationRate','AvgReward'});
        disp(resultsTable);
    catch
        % Octave fallback (table() not implemented) - same data, printed as a formatted table.
        resultsTable = struct('Configuration', {Configuration}, 'Policy', {Policy}, ...
            'AvgD2DThroughput', D2DThroughput, 'AvgCellularThroughput', CUThroughput, ...
            'QoSViolationRate', QoSViolRate, 'AvgReward', Reward);
        fprintf('\n[NOTE: table() unavailable in this environment (Octave) - printing equivalent data]\n');
        fprintf('%-24s | %-11s | %9s | %9s | %9s | %9s\n', ...
            'Configuration', 'Policy', 'D2D Tput', 'CU Tput', 'QoSViol', 'Reward');
        fprintf('%s\n', repmat('-', 1, 90));
        for i = 1:length(Configuration)
            fprintf('%-24s | %-11s | %9.4f | %9.4f | %9.4f | %9.4f\n', ...
                Configuration{i}, Policy{i}, D2DThroughput(i), CUThroughput(i), QoSViolRate(i), Reward(i));
        end
    end

end
