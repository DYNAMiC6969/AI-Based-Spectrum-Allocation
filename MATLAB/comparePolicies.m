function results = comparePolicies(Q, H_D2D, h_CU_BS, h_D2D_BS, envParams, ...
                                     evaluationSteps, numEvaluationRuns)
% COMPAREPOLICIES  Runs 'numEvaluationRuns' independent evaluation runs,
% each comparing the Random policy and the frozen Q-learning policy
% under IDENTICAL environment conditions (same otherD2DChannel sequence
% within each run), and returns per-run results plus mean/std summaries.
%
% Inputs:
%   Q                - [32x8] frozen, trained Q-table (Stage 11 output)
%   H_D2D, h_CU_BS, h_D2D_BS - channel-gain matrices
%   envParams        - environment parameter struct (see Stage 10/11)
%   evaluationSteps  - time steps per evaluation run (e.g. 1000)
%   numEvaluationRuns - number of independent runs (e.g. 10)
%
% Output 'results' struct fields (each [1 x numEvaluationRuns] unless noted):
%   random.d2dThroughput, random.cuThroughput, random.qosViolationRate, random.reward
%   qlearn.d2dThroughput, qlearn.cuThroughput, qlearn.qosViolationRate, qlearn.reward
%   qUnchangedAllRuns - true only if Q was verified unchanged in EVERY run

    results.random.d2dThroughput  = zeros(1, numEvaluationRuns);
    results.random.cuThroughput   = zeros(1, numEvaluationRuns);
    results.random.qosViolationRate = zeros(1, numEvaluationRuns);
    results.random.reward         = zeros(1, numEvaluationRuns);

    results.qlearn.d2dThroughput  = zeros(1, numEvaluationRuns);
    results.qlearn.cuThroughput   = zeros(1, numEvaluationRuns);
    results.qlearn.qosViolationRate = zeros(1, numEvaluationRuns);
    results.qlearn.reward         = zeros(1, numEvaluationRuns);

    qUnchangedAllRuns = true;

    for run = 1:numEvaluationRuns

        % Fresh, shared "other D2D pair" channel sequence for this run -
        % used identically by BOTH policies (fair-comparison requirement)
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

end
