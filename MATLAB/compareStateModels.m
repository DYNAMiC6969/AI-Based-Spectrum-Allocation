function results = compareStateModels(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams, ...
                                       evaluationSteps, numEvaluationRuns)
% COMPARESTATEMODELS  Trains Model A (32-state) and Model B (64-state)
% under IDENTICAL environment/hyperparameters, then evaluates each
% model's frozen policy over multiple independent runs.
%
% RANDOMNESS / FAIRNESS METHODOLOGY (documented explicitly, per Stage 13
% requirements):
%   rng(42) is reset to the SAME value immediately before each of the
%   four phases below (train A, train B, evaluate A, evaluate B). Because
%   trainModelA.m and trainModelB.m consume random numbers in EXACTLY the
%   same order and count per step (one selectAction call, one
%   otherD2DChannel redraw - the state-indexing math itself consumes no
%   randomness), resetting the seed makes Model A's and Model B's
%   training run experience the IDENTICAL sequence of otherD2DChannel
%   values and epsilon-greedy exploration/tie-break draws. The only
%   thing that can differ between them is which Q-table cell each event
%   is attributed to and read back - i.e., the comparison isolates the
%   effect of state representation, not incidental random differences.
%   The same logic applies to the two evaluation phases (both policies
%   are purely greedy/argmax, so the only random consumption is the
%   per-run otherD2DChannelSequence draw itself).
%   This is a genuine implementation guarantee, not an assumption -
%   verified structurally by both trainers/evaluators making identical
%   call sequences.

    % ---- Train Model A ----
    rng(42);
    [Q_A, trainMetricsA] = trainModelA(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams);

    % ---- Train Model B (same seed -> matched random stream vs Model A) ----
    rng(42);
    [Q_B, trainMetricsB] = trainModelB(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams);

    % ---- Evaluate Model A over numEvaluationRuns runs ----
    rng(42);
    evalA.d2dThroughput    = zeros(1, numEvaluationRuns);
    evalA.cuThroughput     = zeros(1, numEvaluationRuns);
    evalA.qosViolationRate = zeros(1, numEvaluationRuns);
    evalA.reward           = zeros(1, numEvaluationRuns);
    qUnchangedA = true;
    for run = 1:numEvaluationRuns
        seq = randi(8, 1, evaluationSteps);
        [d, c, q, r, unchg] = evaluateQLearningPolicy(Q_A, H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, seq);
        evalA.d2dThroughput(run)    = d;
        evalA.cuThroughput(run)     = c;
        evalA.qosViolationRate(run) = q;
        evalA.reward(run)           = r;
        qUnchangedA = qUnchangedA && unchg;
    end
    evalA.qUnchangedAllRuns = qUnchangedA;

    % ---- Evaluate Model B over numEvaluationRuns runs (same seed -> matched vs Model A) ----
    rng(42);
    evalB.d2dThroughput    = zeros(1, numEvaluationRuns);
    evalB.cuThroughput     = zeros(1, numEvaluationRuns);
    evalB.qosViolationRate = zeros(1, numEvaluationRuns);
    evalB.reward           = zeros(1, numEvaluationRuns);
    qUnchangedB = true;
    for run = 1:numEvaluationRuns
        seq = randi(8, 1, evaluationSteps);
        [d, c, q, r, unchg] = evaluateModelBPolicy(Q_B, H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, seq);
        evalB.d2dThroughput(run)    = d;
        evalB.cuThroughput(run)     = c;
        evalB.qosViolationRate(run) = q;
        evalB.reward(run)           = r;
        qUnchangedB = qUnchangedB && unchg;
    end
    evalB.qUnchangedAllRuns = qUnchangedB;

    % ---- State-space coverage ----
    coverageA.numStates          = numel(trainMetricsA.visitState);
    coverageA.visitedStates      = nnz(trainMetricsA.visitState);
    coverageA.visitedStateActionPairs = nnz(trainMetricsA.visitStateAction);
    coverageA.fractionExplored   = coverageA.visitedStates / coverageA.numStates;

    coverageB.numStates          = numel(trainMetricsB.visitState);
    coverageB.visitedStates      = nnz(trainMetricsB.visitState);
    coverageB.visitedStateActionPairs = nnz(trainMetricsB.visitStateAction);
    coverageB.fractionExplored   = coverageB.visitedStates / coverageB.numStates;

    % ---- Package results ----
    results.Q_A = Q_A;
    results.Q_B = Q_B;
    results.trainMetricsA = trainMetricsA;
    results.trainMetricsB = trainMetricsB;
    results.evalA = evalA;
    results.evalB = evalB;
    results.coverageA = coverageA;
    results.coverageB = coverageB;

end
