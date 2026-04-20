package players.alphazero;

import players.heuristics.AlgParams;

/**
 * Parameters for the AlphaZero-style PUCT agent.
 */
public class AZParams extends AlgParams {

    public double cpuct = 1.50;
    public int maxActionsPerNode = 36;
    public int prefilterActions = 96;
    public double priorTemperature = 0.75;
    public double heuristicScale = 45.0;
    public double heuristicBlend = 0.35;
    public double disagreementHeuristicBlend = -1.0;
    public double disagreementHeuristicThreshold = 0.35;
    public double advisorOverrideMargin = 0.08;
    public double greedyPriorityThreshold = 99.0;
    public double policyLogitWeight = 0.06;
    public double actionPolicyLogitWeight = 0.0;
    public double positionBlend = 0.20;
    public double opponentAdversaryWeight = 1.0;
    public double opponentSelfishWeight = 0.0;
    public double rootNoiseFraction = 0.0;
    public double rootDirichletAlpha = 0.30;
    public double rootGumbelScale = 0.0;
    public boolean rootSequentialHalving = false;
    public double rootValueSelectionWeight = 0.0;
    public double improvedPolicyValueWeight = 1.0;
    public double improvedPolicyTemperature = 1.0;
    public double visitSamplingTemperature = 0.0;
    public int visitSamplingUntilTick = 0;
    public int forceTurnAfterActions = 10;
    public boolean tacticalShortcuts = true;
    public boolean advisorOverride = true;
    public boolean staticPriors = true;
    public boolean nextStateValuePrior = true;
    public boolean learnedValueOnly = false;
    public boolean prefilterByStaticScore = true;
    public String networkType = ModelFactory.LINEAR;
    public String modelPath = "models/alphazero-value.tsv";
    public String policyPath = "models/alphazero-policy.tsv";
    public String actionPolicyPath = "";

    public AZParams() {
        stop_type = STOP_FMCALLS;
        num_fmcalls = 2500;
        ROLLOUT_LENGTH = 18;
        FORCE_TURN_END = ROLLOUT_LENGTH + 1;
        heuristic_method = DIFF_HEURISTIC;

        String model = System.getProperty("az.model");
        if (model != null && !model.trim().isEmpty()) {
            modelPath = model.trim();
        }

        String policy = System.getProperty("az.policy");
        if (policy != null && !policy.trim().isEmpty()) {
            policyPath = policy.trim();
        }

        String calls = System.getProperty("az.fmcalls");
        if (calls != null && !calls.trim().isEmpty()) {
            num_fmcalls = Integer.parseInt(calls.trim());
        }

        String depth = System.getProperty("az.depth");
        if (depth != null && !depth.trim().isEmpty()) {
            ROLLOUT_LENGTH = Integer.parseInt(depth.trim());
        }
    }

    public int ROLLOUT_LENGTH;
}
