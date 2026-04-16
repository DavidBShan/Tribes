package players.alphazero;

import core.actions.Action;
import core.game.GameState;
import org.json.JSONObject;
import players.Agent;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

public class RecordingAgent extends Agent {

    private final Agent delegate;
    private final String botName;
    private final String datasetPath;
    private final String policyDatasetPath;
    private final String policyTargetMode;
    private final SftTrajectoryWriter trajectoryWriter;
    private final JSONObject setupMetadata;
    private final int episode;
    private final int seat;
    private final double sampleProbability;
    private final double trajectorySampleProbability;
    private final double valuePositionBlend;
    private final int maxExamplesPerGame;
    private final int maxTrajectoriesPerGame;
    private final Random rnd;
    private final Random trajectoryRnd;
    private final ArrayList<ValueTrainingExample> pending;
    private int localActionIndex = 0;

    public RecordingAgent(Agent delegate, String datasetPath, String policyDatasetPath,
                          double sampleProbability, int maxExamplesPerGame, long seed) {
        this(delegate, "unknown", datasetPath, policyDatasetPath, null, new JSONObject(),
                -1, -1, sampleProbability, maxExamplesPerGame, 0.0, 0, "action", 0.0, seed);
    }

    public RecordingAgent(Agent delegate, String botName, String datasetPath, String policyDatasetPath,
                          SftTrajectoryWriter trajectoryWriter, JSONObject setupMetadata, int episode, int seat,
                          double sampleProbability, int maxExamplesPerGame,
                          double trajectorySampleProbability, int maxTrajectoriesPerGame,
                          String policyTargetMode, double valuePositionBlend, long seed) {
        super(seed);
        this.delegate = delegate;
        this.botName = botName;
        this.datasetPath = datasetPath;
        this.policyDatasetPath = policyDatasetPath;
        this.policyTargetMode = policyTargetMode == null ? "action" : policyTargetMode;
        this.trajectoryWriter = trajectoryWriter;
        this.setupMetadata = setupMetadata == null ? new JSONObject() : new JSONObject(setupMetadata.toString());
        this.episode = episode;
        this.seat = seat;
        this.sampleProbability = sampleProbability;
        this.maxExamplesPerGame = maxExamplesPerGame;
        this.trajectorySampleProbability = trajectorySampleProbability;
        this.maxTrajectoriesPerGame = maxTrajectoriesPerGame;
        this.valuePositionBlend = Math.max(0.0, Math.min(1.0, valuePositionBlend));
        this.rnd = new Random(seed);
        this.trajectoryRnd = new Random(seed ^ 0x5DEECE66DL);
        this.pending = new ArrayList<>();
    }

    @Override
    public void setPlayerIDs(int playerID, ArrayList<Integer> allIds) {
        super.setPlayerIDs(playerID, allIds);
        delegate.setPlayerIDs(playerID, allIds);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        double[] features = StateFeatures.extract(gs, playerID, allPlayerIDs);
        boolean sampled = pending.size() < maxExamplesPerGame && rnd.nextDouble() <= sampleProbability;
        if (sampled) {
            double positionLabel = StateFeatures.positionValue(gs, playerID, allPlayerIDs);
            pending.add(new ValueTrainingExample(0.0, features, positionLabel));
        }

        long started = System.nanoTime();
        Action action = delegate.act(gs, ect);
        long elapsedMicros = (System.nanoTime() - started) / 1000L;
        if (sampled && action != null) {
            ArrayList<PolicyTrainingExample> policyExamples = new ArrayList<>();
            policyExamples.add(policyExample(action, features));
            PolicyDataset.append(policyDatasetPath, policyExamples);
        }
        if (shouldRecordTrajectory() && action != null) {
            trajectoryWriter.writeSample(episode, localActionIndex, botName, seat, playerID, setupMetadata,
                    gs, action, elapsedMicros, allPlayerIDs);
        }
        localActionIndex++;
        return action;
    }

    private PolicyTrainingExample policyExample(Action action, double[] features) {
        if ("visit".equalsIgnoreCase(policyTargetMode) && delegate instanceof AlphaZeroAgent) {
            double[] targets = ((AlphaZeroAgent) delegate).lastVisitPolicyTargets();
            if (targets != null) {
                return new PolicyTrainingExample(targets, features);
            }
        }
        return new PolicyTrainingExample(action.getActionType().ordinal(), features);
    }

    private boolean shouldRecordTrajectory() {
        return trajectoryWriter != null
                && localActionIndex < maxTrajectoriesPerGame
                && trajectoryRnd.nextDouble() <= trajectorySampleProbability;
    }

    @Override
    public void result(GameState gs, double reward) {
        delegate.result(gs, reward);
        if (pending.isEmpty()) {
            return;
        }

        double label = StateFeatures.outcomeLabel(gs, playerID, allPlayerIDs);
        ArrayList<ValueTrainingExample> labeled = new ArrayList<>();
        for (ValueTrainingExample example : pending) {
            double blended = StateFeatures.clamp((1.0 - valuePositionBlend) * label
                    + valuePositionBlend * example.positionLabel);
            labeled.add(new ValueTrainingExample(blended, example.features));
        }
        ValueDataset.append(datasetPath, labeled);
        pending.clear();
    }

    @Override
    public Agent copy() {
        Agent delegateCopy = delegate.copy();
        if (delegateCopy == null) {
            delegateCopy = delegate;
        }
        return new RecordingAgent(delegateCopy, botName, datasetPath, policyDatasetPath, trajectoryWriter,
                setupMetadata, episode, seat, sampleProbability, maxExamplesPerGame,
                trajectorySampleProbability, maxTrajectoriesPerGame, policyTargetMode,
                valuePositionBlend, seed);
    }
}
