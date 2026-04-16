package players.alphazero;

import core.actions.Action;
import core.game.GameState;
import players.Agent;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

public class RecordingAgent extends Agent {

    private final Agent delegate;
    private final String datasetPath;
    private final String policyDatasetPath;
    private final double sampleProbability;
    private final int maxExamplesPerGame;
    private final Random rnd;
    private final ArrayList<ValueTrainingExample> pending;

    public RecordingAgent(Agent delegate, String datasetPath, String policyDatasetPath,
                          double sampleProbability, int maxExamplesPerGame, long seed) {
        super(seed);
        this.delegate = delegate;
        this.datasetPath = datasetPath;
        this.policyDatasetPath = policyDatasetPath;
        this.sampleProbability = sampleProbability;
        this.maxExamplesPerGame = maxExamplesPerGame;
        this.rnd = new Random(seed);
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
            pending.add(new ValueTrainingExample(0.0, features));
        }

        Action action = delegate.act(gs, ect);
        if (sampled && action != null) {
            ArrayList<PolicyTrainingExample> policyExamples = new ArrayList<>();
            policyExamples.add(new PolicyTrainingExample(action.getActionType().ordinal(), features));
            PolicyDataset.append(policyDatasetPath, policyExamples);
        }
        return action;
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
            labeled.add(new ValueTrainingExample(label, example.features));
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
        return new RecordingAgent(delegateCopy, datasetPath, policyDatasetPath, sampleProbability, maxExamplesPerGame, seed);
    }
}
