package players.alphazero;

import core.Types;
import core.actions.Action;
import core.game.GameState;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

final class MapSharedNeuralCore {

    private static final int MAP_INPUTS = MapStateFeatures.FEATURE_COUNT;
    private static final int ACTION_INPUTS = MapActionFeatures.FEATURE_COUNT;
    private static final int ACTION_TAIL_INPUTS = ACTION_INPUTS - MAP_INPUTS;
    private static final int ACTIONS = Types.ACTION.values().length;
    private static final int DEFAULT_HIDDEN = 56;
    private static final int DEFAULT_ACTION_HIDDEN = 32;

    private final int hiddenSize;
    private final int actionHiddenSize;
    private final double[][] stateW;
    private final double[] stateB;
    private final double[] valueW;
    private double valueB;
    private final double[][] policyW;
    private final double[] policyB;
    private final double[][] actionStateW;
    private final double[][] actionTailW;
    private final double[] actionB;
    private final double[] actionOutW;
    private double actionOutB;
    private boolean trained;
    private String valuePath;
    private String policyPath;
    private String actionPath;

    MapSharedNeuralCore() {
        this(DEFAULT_HIDDEN, DEFAULT_ACTION_HIDDEN, 20260419L);
    }

    private MapSharedNeuralCore(int hiddenSize, int actionHiddenSize, long seed) {
        this.hiddenSize = hiddenSize;
        this.actionHiddenSize = actionHiddenSize;
        this.stateW = new double[hiddenSize][MAP_INPUTS];
        this.stateB = new double[hiddenSize];
        this.valueW = new double[hiddenSize];
        this.policyW = new double[ACTIONS][hiddenSize];
        this.policyB = new double[ACTIONS];
        this.actionStateW = new double[actionHiddenSize][hiddenSize];
        this.actionTailW = new double[actionHiddenSize][ACTION_TAIL_INPUTS];
        this.actionB = new double[actionHiddenSize];
        this.actionOutW = new double[actionHiddenSize];
        initialise(seed);
    }

    private void initialise(long seed) {
        Random rnd = new Random(seed);
        double mapScale = 1.0 / Math.sqrt(MAP_INPUTS);
        double hiddenScale = 1.0 / Math.sqrt(hiddenSize);
        double actionScale = 1.0 / Math.sqrt(ACTION_TAIL_INPUTS);
        double actionHiddenScale = 1.0 / Math.sqrt(actionHiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < MAP_INPUTS; i++) {
                stateW[j][i] = rnd.nextGaussian() * mapScale * 0.08;
            }
            valueW[j] = rnd.nextGaussian() * hiddenScale * 0.08;
        }
        for (int action = 0; action < ACTIONS; action++) {
            for (int j = 0; j < hiddenSize; j++) {
                policyW[action][j] = rnd.nextGaussian() * hiddenScale * 0.08;
            }
        }
        for (int k = 0; k < actionHiddenSize; k++) {
            for (int j = 0; j < hiddenSize; j++) {
                actionStateW[k][j] = rnd.nextGaussian() * hiddenScale * 0.08;
            }
            for (int i = 0; i < ACTION_TAIL_INPUTS; i++) {
                actionTailW[k][i] = rnd.nextGaussian() * actionScale * 0.08;
            }
            actionOutW[k] = rnd.nextGaussian() * actionHiddenScale * 0.08;
        }
    }

    boolean isTrained() {
        return trained;
    }

    double predictValue(double[] features) {
        if (features == null || features.length != MAP_INPUTS) {
            return 0.0;
        }
        double[] hidden = stateHidden(features);
        double z = valueB;
        for (int j = 0; j < hiddenSize; j++) {
            z += valueW[j] * hidden[j];
        }
        return Math.tanh(z);
    }

    double[] predictPolicy(double[] features) {
        if (features == null || features.length != MAP_INPUTS) {
            double[] uniform = new double[ACTIONS];
            for (int action = 0; action < ACTIONS; action++) {
                uniform[action] = 1.0 / ACTIONS;
            }
            return uniform;
        }
        return predictPolicyFromHidden(stateHidden(features));
    }

    double actionLogit(double[] features) {
        if (features == null || features.length != ACTION_INPUTS) {
            return 0.0;
        }
        return actionForward(features).logit;
    }

    double trainValue(ArrayList<ValueTrainingExample> examples, int epochs,
                      double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            int used = 0;
            for (ValueTrainingExample example : examples) {
                if (example.features == null || example.features.length != MAP_INPUTS) {
                    continue;
                }
                loss += updateValue(example, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    private double trainActionGrouped(ArrayList<ArrayList<ActionPolicyTrainingExample>> groups,
                                      int epochs, double learningRate, double l2, Random rnd) {
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(groups, rnd);
            double loss = 0.0;
            int used = 0;
            for (ArrayList<ActionPolicyTrainingExample> group : groups) {
                loss += updateActionGroup(group, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    double trainPolicy(ArrayList<PolicyTrainingExample> examples, int epochs,
                       double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            int used = 0;
            for (PolicyTrainingExample example : examples) {
                if (example.features == null || example.features.length != MAP_INPUTS
                        || !example.hasValidTarget(ACTIONS)) {
                    continue;
                }
                loss += updatePolicy(example, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    double trainAction(ArrayList<ActionPolicyTrainingExample> examples, int epochs,
                       double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        ArrayList<ArrayList<ActionPolicyTrainingExample>> groups =
                ActionPolicyTrainingExample.groupedBatches(examples, ACTION_INPUTS);
        if (!groups.isEmpty()) {
            return trainActionGrouped(groups, epochs, learningRate, l2, rnd);
        }

        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            int used = 0;
            for (ActionPolicyTrainingExample example : examples) {
                if (example.features == null || example.features.length != ACTION_INPUTS) {
                    continue;
                }
                loss += updateAction(example, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    private double updateValue(ValueTrainingExample example, double learningRate, double l2) {
        double[] hidden = stateHidden(example.features);
        double z = valueB;
        for (int j = 0; j < hiddenSize; j++) {
            z += valueW[j] * hidden[j];
        }
        double pred = Math.tanh(z);
        double error = pred - example.label;
        double loss = error * error;

        double dz = error * (1.0 - pred * pred);
        double[] oldValueW = valueW.clone();
        for (int j = 0; j < hiddenSize; j++) {
            double grad = dz * hidden[j] + l2 * valueW[j];
            valueW[j] -= learningRate * grad;
        }
        valueB -= learningRate * dz;

        for (int j = 0; j < hiddenSize; j++) {
            double dh = dz * oldValueW[j] * (1.0 - hidden[j] * hidden[j]);
            for (int i = 0; i < MAP_INPUTS; i++) {
                double grad = dh * example.features[i] + l2 * stateW[j][i];
                stateW[j][i] -= learningRate * grad;
            }
            stateB[j] -= learningRate * dh;
        }
        return loss;
    }

    private double updatePolicy(PolicyTrainingExample example, double learningRate, double l2) {
        double[] hidden = stateHidden(example.features);
        double[] probs = predictPolicyFromHidden(hidden);
        double[] error = new double[ACTIONS];
        double loss = 0.0;
        for (int action = 0; action < ACTIONS; action++) {
            double target = example.targetFor(action, ACTIONS);
            error[action] = probs[action] - target;
            if (target > 0.0) {
                loss += -target * Math.log(Math.max(1e-9, probs[action]));
            }
        }

        double[][] oldPolicyW = copy(policyW);
        for (int action = 0; action < ACTIONS; action++) {
            for (int j = 0; j < hiddenSize; j++) {
                double grad = error[action] * hidden[j] + l2 * policyW[action][j];
                policyW[action][j] -= learningRate * grad;
            }
            policyB[action] -= learningRate * error[action];
        }

        for (int j = 0; j < hiddenSize; j++) {
            double dh = 0.0;
            for (int action = 0; action < ACTIONS; action++) {
                dh += error[action] * oldPolicyW[action][j];
            }
            dh *= 1.0 - hidden[j] * hidden[j];
            for (int i = 0; i < MAP_INPUTS; i++) {
                double grad = dh * example.features[i] + l2 * stateW[j][i];
                stateW[j][i] -= learningRate * grad;
            }
            stateB[j] -= learningRate * dh;
        }
        return loss;
    }

    private double updateAction(ActionPolicyTrainingExample example, double learningRate, double l2) {
        ActionForward fwd = actionForward(example.features);
        double prediction = sigmoid(fwd.logit);
        double error = prediction - example.target;
        double loss = -(example.target * Math.log(Math.max(1e-9, prediction))
                + (1.0 - example.target) * Math.log(Math.max(1e-9, 1.0 - prediction)));

        updateActionWithGradient(fwd, error, learningRate, l2);
        return loss;
    }

    private double updateActionGroup(ArrayList<ActionPolicyTrainingExample> group,
                                     double learningRate, double l2) {
        double max = -Double.MAX_VALUE;
        ActionForward[] forwards = new ActionForward[group.size()];
        double[] logits = new double[group.size()];
        for (int i = 0; i < group.size(); i++) {
            forwards[i] = actionForward(group.get(i).features);
            logits[i] = forwards[i].logit;
            max = Math.max(max, logits[i]);
        }

        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = Math.exp(logits[i] - max);
            sum += logits[i];
        }

        double targetSum = ActionPolicyTrainingExample.targetSum(group);
        double loss = 0.0;
        for (int i = 0; i < group.size(); i++) {
            ActionPolicyTrainingExample example = group.get(i);
            double prediction = sum > 0.0 ? logits[i] / sum : 1.0 / group.size();
            double target = targetSum > 0.0 ? example.target / targetSum : 1.0 / group.size();
            loss += -target * Math.log(Math.max(1e-9, prediction));
            updateActionWithGradient(forwards[i], prediction - target, learningRate, l2);
        }
        return loss;
    }

    private void updateActionWithGradient(ActionForward fwd, double error,
                                          double learningRate, double l2) {
        double[] oldActionOutW = actionOutW.clone();
        double[][] oldActionStateW = copy(actionStateW);
        for (int k = 0; k < actionHiddenSize; k++) {
            double grad = error * fwd.actionHidden[k] + l2 * actionOutW[k];
            actionOutW[k] -= learningRate * grad;
        }
        actionOutB -= learningRate * error;

        double[] dhAction = new double[actionHiddenSize];
        for (int k = 0; k < actionHiddenSize; k++) {
            dhAction[k] = error * oldActionOutW[k]
                    * (1.0 - fwd.actionHidden[k] * fwd.actionHidden[k]);
            for (int j = 0; j < hiddenSize; j++) {
                double grad = dhAction[k] * fwd.stateHidden[j] + l2 * actionStateW[k][j];
                actionStateW[k][j] -= learningRate * grad;
            }
            for (int i = 0; i < ACTION_TAIL_INPUTS; i++) {
                double grad = dhAction[k] * fwd.actionTail[i] + l2 * actionTailW[k][i];
                actionTailW[k][i] -= learningRate * grad;
            }
            actionB[k] -= learningRate * dhAction[k];
        }

        for (int j = 0; j < hiddenSize; j++) {
            double dhState = 0.0;
            for (int k = 0; k < actionHiddenSize; k++) {
                dhState += dhAction[k] * oldActionStateW[k][j];
            }
            dhState *= 1.0 - fwd.stateHidden[j] * fwd.stateHidden[j];
            for (int i = 0; i < MAP_INPUTS; i++) {
                double grad = dhState * fwd.mapFeatures[i] + l2 * stateW[j][i];
                stateW[j][i] -= learningRate * grad;
            }
            stateB[j] -= learningRate * dhState;
        }
    }

    void rememberValuePath(String path) {
        valuePath = path;
    }

    void rememberPolicyPath(String path) {
        policyPath = path;
    }

    void rememberActionPath(String path) {
        actionPath = path;
    }

    void saveKnownPaths() {
        if (valuePath != null && !valuePath.isEmpty()) {
            save(valuePath);
        }
        if (policyPath != null && !policyPath.isEmpty() && !policyPath.equals(valuePath)) {
            save(policyPath);
        }
        if (actionPath != null && !actionPath.isEmpty()
                && !actionPath.equals(valuePath) && !actionPath.equals(policyPath)) {
            save(actionPath);
        }
    }

    private double[] stateHidden(double[] features) {
        double[] hidden = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double z = stateB[j];
            for (int i = 0; i < MAP_INPUTS; i++) {
                z += stateW[j][i] * features[i];
            }
            hidden[j] = Math.tanh(z);
        }
        return hidden;
    }

    private double[] predictPolicyFromHidden(double[] hidden) {
        double[] logits = new double[ACTIONS];
        double max = -Double.MAX_VALUE;
        for (int action = 0; action < ACTIONS; action++) {
            double z = policyB[action];
            for (int j = 0; j < hiddenSize; j++) {
                z += policyW[action][j] * hidden[j];
            }
            logits[action] = z;
            max = Math.max(max, z);
        }

        double sum = 0.0;
        for (int action = 0; action < ACTIONS; action++) {
            logits[action] = Math.exp(logits[action] - max);
            sum += logits[action];
        }
        for (int action = 0; action < ACTIONS; action++) {
            logits[action] = sum > 0.0 ? logits[action] / sum : 1.0 / ACTIONS;
        }
        return logits;
    }

    private ActionForward actionForward(double[] features) {
        double[] mapFeatures = new double[MAP_INPUTS];
        double[] actionTail = new double[ACTION_TAIL_INPUTS];
        System.arraycopy(features, 0, mapFeatures, 0, MAP_INPUTS);
        System.arraycopy(features, MAP_INPUTS, actionTail, 0, ACTION_TAIL_INPUTS);

        double[] stateHidden = stateHidden(mapFeatures);
        double[] actionHidden = new double[actionHiddenSize];
        for (int k = 0; k < actionHiddenSize; k++) {
            double z = actionB[k];
            for (int j = 0; j < hiddenSize; j++) {
                z += actionStateW[k][j] * stateHidden[j];
            }
            for (int i = 0; i < ACTION_TAIL_INPUTS; i++) {
                z += actionTailW[k][i] * actionTail[i];
            }
            actionHidden[k] = Math.tanh(z);
        }

        double logit = actionOutB;
        for (int k = 0; k < actionHiddenSize; k++) {
            logit += actionOutW[k] * actionHidden[k];
        }
        return new ActionForward(mapFeatures, actionTail, stateHidden, actionHidden, logit);
    }

    static MapSharedNeuralCore load(String path) {
        if (path == null || path.trim().isEmpty()) {
            return new MapSharedNeuralCore();
        }
        File file = new File(path);
        if (!file.exists()) {
            return new MapSharedNeuralCore();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Map shared neural Polytopia")) {
                return new MapSharedNeuralCore();
            }
            String mapLine = reader.readLine();
            if (mapLine == null || !mapLine.equals("mapFeatures\t" + MAP_INPUTS)) {
                return new MapSharedNeuralCore();
            }
            String actionLine = reader.readLine();
            if (actionLine == null || !actionLine.equals("actionFeatures\t" + ACTION_INPUTS)) {
                return new MapSharedNeuralCore();
            }
            int hidden = parseInt(reader.readLine(), "hidden", DEFAULT_HIDDEN);
            int actionHidden = parseInt(reader.readLine(), "actionHidden", DEFAULT_ACTION_HIDDEN);
            MapSharedNeuralCore core = new MapSharedNeuralCore(hidden, actionHidden, 20260419L);
            readVector(reader.readLine(), core.stateB, "stateB");
            for (int j = 0; j < hidden; j++) {
                readVector(reader.readLine(), core.stateW[j], "stateW");
            }
            readVector(reader.readLine(), core.valueW, "valueW");
            core.valueB = parseDouble(reader.readLine(), "valueB", 0.0);
            readVector(reader.readLine(), core.policyB, "policyB");
            for (int action = 0; action < ACTIONS; action++) {
                readVector(reader.readLine(), core.policyW[action], "policyW");
            }
            readVector(reader.readLine(), core.actionB, "actionB");
            for (int k = 0; k < actionHidden; k++) {
                readVector(reader.readLine(), core.actionStateW[k], "actionStateW");
            }
            for (int k = 0; k < actionHidden; k++) {
                readVector(reader.readLine(), core.actionTailW[k], "actionTailW");
            }
            readVector(reader.readLine(), core.actionOutW, "actionOutW");
            core.actionOutB = parseDouble(reader.readLine(), "actionOutB", 0.0);
            core.trained = true;
            return core;
        } catch (IOException e) {
            throw new RuntimeException("Could not load map-shared neural model from " + path, e);
        }
    }

    void save(String path) {
        if (path == null || path.trim().isEmpty()) {
            return;
        }
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write("# Map shared neural Polytopia policy-value-action model");
                writer.newLine();
                writer.write("mapFeatures\t" + MAP_INPUTS);
                writer.newLine();
                writer.write("actionFeatures\t" + ACTION_INPUTS);
                writer.newLine();
                writer.write("hidden\t" + hiddenSize);
                writer.newLine();
                writer.write("actionHidden\t" + actionHiddenSize);
                writer.newLine();
                writeVector(writer, "stateB", stateB);
                for (int j = 0; j < hiddenSize; j++) {
                    writeVector(writer, "stateW", stateW[j]);
                }
                writeVector(writer, "valueW", valueW);
                writer.write("valueB\t" + valueB);
                writer.newLine();
                writeVector(writer, "policyB", policyB);
                for (int action = 0; action < ACTIONS; action++) {
                    writeVector(writer, "policyW", policyW[action]);
                }
                writeVector(writer, "actionB", actionB);
                for (int k = 0; k < actionHiddenSize; k++) {
                    writeVector(writer, "actionStateW", actionStateW[k]);
                }
                for (int k = 0; k < actionHiddenSize; k++) {
                    writeVector(writer, "actionTailW", actionTailW[k]);
                }
                writeVector(writer, "actionOutW", actionOutW);
                writer.write("actionOutB\t" + actionOutB);
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not save map-shared neural model to " + path, e);
        }
    }

    private static int parseInt(String line, String prefix, int fallback) {
        if (line == null || !line.startsWith(prefix + "\t")) {
            return fallback;
        }
        return Integer.parseInt(line.split("\\t")[1]);
    }

    private static double parseDouble(String line, String prefix, double fallback) {
        if (line == null || !line.startsWith(prefix + "\t")) {
            return fallback;
        }
        return Double.parseDouble(line.split("\\t")[1]);
    }

    private static void readVector(String line, double[] target, String prefix) {
        if (line == null || !line.startsWith(prefix + "\t")) {
            return;
        }
        String[] parts = line.split("\\t");
        for (int i = 0; i < target.length && i + 1 < parts.length; i++) {
            target[i] = Double.parseDouble(parts[i + 1]);
        }
    }

    private static void writeVector(BufferedWriter writer, String prefix, double[] values) throws IOException {
        writer.write(prefix);
        for (double value : values) {
            writer.write('\t');
            writer.write(Double.toString(value));
        }
        writer.newLine();
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            double z = Math.exp(-value);
            return 1.0 / (1.0 + z);
        }
        double z = Math.exp(value);
        return z / (1.0 + z);
    }

    private static double[][] copy(double[][] input) {
        double[][] out = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            out[i] = input[i].clone();
        }
        return out;
    }

    private static final class ActionForward {
        final double[] mapFeatures;
        final double[] actionTail;
        final double[] stateHidden;
        final double[] actionHidden;
        final double logit;

        ActionForward(double[] mapFeatures, double[] actionTail,
                      double[] stateHidden, double[] actionHidden, double logit) {
            this.mapFeatures = mapFeatures;
            this.actionTail = actionTail;
            this.stateHidden = stateHidden;
            this.actionHidden = actionHidden;
            this.logit = logit;
        }
    }
}

final class MapSharedNeuralValueFunction implements ValueModel {

    private final MapSharedNeuralCore core;

    MapSharedNeuralValueFunction(MapSharedNeuralCore core) {
        this.core = core;
    }

    @Override
    public boolean isTrained() {
        return core.isTrained();
    }

    @Override
    public double predict(GameState state, int playerID, ArrayList<Integer> allIds) {
        return predict(MapStateFeatures.extract(state, playerID, allIds));
    }

    @Override
    public double predict(double[] features) {
        return core.predictValue(features);
    }

    @Override
    public double train(ArrayList<ValueTrainingExample> examples, int epochs,
                        double learningRate, double l2, long seed) {
        return core.trainValue(examples, epochs, learningRate, l2, seed);
    }

    @Override
    public void save(String path) {
        core.rememberValuePath(path);
        core.saveKnownPaths();
    }
}

final class MapSharedNeuralPolicyFunction implements PolicyModel {

    private final MapSharedNeuralCore core;

    MapSharedNeuralPolicyFunction(MapSharedNeuralCore core) {
        this.core = core;
    }

    @Override
    public boolean isTrained() {
        return core.isTrained();
    }

    @Override
    public double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType) {
        return predict(MapStateFeatures.extract(state, playerID, allIds))[actionType.ordinal()];
    }

    @Override
    public double[] predict(double[] features) {
        return core.predictPolicy(features);
    }

    @Override
    public double train(ArrayList<PolicyTrainingExample> examples, int epochs,
                        double learningRate, double l2, long seed) {
        return core.trainPolicy(examples, epochs, learningRate, l2, seed);
    }

    @Override
    public void save(String path) {
        core.rememberPolicyPath(path);
        core.saveKnownPaths();
    }
}

final class MapSharedNeuralActionPolicyFunction implements ActionPolicyModel {

    private final MapSharedNeuralCore core;

    MapSharedNeuralActionPolicyFunction(MapSharedNeuralCore core) {
        this.core = core;
    }

    @Override
    public boolean isTrained() {
        return core.isTrained();
    }

    @Override
    public double logit(GameState state, GameState nextState, int playerID,
                        ArrayList<Integer> allIds, Action action) {
        double[] features = MapActionFeatures.extract(state, nextState, playerID, allIds, action);
        return core.actionLogit(features);
    }

    @Override
    public double train(ArrayList<ActionPolicyTrainingExample> examples, int epochs,
                        double learningRate, double l2, long seed) {
        return core.trainAction(examples, epochs, learningRate, l2, seed);
    }

    @Override
    public void save(String path) {
        core.rememberActionPath(path);
        core.saveKnownPaths();
    }
}
