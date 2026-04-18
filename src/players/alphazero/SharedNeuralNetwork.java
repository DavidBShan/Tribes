package players.alphazero;

import core.Types;
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

final class SharedNeuralCore {

    private static final int INPUTS = StateFeatures.FEATURE_COUNT;
    private static final int ACTIONS = Types.ACTION.values().length;
    private static final int DEFAULT_HIDDEN = 64;

    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[] valueW;
    private double valueB;
    private final double[][] policyW;
    private final double[] policyB;
    private boolean trained;
    private String valuePath;
    private String policyPath;

    static final class TrainingResult {
        final double valueLoss;
        final double policyLoss;
        final int valueRows;
        final int policyRows;

        TrainingResult(double valueLoss, double policyLoss, int valueRows, int policyRows) {
            this.valueLoss = valueLoss;
            this.policyLoss = policyLoss;
            this.valueRows = valueRows;
            this.policyRows = policyRows;
        }
    }

    SharedNeuralCore() {
        this(DEFAULT_HIDDEN, 20260418L);
    }

    private SharedNeuralCore(int hiddenSize, long seed) {
        this.hiddenSize = hiddenSize;
        this.w1 = new double[hiddenSize][INPUTS];
        this.b1 = new double[hiddenSize];
        this.valueW = new double[hiddenSize];
        this.policyW = new double[ACTIONS][hiddenSize];
        this.policyB = new double[ACTIONS];
        initialise(seed);
    }

    private void initialise(long seed) {
        Random rnd = new Random(seed);
        double inputScale = 1.0 / Math.sqrt(INPUTS);
        double hiddenScale = 1.0 / Math.sqrt(hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < INPUTS; i++) {
                w1[j][i] = rnd.nextGaussian() * inputScale * 0.10;
            }
            valueW[j] = rnd.nextGaussian() * hiddenScale * 0.10;
        }
        for (int action = 0; action < ACTIONS; action++) {
            for (int j = 0; j < hiddenSize; j++) {
                policyW[action][j] = rnd.nextGaussian() * hiddenScale * 0.10;
            }
        }
    }

    boolean isTrained() {
        return trained;
    }

    double predictValue(double[] features) {
        double[] hidden = hidden(features);
        double z = valueB;
        for (int j = 0; j < hiddenSize; j++) {
            z += valueW[j] * hidden[j];
        }
        return Math.tanh(z);
    }

    double[] predictPolicy(double[] features) {
        return predictPolicyFromHidden(hidden(features));
    }

    double trainValue(ArrayList<ValueTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            for (ValueTrainingExample example : examples) {
                loss += updateValue(example, learningRate, l2);
            }
            lastLoss = loss / examples.size();
        }
        trained = true;
        return lastLoss;
    }

    double trainPolicy(ArrayList<PolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            int trainedRows = 0;
            for (PolicyTrainingExample example : examples) {
                if (!example.hasValidTarget(ACTIONS)) {
                    continue;
                }

                trainedRows++;
                loss += updatePolicy(example, learningRate, l2);
            }
            lastLoss = trainedRows > 0 ? loss / trainedRows : Double.NaN;
        }
        trained = true;
        return lastLoss;
    }

    TrainingResult trainJoint(ArrayList<ValueTrainingExample> valueExamples,
                              ArrayList<PolicyTrainingExample> policyExamples,
                              int valueEpochs, int policyEpochs,
                              double valueLearningRate, double policyLearningRate,
                              double l2, long seed) {
        boolean hasValue = valueExamples != null && !valueExamples.isEmpty() && valueEpochs > 0;
        boolean hasPolicy = policyExamples != null && !policyExamples.isEmpty() && policyEpochs > 0;
        if (!hasValue && !hasPolicy) {
            return new TrainingResult(Double.NaN, Double.NaN, 0, 0);
        }

        ArrayList<ValueTrainingExample> values = hasValue
                ? new ArrayList<>(valueExamples) : new ArrayList<>();
        ArrayList<PolicyTrainingExample> policies = hasPolicy
                ? new ArrayList<>(policyExamples) : new ArrayList<>();
        Random rnd = new Random(seed);
        int rounds = Math.max(valueEpochs, policyEpochs);
        double lastValueLoss = Double.NaN;
        double lastPolicyLoss = Double.NaN;
        int lastValueRows = 0;
        int lastPolicyRows = 0;

        for (int epoch = 0; epoch < rounds; epoch++) {
            boolean trainValueThisEpoch = epoch < valueEpochs && !values.isEmpty();
            boolean trainPolicyThisEpoch = epoch < policyEpochs && !policies.isEmpty();
            if (trainValueThisEpoch) {
                Collections.shuffle(values, rnd);
            }
            if (trainPolicyThisEpoch) {
                Collections.shuffle(policies, rnd);
            }

            int steps = Math.max(trainValueThisEpoch ? values.size() : 0,
                    trainPolicyThisEpoch ? policies.size() : 0);
            double valueLoss = 0.0;
            double policyLoss = 0.0;
            int valueRows = 0;
            int policyRows = 0;

            for (int step = 0; step < steps; step++) {
                if (trainValueThisEpoch) {
                    ValueTrainingExample example = values.get(step % values.size());
                    valueLoss += updateValue(example, valueLearningRate, l2);
                    valueRows++;
                }
                if (trainPolicyThisEpoch) {
                    PolicyTrainingExample example = policies.get(step % policies.size());
                    if (example.hasValidTarget(ACTIONS)) {
                        policyLoss += updatePolicy(example, policyLearningRate, l2);
                        policyRows++;
                    }
                }
            }

            if (valueRows > 0) {
                lastValueLoss = valueLoss / valueRows;
                lastValueRows = valueRows;
            }
            if (policyRows > 0) {
                lastPolicyLoss = policyLoss / policyRows;
                lastPolicyRows = policyRows;
            }
        }

        trained = true;
        return new TrainingResult(lastValueLoss, lastPolicyLoss, lastValueRows, lastPolicyRows);
    }

    private double updateValue(ValueTrainingExample example, double learningRate, double l2) {
        double[] hidden = hidden(example.features);
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
            for (int i = 0; i < INPUTS; i++) {
                double grad = dh * example.features[i] + l2 * w1[j][i];
                w1[j][i] -= learningRate * grad;
            }
            b1[j] -= learningRate * dh;
        }
        return loss;
    }

    private double updatePolicy(PolicyTrainingExample example, double learningRate, double l2) {
        double[] hidden = hidden(example.features);
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
            for (int i = 0; i < INPUTS; i++) {
                double grad = dh * example.features[i] + l2 * w1[j][i];
                w1[j][i] -= learningRate * grad;
            }
            b1[j] -= learningRate * dh;
        }
        return loss;
    }

    void rememberValuePath(String path) {
        valuePath = path;
    }

    void rememberPolicyPath(String path) {
        policyPath = path;
    }

    void saveKnownPaths() {
        if (valuePath != null && !valuePath.isEmpty()) {
            save(valuePath);
        }
        if (policyPath != null && !policyPath.isEmpty() && !policyPath.equals(valuePath)) {
            save(policyPath);
        }
    }

    private double[] hidden(double[] features) {
        double[] hidden = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double z = b1[j];
            for (int i = 0; i < INPUTS; i++) {
                z += w1[j][i] * features[i];
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

    static SharedNeuralCore load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return new SharedNeuralCore();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Shared neural Polytopia policy-value model")) {
                return new SharedNeuralCore();
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new SharedNeuralCore();
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            SharedNeuralCore core = new SharedNeuralCore(hidden, 20260418L);
            readVector(reader.readLine(), core.b1, "b1");
            for (int j = 0; j < hidden; j++) {
                readVector(reader.readLine(), core.w1[j], "w1");
            }
            readVector(reader.readLine(), core.valueW, "valueW");
            String valueBLine = reader.readLine();
            if (valueBLine != null && valueBLine.startsWith("valueB\t")) {
                core.valueB = Double.parseDouble(valueBLine.split("\\t")[1]);
            }
            readVector(reader.readLine(), core.policyB, "policyB");
            for (int action = 0; action < ACTIONS; action++) {
                readVector(reader.readLine(), core.policyW[action], "policyW");
            }
            core.trained = true;
            return core;
        } catch (IOException e) {
            throw new RuntimeException("Could not load shared neural model from " + path, e);
        }
    }

    void save(String path) {
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write("# Shared neural Polytopia policy-value model. Features: " + INPUTS
                        + " Actions: " + ACTIONS);
                writer.newLine();
                writer.write("hidden\t" + hiddenSize);
                writer.newLine();
                writeVector(writer, "b1", b1);
                for (int j = 0; j < hiddenSize; j++) {
                    writeVector(writer, "w1", w1[j]);
                }
                writeVector(writer, "valueW", valueW);
                writer.write("valueB\t" + valueB);
                writer.newLine();
                writeVector(writer, "policyB", policyB);
                for (int action = 0; action < ACTIONS; action++) {
                    writeVector(writer, "policyW", policyW[action]);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not save shared neural model to " + path, e);
        }
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

    private static double[][] copy(double[][] input) {
        double[][] out = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            out[i] = input[i].clone();
        }
        return out;
    }
}

final class SharedNeuralValueFunction implements ValueModel {

    private final SharedNeuralCore core;

    SharedNeuralValueFunction(SharedNeuralCore core) {
        this.core = core;
    }

    @Override
    public boolean isTrained() {
        return core.isTrained();
    }

    @Override
    public double predict(GameState state, int playerID, ArrayList<Integer> allIds) {
        return predict(StateFeatures.extract(state, playerID, allIds));
    }

    @Override
    public double predict(double[] features) {
        return core.predictValue(features);
    }

    @Override
    public double train(ArrayList<ValueTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        return core.trainValue(examples, epochs, learningRate, l2, seed);
    }

    @Override
    public void save(String path) {
        core.rememberValuePath(path);
        core.saveKnownPaths();
    }
}

final class SharedNeuralPolicyFunction implements PolicyModel {

    private final SharedNeuralCore core;

    SharedNeuralPolicyFunction(SharedNeuralCore core) {
        this.core = core;
    }

    @Override
    public boolean isTrained() {
        return core.isTrained();
    }

    @Override
    public double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType) {
        return predict(StateFeatures.extract(state, playerID, allIds))[actionType.ordinal()];
    }

    @Override
    public double[] predict(double[] features) {
        return core.predictPolicy(features);
    }

    @Override
    public double train(ArrayList<PolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        return core.trainPolicy(examples, epochs, learningRate, l2, seed);
    }

    @Override
    public void save(String path) {
        core.rememberPolicyPath(path);
        core.saveKnownPaths();
    }
}
