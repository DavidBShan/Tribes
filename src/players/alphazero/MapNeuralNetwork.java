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

final class MapNeuralValueFunction implements ValueModel {

    private static final int INPUTS = MapStateFeatures.FEATURE_COUNT;
    private static final int DEFAULT_HIDDEN = 32;
    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private boolean trained;

    private MapNeuralValueFunction() {
        this(DEFAULT_HIDDEN, 20260418L);
    }

    private MapNeuralValueFunction(int hiddenSize, long seed) {
        this.hiddenSize = hiddenSize;
        this.w1 = new double[hiddenSize][INPUTS];
        this.b1 = new double[hiddenSize];
        this.w2 = new double[hiddenSize];
        initialise(seed);
    }

    private void initialise(long seed) {
        Random rnd = new Random(seed);
        double inputScale = 1.0 / Math.sqrt(INPUTS);
        double hiddenScale = 1.0 / Math.sqrt(hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < INPUTS; i++) {
                w1[j][i] = rnd.nextGaussian() * inputScale * 0.08;
            }
            w2[j] = rnd.nextGaussian() * hiddenScale * 0.08;
        }
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double predict(GameState state, int playerID, ArrayList<Integer> allIds) {
        return predict(MapStateFeatures.extract(state, playerID, allIds));
    }

    @Override
    public double predict(double[] features) {
        double[] hidden = hidden(features);
        double z = b2;
        for (int j = 0; j < hiddenSize; j++) {
            z += w2[j] * hidden[j];
        }
        return Math.tanh(z);
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

    @Override
    public double train(ArrayList<ValueTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
            for (ValueTrainingExample example : examples) {
                double[] hidden = hidden(example.features);
                double z = b2;
                for (int j = 0; j < hiddenSize; j++) {
                    z += w2[j] * hidden[j];
                }
                double pred = Math.tanh(z);
                double error = pred - example.label;
                loss += error * error;

                double dz = error * (1.0 - pred * pred);
                double[] oldW2 = w2.clone();
                for (int j = 0; j < hiddenSize; j++) {
                    double grad = dz * hidden[j] + l2 * w2[j];
                    w2[j] -= learningRate * grad;
                }
                b2 -= learningRate * dz;

                for (int j = 0; j < hiddenSize; j++) {
                    double dh = dz * oldW2[j] * (1.0 - hidden[j] * hidden[j]);
                    for (int i = 0; i < INPUTS; i++) {
                        double grad = dh * example.features[i] + l2 * w1[j][i];
                        w1[j][i] -= learningRate * grad;
                    }
                    b1[j] -= learningRate * dh;
                }
            }
            lastLoss = loss / examples.size();
        }

        trained = true;
        return lastLoss;
    }

    static MapNeuralValueFunction load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return new MapNeuralValueFunction();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Map Neural Polytopia value model")) {
                return new MapNeuralValueFunction();
            }
            String featureLine = reader.readLine();
            if (featureLine == null || !featureLine.equals("features\t" + INPUTS)) {
                return new MapNeuralValueFunction();
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new MapNeuralValueFunction();
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            MapNeuralValueFunction fn = new MapNeuralValueFunction(hidden, 20260418L);
            readVector(reader.readLine(), fn.b1, "b1");
            for (int j = 0; j < hidden; j++) {
                readVector(reader.readLine(), fn.w1[j], "w1");
            }
            readVector(reader.readLine(), fn.w2, "w2");
            String b2Line = reader.readLine();
            if (b2Line != null && b2Line.startsWith("b2\t")) {
                fn.b2 = Double.parseDouble(b2Line.split("\\t")[1]);
            }
            fn.trained = true;
            return fn;
        } catch (IOException e) {
            throw new RuntimeException("Could not load map-neural value model from " + path, e);
        }
    }

    @Override
    public void save(String path) {
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write("# Map Neural Polytopia value model");
                writer.newLine();
                writer.write("features\t" + INPUTS);
                writer.newLine();
                writer.write("hidden\t" + hiddenSize);
                writer.newLine();
                writeVector(writer, "b1", b1);
                for (int j = 0; j < hiddenSize; j++) {
                    writeVector(writer, "w1", w1[j]);
                }
                writeVector(writer, "w2", w2);
                writer.write("b2\t" + b2);
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not save map-neural value model to " + path, e);
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
}

final class MapNeuralPolicyFunction implements PolicyModel {

    private static final int INPUTS = MapStateFeatures.FEATURE_COUNT;
    private static final int ACTIONS = Types.ACTION.values().length;
    private static final int DEFAULT_HIDDEN = 32;
    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[][] w2;
    private final double[] b2;
    private boolean trained;

    private MapNeuralPolicyFunction() {
        this(DEFAULT_HIDDEN, 20260419L);
    }

    private MapNeuralPolicyFunction(int hiddenSize, long seed) {
        this.hiddenSize = hiddenSize;
        this.w1 = new double[hiddenSize][INPUTS];
        this.b1 = new double[hiddenSize];
        this.w2 = new double[ACTIONS][hiddenSize];
        this.b2 = new double[ACTIONS];
        initialise(seed);
    }

    private void initialise(long seed) {
        Random rnd = new Random(seed);
        double inputScale = 1.0 / Math.sqrt(INPUTS);
        double hiddenScale = 1.0 / Math.sqrt(hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < INPUTS; i++) {
                w1[j][i] = rnd.nextGaussian() * inputScale * 0.08;
            }
        }
        for (int action = 0; action < ACTIONS; action++) {
            for (int j = 0; j < hiddenSize; j++) {
                w2[action][j] = rnd.nextGaussian() * hiddenScale * 0.08;
            }
        }
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType) {
        return predict(MapStateFeatures.extract(state, playerID, allIds))[actionType.ordinal()];
    }

    @Override
    public double[] predict(double[] features) {
        return predictFromHidden(hidden(features));
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

    private double[] predictFromHidden(double[] hidden) {
        double[] logits = new double[ACTIONS];
        double max = -Double.MAX_VALUE;
        for (int action = 0; action < ACTIONS; action++) {
            double z = b2[action];
            for (int j = 0; j < hiddenSize; j++) {
                z += w2[action][j] * hidden[j];
            }
            logits[action] = z;
            max = Math.max(max, z);
        }

        double sum = 0.0;
        for (int action = 0; action < ACTIONS; action++) {
            logits[action] = Math.exp(logits[action] - max);
            sum += logits[action];
        }
        if (sum <= 0.0) {
            double uniform = 1.0 / ACTIONS;
            for (int action = 0; action < ACTIONS; action++) {
                logits[action] = uniform;
            }
            return logits;
        }
        for (int action = 0; action < ACTIONS; action++) {
            logits[action] /= sum;
        }
        return logits;
    }

    @Override
    public double train(ArrayList<PolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
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

                double[] hidden = hidden(example.features);
                double[] probs = predictFromHidden(hidden);
                double[] error = new double[ACTIONS];
                for (int action = 0; action < ACTIONS; action++) {
                    double target = example.targetFor(action, ACTIONS);
                    error[action] = probs[action] - target;
                    if (target > 0.0) {
                        loss += -target * Math.log(Math.max(1e-9, probs[action]));
                    }
                }

                double[][] oldW2 = copy(w2);
                for (int action = 0; action < ACTIONS; action++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        double grad = error[action] * hidden[j] + l2 * w2[action][j];
                        w2[action][j] -= learningRate * grad;
                    }
                    b2[action] -= learningRate * error[action];
                }

                for (int j = 0; j < hiddenSize; j++) {
                    double dh = 0.0;
                    for (int action = 0; action < ACTIONS; action++) {
                        dh += error[action] * oldW2[action][j];
                    }
                    dh *= 1.0 - hidden[j] * hidden[j];
                    for (int i = 0; i < INPUTS; i++) {
                        double grad = dh * example.features[i] + l2 * w1[j][i];
                        w1[j][i] -= learningRate * grad;
                    }
                    b1[j] -= learningRate * dh;
                }
                trainedRows++;
            }
            lastLoss = trainedRows > 0 ? loss / trainedRows : Double.NaN;
        }

        trained = true;
        return lastLoss;
    }

    static MapNeuralPolicyFunction load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return new MapNeuralPolicyFunction();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Map Neural Polytopia policy model")) {
                return new MapNeuralPolicyFunction();
            }
            String featureLine = reader.readLine();
            if (featureLine == null || !featureLine.equals("features\t" + INPUTS)) {
                return new MapNeuralPolicyFunction();
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new MapNeuralPolicyFunction();
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            MapNeuralPolicyFunction fn = new MapNeuralPolicyFunction(hidden, 20260419L);
            readVector(reader.readLine(), fn.b1, "b1");
            for (int j = 0; j < hidden; j++) {
                readVector(reader.readLine(), fn.w1[j], "w1");
            }
            for (int action = 0; action < ACTIONS; action++) {
                readVector(reader.readLine(), fn.w2[action], "w2");
            }
            readVector(reader.readLine(), fn.b2, "b2");
            fn.trained = true;
            return fn;
        } catch (IOException e) {
            throw new RuntimeException("Could not load map-neural policy model from " + path, e);
        }
    }

    @Override
    public void save(String path) {
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write("# Map Neural Polytopia policy model");
                writer.newLine();
                writer.write("features\t" + INPUTS);
                writer.newLine();
                writer.write("hidden\t" + hiddenSize);
                writer.newLine();
                writeVector(writer, "b1", b1);
                for (int j = 0; j < hiddenSize; j++) {
                    writeVector(writer, "w1", w1[j]);
                }
                for (int action = 0; action < ACTIONS; action++) {
                    writeVector(writer, "w2", w2[action]);
                }
                writeVector(writer, "b2", b2);
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not save map-neural policy model to " + path, e);
        }
    }

    private static double[][] copy(double[][] source) {
        double[][] out = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            out[i] = source[i].clone();
        }
        return out;
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
}
