package players.alphazero;

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

public final class NeuralActionPolicyFunction implements ActionPolicyModel {

    private static final int DEFAULT_HIDDEN = 48;

    private final String networkType;
    private final int inputs;
    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private boolean trained;

    private NeuralActionPolicyFunction(String networkType) {
        this(networkType, DEFAULT_HIDDEN, 20260419L);
    }

    private NeuralActionPolicyFunction(String networkType, int hiddenSize, long seed) {
        this.networkType = networkType == null ? ModelFactory.LINEAR : networkType;
        this.inputs = ActionFeatureInputs.featureCount(this.networkType);
        this.hiddenSize = hiddenSize;
        this.w1 = new double[hiddenSize][inputs];
        this.b1 = new double[hiddenSize];
        this.w2 = new double[hiddenSize];
        initialise(seed);
    }

    private void initialise(long seed) {
        Random rnd = new Random(seed);
        double inputScale = 1.0 / Math.sqrt(inputs);
        double hiddenScale = 1.0 / Math.sqrt(hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int i = 0; i < inputs; i++) {
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
    public double logit(GameState state, GameState nextState, int playerID,
                        ArrayList<Integer> allIds, Action action) {
        double[] features = ActionFeatureInputs.extract(networkType, state, nextState, playerID, allIds, action);
        return rawLogit(features);
    }

    @Override
    public double train(ArrayList<ActionPolicyTrainingExample> examples, int epochs,
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
            for (ActionPolicyTrainingExample example : examples) {
                if (example.features == null || example.features.length != inputs) {
                    continue;
                }
                loss += update(example, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    private double update(ActionPolicyTrainingExample example, double learningRate, double l2) {
        double[] hidden = hidden(example.features);
        double z = b2;
        for (int j = 0; j < hiddenSize; j++) {
            z += w2[j] * hidden[j];
        }
        double prediction = sigmoid(z);
        double error = prediction - example.target;
        double loss = -(example.target * Math.log(Math.max(1e-9, prediction))
                + (1.0 - example.target) * Math.log(Math.max(1e-9, 1.0 - prediction)));

        double[] oldW2 = w2.clone();
        for (int j = 0; j < hiddenSize; j++) {
            double grad = error * hidden[j] + l2 * w2[j];
            w2[j] -= learningRate * grad;
        }
        b2 -= learningRate * error;

        for (int j = 0; j < hiddenSize; j++) {
            double dh = error * oldW2[j] * (1.0 - hidden[j] * hidden[j]);
            for (int i = 0; i < inputs; i++) {
                double grad = dh * example.features[i] + l2 * w1[j][i];
                w1[j][i] -= learningRate * grad;
            }
            b1[j] -= learningRate * dh;
        }
        return loss;
    }

    private double rawLogit(double[] features) {
        double[] hidden = hidden(features);
        double z = b2;
        for (int j = 0; j < hiddenSize; j++) {
            z += w2[j] * hidden[j];
        }
        return z;
    }

    private double[] hidden(double[] features) {
        double[] hidden = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double z = b1[j];
            for (int i = 0; i < inputs; i++) {
                z += w1[j][i] * features[i];
            }
            hidden[j] = Math.tanh(z);
        }
        return hidden;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            double z = Math.exp(-value);
            return 1.0 / (1.0 + z);
        }
        double z = Math.exp(value);
        return z / (1.0 + z);
    }

    public static NeuralActionPolicyFunction load(String networkType, String path) {
        if (path == null || path.trim().isEmpty()) {
            return new NeuralActionPolicyFunction(networkType);
        }
        File file = new File(path);
        if (!file.exists()) {
            return new NeuralActionPolicyFunction(networkType);
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Neural Polytopia legal-action policy model")) {
                return new NeuralActionPolicyFunction(networkType);
            }
            String networkLine = reader.readLine();
            if (networkLine == null || !networkLine.startsWith("networkType\t")) {
                return new NeuralActionPolicyFunction(networkType);
            }
            String storedNetworkType = networkLine.split("\\t", 2)[1];
            String featureLine = reader.readLine();
            if (featureLine == null || !featureLine.startsWith("features\t")) {
                return new NeuralActionPolicyFunction(networkType);
            }
            int storedInputs = Integer.parseInt(featureLine.split("\\t")[1]);
            if (storedInputs != ActionFeatureInputs.featureCount(networkType)) {
                return new NeuralActionPolicyFunction(networkType);
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new NeuralActionPolicyFunction(networkType);
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            NeuralActionPolicyFunction fn = new NeuralActionPolicyFunction(storedNetworkType, hidden, 20260419L);
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
            throw new RuntimeException("Could not load neural action policy from " + path, e);
        }
    }

    @Override
    public void save(String path) {
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
                writer.write("# Neural Polytopia legal-action policy model");
                writer.newLine();
                writer.write("networkType\t" + networkType);
                writer.newLine();
                writer.write("features\t" + inputs);
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
            throw new RuntimeException("Could not save neural action policy to " + path, e);
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
