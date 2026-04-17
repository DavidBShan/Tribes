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

public class NeuralPolicyFunction implements PolicyModel {

    private static final int INPUTS = StateFeatures.FEATURE_COUNT;
    private static final int ACTIONS = Types.ACTION.values().length;
    private static final int DEFAULT_HIDDEN = 64;
    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[][] w2;
    private final double[] b2;
    private boolean trained;

    public NeuralPolicyFunction() {
        this(DEFAULT_HIDDEN, 20260418L);
    }

    public NeuralPolicyFunction(int hiddenSize, long seed) {
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
                w1[j][i] = rnd.nextGaussian() * inputScale * 0.10;
            }
        }
        for (int action = 0; action < ACTIONS; action++) {
            for (int j = 0; j < hiddenSize; j++) {
                w2[action][j] = rnd.nextGaussian() * hiddenScale * 0.10;
            }
        }
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType) {
        return predict(StateFeatures.extract(state, playerID, allIds))[actionType.ordinal()];
    }

    @Override
    public double[] predict(double[] features) {
        double[] hidden = hidden(features);
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
    public double train(ArrayList<PolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(examples, rnd);
            double loss = 0.0;
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
            }
            lastLoss = loss / examples.size();
        }
        trained = true;
        return lastLoss;
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
        for (int action = 0; action < ACTIONS; action++) {
            logits[action] = sum > 0.0 ? logits[action] / sum : 1.0 / ACTIONS;
        }
        return logits;
    }

    private static double[][] copy(double[][] input) {
        double[][] out = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            out[i] = input[i].clone();
        }
        return out;
    }

    public static NeuralPolicyFunction load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return new NeuralPolicyFunction();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Neural Polytopia policy model")) {
                return new NeuralPolicyFunction();
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new NeuralPolicyFunction();
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            NeuralPolicyFunction fn = new NeuralPolicyFunction(hidden, 20260418L);
            readVector(reader.readLine(), fn.b1, "b1");
            for (int j = 0; j < hidden; j++) {
                readVector(reader.readLine(), fn.w1[j], "w1");
            }
            readVector(reader.readLine(), fn.b2, "b2");
            for (int action = 0; action < ACTIONS; action++) {
                readVector(reader.readLine(), fn.w2[action], "w2");
            }
            fn.trained = true;
            return fn;
        } catch (IOException e) {
            throw new RuntimeException("Could not load neural policy model from " + path, e);
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

    @Override
    public void save(String path) {
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write("# Neural Polytopia policy model. Features: " + INPUTS);
                writer.newLine();
                writer.write("hidden\t" + hiddenSize);
                writer.newLine();
                writeVector(writer, "b1", b1);
                for (int j = 0; j < hiddenSize; j++) {
                    writeVector(writer, "w1", w1[j]);
                }
                writeVector(writer, "b2", b2);
                for (int action = 0; action < ACTIONS; action++) {
                    writeVector(writer, "w2", w2[action]);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not save neural policy model to " + path, e);
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
