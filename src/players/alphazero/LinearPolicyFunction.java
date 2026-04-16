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

public class LinearPolicyFunction {

    private final double[][] weights;
    private boolean trained;

    public LinearPolicyFunction() {
        this.weights = new double[Types.ACTION.values().length][StateFeatures.FEATURE_COUNT];
        this.trained = false;
    }

    public boolean isTrained() {
        return trained;
    }

    public double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType) {
        double[] probs = predict(StateFeatures.extract(state, playerID, allIds));
        return probs[actionType.ordinal()];
    }

    public double[] predict(double[] features) {
        double[] logits = new double[weights.length];
        double max = -Double.MAX_VALUE;
        for (int action = 0; action < weights.length; action++) {
            double z = 0.0;
            for (int i = 0; i < features.length; i++) {
                z += weights[action][i] * features[i];
            }
            logits[action] = z;
            if (z > max) {
                max = z;
            }
        }

        double sum = 0.0;
        for (int action = 0; action < logits.length; action++) {
            logits[action] = Math.exp(logits[action] - max);
            sum += logits[action];
        }
        if (sum <= 0.0) {
            double uniform = 1.0 / logits.length;
            for (int action = 0; action < logits.length; action++) {
                logits[action] = uniform;
            }
            return logits;
        }

        for (int action = 0; action < logits.length; action++) {
            logits[action] /= sum;
        }
        return logits;
    }

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
                if (!example.hasValidTarget(weights.length)) {
                    continue;
                }

                double[] probs = predict(example.features);
                for (int action = 0; action < weights.length; action++) {
                    double target = example.targetFor(action, weights.length);
                    if (target > 0.0) {
                        loss += -target * Math.log(Math.max(1e-9, probs[action]));
                    }
                }

                for (int action = 0; action < weights.length; action++) {
                    double target = example.targetFor(action, weights.length);
                    double error = probs[action] - target;
                    for (int i = 0; i < StateFeatures.FEATURE_COUNT; i++) {
                        double grad = error * example.features[i] + l2 * weights[action][i];
                        weights[action][i] -= learningRate * grad;
                    }
                }
            }

            lastLoss = loss / examples.size();
        }

        trained = true;
        return lastLoss;
    }

    public static LinearPolicyFunction load(String path) {
        LinearPolicyFunction fn = new LinearPolicyFunction();
        File file = new File(path);
        if (!file.exists()) {
            return fn;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            int row = 0;
            while ((line = reader.readLine()) != null && row < fn.weights.length) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != StateFeatures.FEATURE_COUNT) {
                    continue;
                }

                for (int i = 0; i < parts.length; i++) {
                    fn.weights[row][i] = Double.parseDouble(parts[i]);
                }
                row++;
            }
            reader.close();
            fn.trained = row == fn.weights.length;
        } catch (IOException e) {
            throw new RuntimeException("Could not load policy model from " + path, e);
        }

        return fn;
    }

    public void save(String path) {
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.write("# Linear Polytopia policy model. Rows follow Types.ACTION ordinal order.");
            writer.newLine();
            for (int action = 0; action < weights.length; action++) {
                for (int i = 0; i < weights[action].length; i++) {
                    if (i > 0) {
                        writer.write('\t');
                    }
                    writer.write(Double.toString(weights[action][i]));
                }
                writer.newLine();
            }
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not save policy model to " + path, e);
        }
    }
}
