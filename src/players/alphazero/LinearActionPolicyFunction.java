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

public class LinearActionPolicyFunction implements ActionPolicyModel {

    private final double[] weights;
    private boolean trained;

    public LinearActionPolicyFunction() {
        this.weights = new double[ActionFeatures.FEATURE_COUNT];
        this.trained = false;
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double logit(GameState state, GameState nextState, int playerID, ArrayList<Integer> allIds, Action action) {
        return dot(ActionFeatures.extract(state, nextState, playerID, allIds, action));
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
                if (example.features == null || example.features.length != weights.length) {
                    continue;
                }

                double prediction = sigmoid(dot(example.features));
                double error = prediction - example.target;
                loss += -(example.target * Math.log(Math.max(1e-9, prediction))
                        + (1.0 - example.target) * Math.log(Math.max(1e-9, 1.0 - prediction)));
                for (int i = 0; i < weights.length; i++) {
                    double grad = error * example.features[i] + l2 * weights[i];
                    weights[i] -= learningRate * grad;
                }
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    private double dot(double[] features) {
        double z = 0.0;
        for (int i = 0; i < Math.min(features.length, weights.length); i++) {
            z += weights[i] * features[i];
        }
        return z;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            double z = Math.exp(-value);
            return 1.0 / (1.0 + z);
        }
        double z = Math.exp(value);
        return z / (1.0 + z);
    }

    public static LinearActionPolicyFunction load(String path) {
        LinearActionPolicyFunction fn = new LinearActionPolicyFunction();
        if (path == null || path.trim().isEmpty()) {
            return fn;
        }
        File file = new File(path);
        if (!file.exists()) {
            return fn;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != fn.weights.length) {
                    continue;
                }
                for (int i = 0; i < parts.length; i++) {
                    fn.weights[i] = Double.parseDouble(parts[i]);
                }
                fn.trained = true;
                break;
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load action policy model from " + path, e);
        }
        return fn;
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

            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.write("# Linear Polytopia legal-action policy model");
            writer.newLine();
            for (int i = 0; i < weights.length; i++) {
                if (i > 0) {
                    writer.write('\t');
                }
                writer.write(Double.toString(weights[i]));
            }
            writer.newLine();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not save action policy model to " + path, e);
        }
    }
}
