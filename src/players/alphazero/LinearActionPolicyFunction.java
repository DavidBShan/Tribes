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
    private final String networkType;
    private boolean trained;

    public LinearActionPolicyFunction() {
        this(ModelFactory.LINEAR);
    }

    private LinearActionPolicyFunction(String networkType) {
        this.networkType = networkType == null ? ModelFactory.LINEAR : networkType;
        this.weights = new double[ActionFeatureInputs.featureCount(this.networkType)];
        this.trained = false;
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double logit(GameState state, GameState nextState, int playerID, ArrayList<Integer> allIds, Action action) {
        return dot(ActionFeatureInputs.extract(networkType, state, nextState, playerID, allIds, action));
    }

    @Override
    public double train(ArrayList<ActionPolicyTrainingExample> examples, int epochs,
                        double learningRate, double l2, long seed) {
        if (examples == null || examples.isEmpty()) {
            return Double.NaN;
        }

        Random rnd = new Random(seed);
        ArrayList<ArrayList<ActionPolicyTrainingExample>> groups =
                ActionPolicyTrainingExample.groupedBatches(examples, weights.length);
        if (!groups.isEmpty()) {
            return trainGrouped(groups, epochs, learningRate, l2, rnd);
        }

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

    private double trainGrouped(ArrayList<ArrayList<ActionPolicyTrainingExample>> groups, int epochs,
                                double learningRate, double l2, Random rnd) {
        double lastLoss = Double.NaN;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(groups, rnd);
            double loss = 0.0;
            int used = 0;
            for (ArrayList<ActionPolicyTrainingExample> group : groups) {
                loss += updateGroup(group, learningRate, l2);
                used++;
            }
            lastLoss = used == 0 ? Double.NaN : loss / used;
        }
        trained = true;
        return lastLoss;
    }

    private double updateGroup(ArrayList<ActionPolicyTrainingExample> group,
                               double learningRate, double l2) {
        double max = -Double.MAX_VALUE;
        double[] logits = new double[group.size()];
        for (int i = 0; i < group.size(); i++) {
            logits[i] = dot(group.get(i).features);
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
            double error = prediction - target;
            for (int f = 0; f < weights.length; f++) {
                double grad = error * example.features[f] + l2 * weights[f];
                weights[f] -= learningRate * grad;
            }
        }
        return loss;
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
        return load(ModelFactory.LINEAR, path);
    }

    public static LinearActionPolicyFunction load(String networkType, String path) {
        LinearActionPolicyFunction fn = new LinearActionPolicyFunction(networkType);
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
            writer.write("# networkType\t" + networkType);
            writer.newLine();
            writer.write("# features\t" + weights.length);
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
