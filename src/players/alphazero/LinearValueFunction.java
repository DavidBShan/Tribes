package players.alphazero;

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

public class LinearValueFunction implements ValueModel {

    private final double[] weights;
    private boolean trained;

    public LinearValueFunction() {
        this.weights = new double[StateFeatures.FEATURE_COUNT];
        this.trained = false;
    }

    public boolean isTrained() {
        return trained;
    }

    public double predict(GameState state, int playerID, ArrayList<Integer> allIds) {
        return predict(StateFeatures.extract(state, playerID, allIds));
    }

    public double predict(double[] features) {
        double z = 0.0;
        int len = Math.min(features.length, weights.length);
        for (int i = 0; i < len; i++) {
            z += weights[i] * features[i];
        }
        return Math.tanh(z);
    }

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
                double pred = predict(example.features);
                double error = pred - example.label;
                loss += error * error;

                double dz = error * (1.0 - pred * pred);
                for (int i = 0; i < weights.length; i++) {
                    double grad = dz * example.features[i] + l2 * weights[i];
                    weights[i] -= learningRate * grad;
                }
            }

            lastLoss = loss / examples.size();
        }

        trained = true;
        return lastLoss;
    }

    public static LinearValueFunction load(String path) {
        LinearValueFunction fn = new LinearValueFunction();
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
                if (parts.length == fn.weights.length) {
                    for (int i = 0; i < parts.length; i++) {
                        fn.weights[i] = Double.parseDouble(parts[i]);
                    }
                    fn.trained = true;
                    break;
                }
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load value model from " + path, e);
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
            writer.write("# Linear Polytopia value model. Features: " + StateFeatures.FEATURE_COUNT);
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
            throw new RuntimeException("Could not save value model to " + path, e);
        }
    }
}
