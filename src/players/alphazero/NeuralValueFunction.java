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

public class NeuralValueFunction implements ValueModel {

    private static final int INPUTS = StateFeatures.FEATURE_COUNT;
    private static final int DEFAULT_HIDDEN = 64;
    private final int hiddenSize;
    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private boolean trained;

    public NeuralValueFunction() {
        this(DEFAULT_HIDDEN, 20260417L);
    }

    public NeuralValueFunction(int hiddenSize, long seed) {
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
                w1[j][i] = rnd.nextGaussian() * inputScale * 0.10;
            }
            w2[j] = rnd.nextGaussian() * hiddenScale * 0.10;
        }
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public double predict(GameState state, int playerID, ArrayList<Integer> allIds) {
        return predict(StateFeatures.extract(state, playerID, allIds));
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

    public static NeuralValueFunction load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return new NeuralValueFunction();
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("# Neural Polytopia value model")) {
                return new NeuralValueFunction();
            }
            String hiddenLine = reader.readLine();
            if (hiddenLine == null || !hiddenLine.startsWith("hidden\t")) {
                return new NeuralValueFunction();
            }
            int hidden = Integer.parseInt(hiddenLine.split("\\t")[1]);
            NeuralValueFunction fn = new NeuralValueFunction(hidden, 20260417L);
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
            throw new RuntimeException("Could not load neural value model from " + path, e);
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
                writer.write("# Neural Polytopia value model. Features: " + INPUTS);
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
            throw new RuntimeException("Could not save neural value model to " + path, e);
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
