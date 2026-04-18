package players.alphazero;

import core.Types;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

public final class PolicyDataset {

    private PolicyDataset() {
    }

    public static synchronized void append(String path, List<PolicyTrainingExample> examples) {
        if (path == null || examples == null || examples.isEmpty()) {
            return;
        }

        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            boolean writeHeader = !file.exists() || file.length() == 0;
            BufferedWriter writer = new BufferedWriter(new FileWriter(file, true));
            if (writeHeader) {
                writer.write(FeatureInputs.policyHeader(examples.get(0).features.length));
                writer.newLine();
            }

            for (PolicyTrainingExample example : examples) {
                if (example.targetProbs != null) {
                    writer.write("soft:");
                    for (int i = 0; i < example.targetProbs.length; i++) {
                        if (i > 0) {
                            writer.write(',');
                        }
                        writer.write(Double.toString(example.targetProbs[i]));
                    }
                } else {
                    writer.write(Integer.toString(example.actionType));
                }
                for (double feature : example.features) {
                    writer.write('\t');
                    writer.write(Double.toString(feature));
                }
                writer.newLine();
            }
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not append policy training data to " + path, e);
        }
    }

    public static ArrayList<PolicyTrainingExample> load(String path, int maxExamples) {
        return load(path, maxExamples, StateFeatures.FEATURE_COUNT);
    }

    public static ArrayList<PolicyTrainingExample> load(String path, int maxExamples, int expectedFeatureCount) {
        ArrayDeque<PolicyTrainingExample> examples = new ArrayDeque<>();
        File file = new File(path);
        if (!file.exists()) {
            return new ArrayList<>();
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty() || line.startsWith("action")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != expectedFeatureCount + 1) {
                    continue;
                }

                double[] features = new double[expectedFeatureCount];
                for (int i = 0; i < features.length; i++) {
                    features[i] = Double.parseDouble(parts[i + 1]);
                }
                if (parts[0].startsWith("soft:")) {
                    examples.addLast(new PolicyTrainingExample(parseSoftTarget(parts[0]), features));
                } else {
                    int actionType = Integer.parseInt(parts[0]);
                    examples.addLast(new PolicyTrainingExample(actionType, features));
                }

                if (maxExamples > 0 && examples.size() > maxExamples) {
                    examples.removeFirst();
                }
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load policy training data from " + path, e);
        }

        return new ArrayList<>(examples);
    }

    private static double[] parseSoftTarget(String value) {
        double[] target = new double[Types.ACTION.values().length];
        String payload = value.substring("soft:".length());
        if (payload.trim().isEmpty()) {
            return target;
        }

        String[] parts = payload.split(",");
        for (int i = 0; i < Math.min(parts.length, target.length); i++) {
            target[i] = Double.parseDouble(parts[i]);
        }
        return target;
    }
}
