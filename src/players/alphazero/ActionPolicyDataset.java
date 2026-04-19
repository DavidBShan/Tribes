package players.alphazero;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

public final class ActionPolicyDataset {

    private ActionPolicyDataset() {
    }

    public static synchronized void append(String path, List<ActionPolicyTrainingExample> examples) {
        if (path == null || path.trim().isEmpty() || examples == null || examples.isEmpty()) {
            return;
        }

        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }

            int featureCount = featureCount(examples);
            if (featureCount <= 0) {
                return;
            }
            boolean writeHeader = !file.exists() || file.length() == 0;
            BufferedWriter writer = new BufferedWriter(new FileWriter(file, true));
            if (writeHeader) {
                writer.write(header(featureCount));
                writer.newLine();
            }

            for (ActionPolicyTrainingExample example : examples) {
                if (example.features == null || example.features.length != featureCount) {
                    continue;
                }
                writer.write(Double.toString(example.target));
                for (double feature : example.features) {
                    writer.write('\t');
                    writer.write(Double.toString(feature));
                }
                writer.newLine();
            }
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not append action-policy training data to " + path, e);
        }
    }

    public static ArrayList<ActionPolicyTrainingExample> load(String path, int maxExamples) {
        return load(path, maxExamples, ActionFeatures.FEATURE_COUNT);
    }

    public static ArrayList<ActionPolicyTrainingExample> load(String path, int maxExamples, int expectedFeatureCount) {
        ArrayDeque<ActionPolicyTrainingExample> examples = new ArrayDeque<>();
        if (path == null || path.trim().isEmpty()) {
            return new ArrayList<>();
        }
        File file = new File(path);
        if (!file.exists()) {
            return new ArrayList<>();
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty() || line.startsWith("target")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != expectedFeatureCount + 1) {
                    continue;
                }

                double target = Double.parseDouble(parts[0]);
                double[] features = new double[expectedFeatureCount];
                for (int i = 0; i < features.length; i++) {
                    features[i] = Double.parseDouble(parts[i + 1]);
                }
                examples.addLast(new ActionPolicyTrainingExample(target, features));

                if (maxExamples > 0 && examples.size() > maxExamples) {
                    examples.removeFirst();
                }
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load action-policy training data from " + path, e);
        }

        return new ArrayList<>(examples);
    }

    private static int featureCount(List<ActionPolicyTrainingExample> examples) {
        for (ActionPolicyTrainingExample example : examples) {
            if (example.features != null && example.features.length > 0) {
                return example.features.length;
            }
        }
        return 0;
    }

    private static String header(int featureCount) {
        if (featureCount == MapActionFeatures.FEATURE_COUNT) {
            return MapActionFeatures.header();
        }
        if (featureCount == ActionFeatures.FEATURE_COUNT) {
            return ActionFeatures.header();
        }
        StringBuilder sb = new StringBuilder("target");
        for (int i = 0; i < featureCount; i++) {
            sb.append('\t').append("apf").append(i);
        }
        return sb.toString();
    }
}
