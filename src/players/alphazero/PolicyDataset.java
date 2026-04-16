package players.alphazero;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
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
                writer.write("action");
                for (int i = 0; i < StateFeatures.FEATURE_COUNT; i++) {
                    writer.write('\t');
                    writer.write("f" + i);
                }
                writer.newLine();
            }

            for (PolicyTrainingExample example : examples) {
                writer.write(Integer.toString(example.actionType));
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
        ArrayList<PolicyTrainingExample> examples = new ArrayList<>();
        File file = new File(path);
        if (!file.exists()) {
            return examples;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty() || line.startsWith("action")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != StateFeatures.FEATURE_COUNT + 1) {
                    continue;
                }

                int actionType = Integer.parseInt(parts[0]);
                double[] features = new double[StateFeatures.FEATURE_COUNT];
                for (int i = 0; i < features.length; i++) {
                    features[i] = Double.parseDouble(parts[i + 1]);
                }
                examples.add(new PolicyTrainingExample(actionType, features));

                if (maxExamples > 0 && examples.size() >= maxExamples) {
                    break;
                }
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load policy training data from " + path, e);
        }

        return examples;
    }
}
