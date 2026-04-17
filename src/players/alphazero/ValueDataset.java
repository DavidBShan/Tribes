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

public final class ValueDataset {

    private ValueDataset() {
    }

    public static synchronized void append(String path, List<ValueTrainingExample> examples) {
        if (examples == null || examples.isEmpty()) {
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
                writer.write(StateFeatures.header());
                writer.newLine();
            }

            for (ValueTrainingExample example : examples) {
                writer.write(Double.toString(example.label));
                for (double feature : example.features) {
                    writer.write('\t');
                    writer.write(Double.toString(feature));
                }
                writer.newLine();
            }
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not append value training data to " + path, e);
        }
    }

    public static ArrayList<ValueTrainingExample> load(String path, int maxExamples) {
        ArrayDeque<ValueTrainingExample> examples = new ArrayDeque<>();
        File file = new File(path);
        if (!file.exists()) {
            return new ArrayList<>();
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty() || line.startsWith("label")) {
                    continue;
                }

                String[] parts = line.split("\\t");
                if (parts.length != StateFeatures.FEATURE_COUNT + 1) {
                    continue;
                }

                double label = Double.parseDouble(parts[0]);
                double[] features = new double[StateFeatures.FEATURE_COUNT];
                for (int i = 0; i < features.length; i++) {
                    features[i] = Double.parseDouble(parts[i + 1]);
                }
                examples.addLast(new ValueTrainingExample(label, features));

                if (maxExamples > 0 && examples.size() > maxExamples) {
                    examples.removeFirst();
                }
            }
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not load value training data from " + path, e);
        }

        return new ArrayList<>(examples);
    }
}
