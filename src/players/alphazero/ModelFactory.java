package players.alphazero;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public final class ModelFactory {

    public static final String LINEAR = "linear";
    public static final String NEURAL = "neural";
    public static final String SHARED_NEURAL = "shared-neural";
    private static final Map<String, SharedNeuralCore> SHARED_NEURAL_CACHE = new HashMap<>();

    private ModelFactory() {
    }

    public static ValueModel loadValue(String type, String path) {
        if (SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new SharedNeuralValueFunction(sharedNeuralCore(path));
        }
        if (NEURAL.equalsIgnoreCase(type)) {
            return NeuralValueFunction.load(path);
        }
        return LinearValueFunction.load(path);
    }

    public static PolicyModel loadPolicy(String type, String path) {
        if (SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new SharedNeuralPolicyFunction(sharedNeuralCore(path));
        }
        if (NEURAL.equalsIgnoreCase(type)) {
            return NeuralPolicyFunction.load(path);
        }
        return LinearPolicyFunction.load(path);
    }

    public static ActionPolicyModel loadActionPolicy(String path) {
        return LinearActionPolicyFunction.load(path);
    }

    public static boolean isSharedNeural(String type) {
        return SHARED_NEURAL.equalsIgnoreCase(type);
    }

    static SharedNeuralCore loadSharedNeuralCore(String path) {
        return sharedNeuralCore(path);
    }

    private static synchronized SharedNeuralCore sharedNeuralCore(String path) {
        File file = new File(path);
        File parent = file.getAbsoluteFile().getParentFile();
        String key = parent == null ? file.getAbsolutePath() : parent.getAbsolutePath();
        SharedNeuralCore core = SHARED_NEURAL_CACHE.get(key);
        if (core == null) {
            core = SharedNeuralCore.load(path);
            SHARED_NEURAL_CACHE.put(key, core);
        }
        return core;
    }
}
