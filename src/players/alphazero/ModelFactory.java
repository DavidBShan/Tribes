package players.alphazero;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public final class ModelFactory {

    public static final String LINEAR = "linear";
    public static final String NEURAL = "neural";
    public static final String SHARED_NEURAL = "shared-neural";
    public static final String MAP_NEURAL = "map-neural";
    public static final String MAP_SHARED_NEURAL = "map-shared-neural";
    public static final String MAP_LINEAR = "map-linear";
    public static final String ACTION_NEURAL = "action-neural";
    private static final Map<String, SharedNeuralCore> SHARED_NEURAL_CACHE = new HashMap<>();
    private static final Map<String, MapSharedNeuralCore> MAP_SHARED_NEURAL_CACHE = new HashMap<>();

    private ModelFactory() {
    }

    public static ValueModel loadValue(String type, String path) {
        if (SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new SharedNeuralValueFunction(sharedNeuralCore(path));
        }
        if (MAP_NEURAL.equalsIgnoreCase(type)) {
            return MapNeuralValueFunction.load(path);
        }
        if (MAP_SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new MapSharedNeuralValueFunction(mapSharedNeuralCore(path));
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
        if (MAP_NEURAL.equalsIgnoreCase(type)) {
            return MapNeuralPolicyFunction.load(path);
        }
        if (MAP_SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new MapSharedNeuralPolicyFunction(mapSharedNeuralCore(path));
        }
        if (NEURAL.equalsIgnoreCase(type)) {
            return NeuralPolicyFunction.load(path);
        }
        return LinearPolicyFunction.load(path);
    }

    public static ActionPolicyModel loadActionPolicy(String path) {
        return LinearActionPolicyFunction.load(path);
    }

    public static ActionPolicyModel loadActionPolicy(String type, String path) {
        if (MAP_LINEAR.equalsIgnoreCase(type)) {
            return LinearActionPolicyFunction.load(type, path);
        }
        if (MAP_NEURAL.equalsIgnoreCase(type)) {
            return NeuralActionPolicyFunction.load(type, path);
        }
        if (ACTION_NEURAL.equalsIgnoreCase(type)) {
            return NeuralActionPolicyFunction.load(type, path);
        }
        if (MAP_SHARED_NEURAL.equalsIgnoreCase(type)) {
            return new MapSharedNeuralActionPolicyFunction(mapSharedNeuralCore(path));
        }
        return LinearActionPolicyFunction.load(type, path);
    }

    public static boolean isSharedNeural(String type) {
        return SHARED_NEURAL.equalsIgnoreCase(type);
    }

    static SharedNeuralCore loadSharedNeuralCore(String path) {
        return sharedNeuralCore(path);
    }

    public static synchronized void clearCaches() {
        SHARED_NEURAL_CACHE.clear();
        MAP_SHARED_NEURAL_CACHE.clear();
    }

    private static synchronized SharedNeuralCore sharedNeuralCore(String path) {
        String key = modelGroupKey(path);
        SharedNeuralCore core = SHARED_NEURAL_CACHE.get(key);
        if (core == null) {
            core = SharedNeuralCore.load(path);
            SHARED_NEURAL_CACHE.put(key, core);
        }
        return core;
    }

    private static synchronized MapSharedNeuralCore mapSharedNeuralCore(String path) {
        String key = modelGroupKey(path);
        MapSharedNeuralCore core = MAP_SHARED_NEURAL_CACHE.get(key);
        if (core == null) {
            core = MapSharedNeuralCore.load(path);
            MAP_SHARED_NEURAL_CACHE.put(key, core);
        }
        return core;
    }

    private static String modelGroupKey(String path) {
        File file = new File(path == null ? "" : path);
        String key = file.getAbsolutePath();
        String[] suffixes = {
                "-action-policy.tsv",
                "-policy.tsv",
                "-value.tsv"
        };
        for (String suffix : suffixes) {
            if (key.endsWith(suffix)) {
                return key.substring(0, key.length() - suffix.length());
            }
        }
        return key;
    }
}
