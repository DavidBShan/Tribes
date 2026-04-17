package players.alphazero;

public final class ModelFactory {

    public static final String LINEAR = "linear";
    public static final String NEURAL = "neural";

    private ModelFactory() {
    }

    public static ValueModel loadValue(String type, String path) {
        if (NEURAL.equalsIgnoreCase(type)) {
            return NeuralValueFunction.load(path);
        }
        return LinearValueFunction.load(path);
    }

    public static PolicyModel loadPolicy(String type, String path) {
        if (NEURAL.equalsIgnoreCase(type)) {
            return NeuralPolicyFunction.load(path);
        }
        return LinearPolicyFunction.load(path);
    }

    public static ActionPolicyModel loadActionPolicy(String path) {
        return LinearActionPolicyFunction.load(path);
    }
}
