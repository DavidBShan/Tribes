package players.alphazero;

public class PolicyTrainingExample {
    public final int actionType;
    public final double[] targetProbs;
    public final double[] features;

    public PolicyTrainingExample(int actionType, double[] features) {
        this.actionType = actionType;
        this.targetProbs = null;
        this.features = features;
    }

    public PolicyTrainingExample(double[] targetProbs, double[] features) {
        this.actionType = -1;
        this.targetProbs = normalize(targetProbs);
        this.features = features;
    }

    public double targetFor(int action, int actionCount) {
        if (targetProbs != null && action >= 0 && action < targetProbs.length) {
            return targetProbs[action];
        }
        return action == actionType ? 1.0 : 0.0;
    }

    public boolean hasValidTarget(int actionCount) {
        if (targetProbs == null) {
            return actionType >= 0 && actionType < actionCount;
        }

        double sum = 0.0;
        for (int i = 0; i < Math.min(actionCount, targetProbs.length); i++) {
            sum += targetProbs[i];
        }
        return sum > 0.0;
    }

    private static double[] normalize(double[] input) {
        if (input == null) {
            return null;
        }

        double[] out = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            out[i] = Math.max(0.0, input[i]);
            sum += out[i];
        }
        if (sum <= 0.0) {
            return out;
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }
}
