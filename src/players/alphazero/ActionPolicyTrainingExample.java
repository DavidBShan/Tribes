package players.alphazero;

public class ActionPolicyTrainingExample {
    public final double target;
    public final double[] features;

    public ActionPolicyTrainingExample(double target, double[] features) {
        this.target = Math.max(0.0, Math.min(1.0, target));
        this.features = features;
    }
}
