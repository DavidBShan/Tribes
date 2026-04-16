package players.alphazero;

public class PolicyTrainingExample {
    public final int actionType;
    public final double[] features;

    public PolicyTrainingExample(int actionType, double[] features) {
        this.actionType = actionType;
        this.features = features;
    }
}
