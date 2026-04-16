package players.alphazero;

public class ValueTrainingExample {
    public final double label;
    public final double[] features;

    public ValueTrainingExample(double label, double[] features) {
        this.label = label;
        this.features = features;
    }
}
