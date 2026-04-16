package players.alphazero;

public class ValueTrainingExample {
    public final double label;
    public final double[] features;
    public final double positionLabel;

    public ValueTrainingExample(double label, double[] features) {
        this(label, features, 0.0);
    }

    public ValueTrainingExample(double label, double[] features, double positionLabel) {
        this.label = label;
        this.features = features;
        this.positionLabel = positionLabel;
    }
}
