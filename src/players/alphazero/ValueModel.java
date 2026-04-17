package players.alphazero;

import core.game.GameState;

import java.util.ArrayList;

public interface ValueModel {
    boolean isTrained();

    double predict(GameState state, int playerID, ArrayList<Integer> allIds);

    double predict(double[] features);

    double train(ArrayList<ValueTrainingExample> examples, int epochs, double learningRate, double l2, long seed);

    void save(String path);
}
