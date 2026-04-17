package players.alphazero;

import core.Types;
import core.game.GameState;

import java.util.ArrayList;

public interface PolicyModel {
    boolean isTrained();

    double probability(GameState state, int playerID, ArrayList<Integer> allIds, Types.ACTION actionType);

    double[] predict(double[] features);

    double train(ArrayList<PolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed);

    void save(String path);
}
