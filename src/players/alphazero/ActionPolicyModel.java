package players.alphazero;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

public interface ActionPolicyModel {
    boolean isTrained();

    double logit(GameState state, GameState nextState, int playerID, ArrayList<Integer> allIds, Action action);

    double train(ArrayList<ActionPolicyTrainingExample> examples, int epochs, double learningRate, double l2, long seed);

    void save(String path);
}
