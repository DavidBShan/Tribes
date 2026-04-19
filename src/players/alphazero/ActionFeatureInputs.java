package players.alphazero;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

final class ActionFeatureInputs {

    private ActionFeatureInputs() {
    }

    static boolean usesMap(String networkType) {
        return ModelFactory.MAP_NEURAL.equalsIgnoreCase(networkType);
    }

    static int featureCount(String networkType) {
        return usesMap(networkType) ? MapActionFeatures.FEATURE_COUNT : ActionFeatures.FEATURE_COUNT;
    }

    static double[] extract(String networkType, GameState state, GameState nextState,
                            int playerID, ArrayList<Integer> allIds, Action action) {
        return usesMap(networkType)
                ? MapActionFeatures.extract(state, nextState, playerID, allIds, action)
                : ActionFeatures.extract(state, nextState, playerID, allIds, action);
    }

    static String header(String networkType) {
        return usesMap(networkType) ? MapActionFeatures.header() : ActionFeatures.header();
    }
}
