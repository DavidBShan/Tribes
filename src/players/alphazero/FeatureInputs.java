package players.alphazero;

import core.game.GameState;

import java.util.ArrayList;

final class FeatureInputs {

    private FeatureInputs() {
    }

    static boolean usesMap(String networkType) {
        return ModelFactory.MAP_NEURAL.equalsIgnoreCase(networkType)
                || ModelFactory.MAP_SHARED_NEURAL.equalsIgnoreCase(networkType);
    }

    static int featureCount(String networkType) {
        return usesMap(networkType) ? MapStateFeatures.FEATURE_COUNT : StateFeatures.FEATURE_COUNT;
    }

    static double[] extract(String networkType, GameState state, int playerID, ArrayList<Integer> allIds) {
        return usesMap(networkType)
                ? MapStateFeatures.extract(state, playerID, allIds)
                : StateFeatures.extract(state, playerID, allIds);
    }

    static String header(int featureCount) {
        StringBuilder sb = new StringBuilder("label");
        for (int i = 0; i < featureCount; i++) {
            sb.append('\t').append("f").append(i);
        }
        return sb.toString();
    }

    static String policyHeader(int featureCount) {
        StringBuilder sb = new StringBuilder("action");
        for (int i = 0; i < featureCount; i++) {
            sb.append('\t').append("f").append(i);
        }
        return sb.toString();
    }
}
