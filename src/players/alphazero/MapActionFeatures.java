package players.alphazero;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

public final class MapActionFeatures {

    public static final int FEATURE_COUNT = MapStateFeatures.FEATURE_COUNT
            + ActionFeatures.ACTION_COUNT
            + StateFeatures.FEATURE_COUNT
            + ActionFeatures.EXTRA_COUNT;

    private MapActionFeatures() {
    }

    public static double[] extract(GameState state, GameState nextState,
                                   int playerID, ArrayList<Integer> allIds, Action action) {
        double[] features = new double[FEATURE_COUNT];
        int offset = 0;

        double[] beforeMap = MapStateFeatures.extract(state, playerID, allIds);
        offset = copy(features, offset, beforeMap);

        if (action != null) {
            int ordinal = action.getActionType().ordinal();
            if (ordinal >= 0 && ordinal < ActionFeatures.ACTION_COUNT) {
                features[offset + ordinal] = 1.0;
            }
        }
        offset += ActionFeatures.ACTION_COUNT;

        double[] beforeSummary = StateFeatures.extract(state, playerID, allIds);
        double[] afterSummary = nextState == null
                ? beforeSummary : StateFeatures.extract(nextState, playerID, allIds);
        for (int i = 0; i < StateFeatures.FEATURE_COUNT; i++) {
            features[offset + i] = afterSummary[i] - beforeSummary[i];
        }
        offset += StateFeatures.FEATURE_COUNT;

        ActionFeatures.fillExtras(features, offset, state, nextState, playerID, allIds, action);
        return features;
    }

    private static int copy(double[] out, int offset, double[] input) {
        for (double value : input) {
            out[offset++] = value;
        }
        return offset;
    }

    public static String header() {
        StringBuilder sb = new StringBuilder("target");
        for (int i = 0; i < FEATURE_COUNT; i++) {
            sb.append('\t').append("maf").append(i);
        }
        return sb.toString();
    }
}
