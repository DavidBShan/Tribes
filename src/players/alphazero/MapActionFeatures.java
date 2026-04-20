package players.alphazero;

import core.actions.Action;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;

public final class MapActionFeatures {

    static final int ACTION_MAP_CHANNELS = 4;
    static final int ACTION_MAP_FEATURE_COUNT = MapStateFeatures.MAX_BOARD_SIZE
            * MapStateFeatures.MAX_BOARD_SIZE * ACTION_MAP_CHANNELS;
    private static final int CH_SOURCE = 0;
    private static final int CH_TARGET = 1;
    private static final int CH_SOURCE_RING = 2;
    private static final int CH_TARGET_RING = 3;

    public static final int FEATURE_COUNT = MapStateFeatures.FEATURE_COUNT
            + ACTION_MAP_FEATURE_COUNT
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

        fillActionMap(features, offset, state, action);
        offset += ACTION_MAP_FEATURE_COUNT;

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

    private static void fillActionMap(double[] features, int offset, GameState state, Action action) {
        if (state == null || action == null || state.getBoard() == null) {
            return;
        }

        Unit unit = ActionFeatures.unitFor(state, action);
        City city = ActionFeatures.cityFor(state, action);
        Vector2d source = sourcePosition(unit, city);
        Vector2d target = ActionFeatures.targetPosition(state, action, unit, city);
        Board board = state.getBoard();
        int size = Math.min(board.getSize(), MapStateFeatures.MAX_BOARD_SIZE);

        mark(features, offset, source, size, CH_SOURCE, 1.0);
        mark(features, offset, target, size, CH_TARGET, 1.0);
        markNeighborhood(features, offset, source, size, CH_SOURCE_RING, 0.5);
        markNeighborhood(features, offset, target, size, CH_TARGET_RING, 0.5);
    }

    private static Vector2d sourcePosition(Unit unit, City city) {
        if (unit != null) {
            return unit.getPosition();
        }
        return city == null ? null : city.getPosition();
    }

    private static void markNeighborhood(double[] features, int offset, Vector2d center,
                                         int size, int channel, double value) {
        if (center == null) {
            return;
        }
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                mark(features, offset, new Vector2d(center.x + dx, center.y + dy), size, channel, value);
            }
        }
    }

    private static void mark(double[] features, int offset, Vector2d pos,
                             int size, int channel, double value) {
        if (pos == null || channel < 0 || channel >= ACTION_MAP_CHANNELS) {
            return;
        }
        if (pos.x < 0 || pos.y < 0 || pos.x >= size || pos.y >= size) {
            return;
        }
        int index = offset + ((pos.x * MapStateFeatures.MAX_BOARD_SIZE + pos.y) * ACTION_MAP_CHANNELS) + channel;
        features[index] = value;
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
