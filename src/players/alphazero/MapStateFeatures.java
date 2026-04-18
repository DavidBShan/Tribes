package players.alphazero;

import core.Types;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;

import java.util.ArrayList;

/**
 * Fixed-size board-plane features for the lightweight map-aware neural model.
 *
 * This is a CPU-friendly bridge toward AlphaZero-style board inputs: it exposes
 * tile layout, ownership, units, cities, roads, resources, and visibility while
 * still appending the existing strategic summary features.
 */
public final class MapStateFeatures {

    static final int MAX_BOARD_SIZE = 16;
    static final int MAP_CHANNELS = 10;
    public static final int FEATURE_COUNT = StateFeatures.FEATURE_COUNT
            + MAX_BOARD_SIZE * MAX_BOARD_SIZE * MAP_CHANNELS;

    private static final int CH_VISIBLE = 0;
    private static final int CH_WATER = 1;
    private static final int CH_MOUNTAIN = 2;
    private static final int CH_FOREST = 3;
    private static final int CH_RESOURCE = 4;
    private static final int CH_ROAD = 5;
    private static final int CH_VILLAGE = 6;
    private static final int CH_CITY_OWNER = 7;
    private static final int CH_TERRITORY_OWNER = 8;
    private static final int CH_UNIT_HP_OWNER = 9;

    private MapStateFeatures() {
    }

    public static double[] extract(GameState state, int playerID, ArrayList<Integer> allIds) {
        double[] base = StateFeatures.extract(state, playerID, allIds);
        double[] features = new double[FEATURE_COUNT];
        System.arraycopy(base, 0, features, 0, base.length);

        Board board = state.getBoard();
        if (board == null) {
            return features;
        }

        int size = Math.min(board.getSize(), MAX_BOARD_SIZE);
        boolean[][] visible = state.getTribe(playerID).getObsGrid();
        int offset = StateFeatures.FEATURE_COUNT;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                int tileOffset = offset + ((x * MAX_BOARD_SIZE + y) * MAP_CHANNELS);
                Types.TERRAIN terrain = board.getTerrainAt(x, y);
                features[tileOffset + CH_VISIBLE] = isVisible(visible, x, y) ? 1.0 : 0.0;
                if (terrain != null) {
                    features[tileOffset + CH_WATER] = terrain.isWater() ? 1.0 : 0.0;
                    features[tileOffset + CH_MOUNTAIN] = terrain == Types.TERRAIN.MOUNTAIN ? 1.0 : 0.0;
                    features[tileOffset + CH_FOREST] = terrain == Types.TERRAIN.FOREST ? 1.0 : 0.0;
                    features[tileOffset + CH_VILLAGE] = terrain == Types.TERRAIN.VILLAGE ? 1.0 : 0.0;
                }
                features[tileOffset + CH_RESOURCE] = board.getResourceAt(x, y) == null ? 0.0 : 1.0;
                features[tileOffset + CH_ROAD] = board.isRoad(x, y) ? 1.0 : 0.0;

                City city = board.getCityInBorders(x, y);
                if (city != null) {
                    double owner = ownerValue(city.getTribeId(), playerID);
                    features[tileOffset + CH_TERRITORY_OWNER] = owner;
                    if (city.getPosition().x == x && city.getPosition().y == y) {
                        double capitalScale = city.isCapital() ? 1.0 : 0.65;
                        features[tileOffset + CH_CITY_OWNER] = owner * capitalScale;
                    }
                }

                Unit unit = board.getUnitAt(x, y);
                if (unit != null) {
                    double hp = unit.getMaxHP() <= 0 ? 0.0 : unit.getCurrentHP() / (double) unit.getMaxHP();
                    features[tileOffset + CH_UNIT_HP_OWNER] = ownerValue(unit.getTribeId(), playerID) * hp;
                }
            }
        }

        return features;
    }

    private static boolean isVisible(boolean[][] visible, int x, int y) {
        return visible != null
                && x >= 0 && x < visible.length
                && y >= 0 && y < visible[x].length
                && visible[x][y];
    }

    private static double ownerValue(int owner, int playerID) {
        if (owner < 0) {
            return 0.0;
        }
        return owner == playerID ? 1.0 : -1.0;
    }
}
