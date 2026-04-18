package players.alphazero;

import core.Types;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.GameState;

import java.util.ArrayList;

/**
 * Compact, scaled feature vector used by the dependency-free value function.
 *
 * The features are deliberately high-level game signals, not board pixels. This keeps training fast enough
 * to run from the Java project without adding a neural-network dependency.
 */
public final class StateFeatures {

    private static final int METRIC_COUNT = 13;
    public static final int FEATURE_COUNT = 3 + METRIC_COUNT * 3;

    private StateFeatures() {
    }

    public static double[] extract(GameState state, int playerID, ArrayList<Integer> allIds) {
        double[] features = new double[FEATURE_COUNT];
        features[0] = 1.0;
        features[1] = state.getTick() / 50.0;
        features[2] = state.getActiveTribeID() == playerID ? 1.0 : -1.0;

        double[] mine = metrics(state, playerID);
        double[] oppMean = new double[METRIC_COUNT];
        double[] oppMax = new double[METRIC_COUNT];
        int oppCount = 0;

        for (int i = 0; i < oppMax.length; i++) {
            oppMax[i] = -Double.MAX_VALUE;
        }

        for (Integer id : allIds) {
            if (id == playerID) {
                continue;
            }

            double[] opp = metrics(state, id);
            oppCount++;
            for (int i = 0; i < METRIC_COUNT; i++) {
                oppMean[i] += opp[i];
                if (opp[i] > oppMax[i]) {
                    oppMax[i] = opp[i];
                }
            }
        }

        if (oppCount > 0) {
            for (int i = 0; i < METRIC_COUNT; i++) {
                oppMean[i] /= oppCount;
            }
        } else {
            for (int i = 0; i < METRIC_COUNT; i++) {
                oppMax[i] = 0.0;
            }
        }

        int offset = 3;
        for (int i = 0; i < METRIC_COUNT; i++) {
            features[offset++] = mine[i];
        }
        for (int i = 0; i < METRIC_COUNT; i++) {
            features[offset++] = mine[i] - oppMean[i];
        }
        for (int i = 0; i < METRIC_COUNT; i++) {
            features[offset++] = mine[i] - oppMax[i];
        }

        return features;
    }

    public static double outcomeLabel(GameState finalState, int playerID, ArrayList<Integer> allIds) {
        return outcomeLabel(finalState, playerID, allIds, 0.0);
    }

    public static double outcomeLabel(GameState finalState, int playerID, ArrayList<Integer> allIds,
                                      double rankBlend) {
        Types.RESULT result = finalState.getTribeWinStatus(playerID);
        double margin = scoreMargin(finalState, playerID, allIds);
        double marginValue = Math.tanh(margin / 12000.0);

        double base;
        if (result == Types.RESULT.WIN) {
            base = clamp(0.80 + 0.20 * marginValue);
        } else if (result == Types.RESULT.LOSS) {
            base = clamp(-0.80 + 0.20 * marginValue);
        } else {
            base = clamp(marginValue);
        }

        double blend = Math.max(0.0, Math.min(1.0, rankBlend));
        if (blend <= 0.0) {
            return base;
        }
        return clamp((1.0 - blend) * base + blend * rankValue(finalState, playerID, allIds));
    }

    public static double scoreMargin(GameState state, int playerID, ArrayList<Integer> allIds) {
        double myScore = state.getScore(playerID);
        double bestOther = -Double.MAX_VALUE;
        for (Integer id : allIds) {
            if (id != playerID) {
                bestOther = Math.max(bestOther, state.getScore(id));
            }
        }
        if (bestOther == -Double.MAX_VALUE) {
            bestOther = 0.0;
        }
        return myScore - bestOther;
    }

    private static double rankValue(GameState state, int playerID, ArrayList<Integer> allIds) {
        if (allIds == null || allIds.size() <= 1) {
            return Math.tanh(state.getScore(playerID) / 12000.0);
        }

        double myScore = state.getScore(playerID);
        int better = 0;
        int opponents = 0;
        double opponentTotal = 0.0;
        for (Integer id : allIds) {
            if (id == playerID) {
                continue;
            }
            opponents++;
            double score = state.getScore(id);
            opponentTotal += score;
            if (score > myScore) {
                better++;
            }
        }
        if (opponents <= 0) {
            return Math.tanh(myScore / 12000.0);
        }

        double rank = 1.0 - 2.0 * better / opponents;
        double averageOpponent = opponentTotal / opponents;
        double averageMargin = Math.tanh((myScore - averageOpponent) / 12000.0);
        return clamp(0.70 * rank + 0.30 * averageMargin);
    }

    public static double positionValue(GameState state, int playerID, ArrayList<Integer> allIds) {
        double myCities = state.getCities(playerID).size();
        double myProduction = state.getTribeProduction(playerID);
        double myUnits = state.getUnits(playerID).size();
        double myScore = state.getScore(playerID);
        boolean controlsCapital = state.getTribe(playerID).controlsCapital();

        if (myCities <= 0.0) {
            return -1.0;
        }

        double bestOtherCities = 0.0;
        double bestOtherProduction = 0.0;
        double bestOtherUnits = 0.0;
        double bestOtherScore = 0.0;
        for (Integer id : allIds) {
            if (id == playerID) {
                continue;
            }
            bestOtherCities = Math.max(bestOtherCities, state.getCities(id).size());
            bestOtherProduction = Math.max(bestOtherProduction, state.getTribeProduction(id));
            bestOtherUnits = Math.max(bestOtherUnits, state.getUnits(id).size());
            bestOtherScore = Math.max(bestOtherScore, state.getScore(id));
        }

        double cityDeficit = Math.max(0.0, bestOtherCities - myCities);
        double capitalSafety = controlsCapital ? 1.8 : -2.6;
        double raw = 1.75 * (myCities - bestOtherCities)
                - 0.75 * cityDeficit
                + 0.08 * (myProduction - bestOtherProduction)
                + 0.04 * (myUnits - bestOtherUnits)
                + 0.00025 * (myScore - bestOtherScore)
                + capitalSafety;
        return Math.tanh(raw / 3.0);
    }

    public static double survivalValue(GameState state, int playerID, ArrayList<Integer> allIds) {
        double myCities = state.getCities(playerID).size();
        if (myCities <= 0.0) {
            return -1.0;
        }

        double bestOtherCities = 0.0;
        double bestOtherProduction = 0.0;
        for (Integer id : allIds) {
            if (id == playerID) {
                continue;
            }
            bestOtherCities = Math.max(bestOtherCities, state.getCities(id).size());
            bestOtherProduction = Math.max(bestOtherProduction, state.getTribeProduction(id));
        }

        double cityDelta = myCities - bestOtherCities;
        double productionDelta = state.getTribeProduction(playerID) - bestOtherProduction;
        double capitalSafety = state.getTribe(playerID).controlsCapital() ? 1.0 : -1.8;
        double raw = 1.35 * cityDelta + 0.05 * productionDelta + capitalSafety;
        return Math.tanh(raw / 3.0);
    }

    private static double[] metrics(GameState state, int playerID) {
        double[] m = new double[METRIC_COUNT];
        Tribe tribe = state.getTribe(playerID);
        ArrayList<City> cities = state.getCities(playerID);
        ArrayList<Unit> units = state.getUnits(playerID);

        int cityLevelSum = 0;
        for (City city : cities) {
            cityLevelSum += city.getLevel();
        }

        int visible = 0;
        boolean[][] obs = tribe.getObsGrid();
        if (obs != null) {
            for (boolean[] row : obs) {
                for (boolean cell : row) {
                    if (cell) {
                        visible++;
                    }
                }
            }
        }

        int actionCount = 0;
        if (state.getActiveTribeID() == playerID) {
            actionCount = state.getAllAvailableActions().size();
        }

        m[0] = state.getScore(playerID) / 25000.0;
        m[1] = state.getTribeProduction(playerID) / 100.0;
        m[2] = tribe.getStars() / 100.0;
        m[3] = state.getTribeTechTree(playerID).getNumResearched() / 30.0;
        m[4] = cities.size() / 10.0;
        m[5] = units.size() / 60.0;
        m[6] = state.getNKills(playerID) / 30.0;
        m[7] = cityLevelSum / 80.0;
        m[8] = tribe.getConnectedCities().size() / 10.0;
        m[9] = visible / 400.0;
        m[10] = tribe.controlsCapital() ? 1.0 : 0.0;
        m[11] = resultValue(tribe.getWinner());
        m[12] = actionCount / 300.0;

        return m;
    }

    private static double resultValue(Types.RESULT result) {
        if (result == Types.RESULT.WIN) {
            return 1.0;
        }
        if (result == Types.RESULT.LOSS) {
            return -1.0;
        }
        return 0.0;
    }

    static double clamp(double value) {
        if (value > 1.0) {
            return 1.0;
        }
        if (value < -1.0) {
            return -1.0;
        }
        return value;
    }

    public static String header() {
        StringBuilder sb = new StringBuilder("label");
        for (int i = 0; i < FEATURE_COUNT; i++) {
            sb.append('\t').append("f").append(i);
        }
        return sb.toString();
    }
}
