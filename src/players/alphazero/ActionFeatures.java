package players.alphazero;

import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.CityAction;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.ResearchTech;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Capture;
import core.actions.unitactions.Convert;
import core.actions.unitactions.Move;
import core.actions.unitactions.UnitAction;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;

public final class ActionFeatures {

    static final int ACTION_COUNT = Types.ACTION.values().length;
    static final int EXTRA_COUNT = 24;
    public static final int FEATURE_COUNT = StateFeatures.FEATURE_COUNT + ACTION_COUNT
            + StateFeatures.FEATURE_COUNT + EXTRA_COUNT;

    private ActionFeatures() {
    }

    public static double[] extract(GameState state, GameState nextState,
                                   int playerID, ArrayList<Integer> allIds, Action action) {
        double[] features = new double[FEATURE_COUNT];
        int offset = 0;
        double[] before = StateFeatures.extract(state, playerID, allIds);
        offset = copy(features, offset, before);

        if (action != null) {
            int ordinal = action.getActionType().ordinal();
            if (ordinal >= 0 && ordinal < ACTION_COUNT) {
                features[offset + ordinal] = 1.0;
            }
        }
        offset += ACTION_COUNT;

        double[] after = nextState == null ? before : StateFeatures.extract(nextState, playerID, allIds);
        for (int i = 0; i < StateFeatures.FEATURE_COUNT; i++) {
            features[offset + i] = after[i] - before[i];
        }
        offset += StateFeatures.FEATURE_COUNT;

        fillExtras(features, offset, state, nextState, playerID, allIds, action);
        return features;
    }

    private static int copy(double[] out, int offset, double[] input) {
        for (double value : input) {
            out[offset++] = value;
        }
        return offset;
    }

    static void fillExtras(double[] out, int offset, GameState state, GameState nextState,
                           int playerID, ArrayList<Integer> allIds, Action action) {
        if (action == null) {
            return;
        }

        Board board = state.getBoard();
        int size = Math.max(1, board.getSize() - 1);
        int i = offset;
        out[i++] = action.getActionType().ordinal() / Math.max(1.0, ACTION_COUNT - 1.0);
        out[i++] = feasible(state, action) ? 1.0 : 0.0;
        out[i++] = state.getActiveTribeID() == playerID ? 1.0 : 0.0;
        out[i++] = nextState == null ? 0.0
                : StateFeatures.positionValue(nextState, playerID, allIds)
                - StateFeatures.positionValue(state, playerID, allIds);

        Unit unit = unitFor(state, action);
        if (unit != null) {
            out[i++] = unit.getType().ordinal() / Math.max(1.0, Types.UNIT.values().length - 1.0);
            out[i++] = unit.getMaxHP() <= 0 ? 0.0 : (double) unit.getCurrentHP() / unit.getMaxHP();
            out[i++] = unit.isVeteran() ? 1.0 : 0.0;
            out[i++] = unit.canMove() ? 1.0 : 0.0;
            out[i++] = unit.canAttack() ? 1.0 : 0.0;
            out[i++] = unit.getKills() / 5.0;
        } else {
            i += 6;
        }

        City city = cityFor(state, action);
        if (city != null) {
            out[i++] = city.getLevel() / 8.0;
            out[i++] = city.canLevelUp() ? 1.0 : 0.0;
            out[i++] = city.getProduction() / 30.0;
        } else {
            i += 3;
        }

        Vector2d target = targetPosition(state, action, unit, city);
        if (target != null) {
            out[i++] = target.x / (double) size;
            out[i++] = target.y / (double) size;
            out[i++] = ordinal(board.getTerrainAt(target.x, target.y), Types.TERRAIN.values());
            Types.RESOURCE resource = board.getResourceAt(target.x, target.y);
            out[i++] = resource == null ? 0.0 : (resource.ordinal() + 1.0) / (Types.RESOURCE.values().length + 1.0);
            out[i++] = unit == null ? 0.0
                    : Vector2d.chebychevDistance(unit.getPosition(), target) / Math.max(1.0, board.getSize());
        } else {
            i += 5;
        }

        Unit targetUnit = targetUnitFor(state, action);
        if (targetUnit != null) {
            out[i++] = targetUnit.getMaxHP() <= 0 ? 0.0 : (double) targetUnit.getCurrentHP() / targetUnit.getMaxHP();
            out[i++] = targetUnit.getType().ordinal() / Math.max(1.0, Types.UNIT.values().length - 1.0);
        } else {
            i += 2;
        }

        City targetCity = targetCityFor(state, action);
        if (targetCity != null) {
            out[i++] = targetCity.isCapital() ? 1.0 : 0.0;
            out[i++] = targetCity.getLevel() / 8.0;
        } else {
            i += 2;
        }

        out[i] = costFeature(state, action);
    }

    private static boolean feasible(GameState state, Action action) {
        try {
            return action.isFeasible(state);
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static Unit unitFor(GameState state, Action action) {
        if (!(action instanceof UnitAction)) {
            return null;
        }
        Object actor = state.getActor(((UnitAction) action).getUnitId());
        return actor instanceof Unit ? (Unit) actor : null;
    }

    private static City cityFor(GameState state, Action action) {
        if (!(action instanceof CityAction)) {
            return null;
        }
        Object actor = state.getActor(((CityAction) action).getCityId());
        return actor instanceof City ? (City) actor : null;
    }

    private static Vector2d targetPosition(GameState state, Action action, Unit unit, City city) {
        if (action instanceof Move) {
            return ((Move) action).getDestination();
        }
        if (action instanceof BuildRoad) {
            return ((BuildRoad) action).getPosition();
        }
        if (action instanceof CityAction) {
            return ((CityAction) action).getTargetPos();
        }
        Unit targetUnit = targetUnitFor(state, action);
        if (targetUnit != null) {
            return targetUnit.getPosition();
        }
        City targetCity = targetCityFor(state, action);
        if (targetCity != null) {
            return targetCity.getPosition();
        }
        return unit != null ? unit.getPosition() : city == null ? null : city.getPosition();
    }

    private static Unit targetUnitFor(GameState state, Action action) {
        int targetId = -1;
        if (action instanceof Attack) {
            targetId = ((Attack) action).getTargetId();
        } else if (action instanceof Convert) {
            targetId = ((Convert) action).getTargetId();
        }
        if (targetId < 0) {
            return null;
        }
        Object actor = state.getActor(targetId);
        return actor instanceof Unit ? (Unit) actor : null;
    }

    private static City targetCityFor(GameState state, Action action) {
        if (!(action instanceof Capture)) {
            return null;
        }
        int targetId = ((Capture) action).getTargetCity();
        Object actor = state.getActor(targetId);
        return actor instanceof City ? (City) actor : null;
    }

    private static double costFeature(GameState state, Action action) {
        try {
            if (action instanceof Spawn) {
                return ((Spawn) action).getUnitType().getCost() / 30.0;
            }
            if (action instanceof Build) {
                return ((Build) action).getBuildingType().getCost() / 30.0;
            }
            if (action instanceof ResearchTech) {
                ResearchTech research = (ResearchTech) action;
                Tribe tribe = state.getTribe(research.getTribeId());
                return research.getTech().getCost(tribe.getNumCities(), tribe.getTechTree()) / 40.0;
            }
        } catch (Throwable ignored) {
            return 0.0;
        }
        return 0.0;
    }

    private static double ordinal(Enum<?> value, Enum<?>[] values) {
        if (value == null || values.length <= 1) {
            return 0.0;
        }
        return value.ordinal() / (double) (values.length - 1);
    }

    public static String header() {
        StringBuilder sb = new StringBuilder("target");
        for (int i = 0; i < FEATURE_COUNT; i++) {
            sb.append('\t').append("af").append(i);
        }
        return sb.toString();
    }
}
