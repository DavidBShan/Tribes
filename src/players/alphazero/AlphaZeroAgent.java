package players.alphazero;

import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.game.GameState;
import players.Agent;
import players.SimpleAgent;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class AlphaZeroAgent extends Agent {

    private final Random rnd;
    private final AZParams params;
    private final SimpleAgent advisor;
    private LinearValueFunction valueFunction;
    private LinearPolicyFunction policyFunction;
    private StateHeuristic heuristic;
    private GameState rootState;
    private Action advisorAction;
    private int fmCalls;
    private int lastTick = -1;
    private int lastActiveTribe = -1;
    private int actionsThisTurn = 0;

    public AlphaZeroAgent(long seed) {
        this(seed, new AZParams());
    }

    public AlphaZeroAgent(long seed, AZParams params) {
        super(seed);
        this.rnd = new Random(seed);
        this.params = params;
        this.advisor = new SimpleAgent(seed + 17);
        this.valueFunction = LinearValueFunction.load(params.modelPath);
        this.policyFunction = LinearPolicyFunction.load(params.policyPath);
    }

    @Override
    public void setPlayerIDs(int playerID, ArrayList<Integer> allIds) {
        super.setPlayerIDs(playerID, allIds);
        advisor.setPlayerIDs(playerID, allIds);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        if (allActions.size() == 1) {
            return allActions.get(0);
        }

        updateTurnCounter(gs);
        this.valueFunction = LinearValueFunction.load(params.modelPath);
        this.policyFunction = LinearPolicyFunction.load(params.policyPath);
        this.heuristic = params.getStateHeuristic(playerID, allPlayerIDs);
        this.rootState = gs;
        this.advisorAction = safeAdvisorAction(gs, ect);

        Action urgent = urgentTacticalAction(gs, allActions);
        if (urgent != null) {
            return urgent;
        }

        Action greedy = greedyPriorityAction(gs, allActions);
        if (greedy != null) {
            return greedy;
        }

        Action forcedEnd = forcedEndTurn(allActions);
        if (forcedEnd != null) {
            return forcedEnd;
        }

        this.fmCalls = 0;

        ArrayList<Action> rootActions = candidateActions(gs, true);
        if (rootActions.isEmpty()) {
            return new EndTurn(gs.getActiveTribeID());
        }

        Node root = new Node(null, gs.copy(), null, 1.0, 0);
        root.expand(rootActions);

        int iterations = 0;
        while (fmCalls < params.num_fmcalls) {
            Node leaf = root;
            while (leaf.expanded && !leaf.state.isGameOver()
                    && leaf.depth < params.ROLLOUT_LENGTH && !leaf.children.isEmpty()) {
                leaf = leaf.selectChild();
            }

            if (!leaf.state.isGameOver() && leaf.depth < params.ROLLOUT_LENGTH && !leaf.expanded) {
                leaf.expand(null);
            }

            double value = evaluateLeaf(leaf.state);
            leaf.backup(value);
            iterations++;

            if (params.stop_type == params.STOP_ITERATIONS && iterations >= params.num_iterations) {
                break;
            }
        }

        Node selected = root.bestChildByVisits();
        Node advised = root.findChild(advisorAction);
        if (advised != null && selected != null && selected != advised) {
            double selectedQ = selected.meanValue();
            double advisedQ = advised.meanValue();
            if (selectedQ < advisedQ + params.advisorOverrideMargin) {
                return advised.actionFromParent;
            }
        }
        if (selected != null && selected.actionFromParent != null) {
            return selected.actionFromParent;
        }

        return bestImmediateAction(gs, rootActions);
    }

    private Action safeAdvisorAction(GameState gs, ElapsedCpuTimer ect) {
        try {
            Action action = advisor.act(gs, ect);
            if (action != null && action.isFeasible(gs)) {
                return action;
            }
        } catch (Throwable ignored) {
            // Advisor is a tactical hint, not a required part of the search.
        }
        return null;
    }

    private void updateTurnCounter(GameState gs) {
        if (lastTick == gs.getTick() && lastActiveTribe == gs.getActiveTribeID()) {
            actionsThisTurn++;
        } else {
            lastTick = gs.getTick();
            lastActiveTribe = gs.getActiveTribeID();
            actionsThisTurn = 0;
        }
    }

    private Action forcedEndTurn(ArrayList<Action> allActions) {
        if (actionsThisTurn < params.forceTurnAfterActions) {
            return null;
        }

        for (Action action : allActions) {
            if (action.getActionType() == Types.ACTION.END_TURN) {
                return action;
            }
        }
        return null;
    }

    private Action urgentTacticalAction(GameState gs, ArrayList<Action> allActions) {
        ArrayList<Action> urgent = new ArrayList<>();
        for (Action action : allActions) {
            Types.ACTION type = action.getActionType();
            if (type == Types.ACTION.CAPTURE || type == Types.ACTION.EXAMINE || type == Types.ACTION.MAKE_VETERAN) {
                urgent.add(action);
            }
        }

        if (urgent.isEmpty()) {
            return null;
        }
        return bestImmediateAction(gs, urgent);
    }

    private Action greedyPriorityAction(GameState gs, ArrayList<Action> allActions) {
        double bestPriority = -Double.MAX_VALUE;
        ArrayList<Action> candidates = new ArrayList<>();

        for (Action action : allActions) {
            Types.ACTION type = action.getActionType();
            if (type == Types.ACTION.END_TURN || type == Types.ACTION.DESTROY || type == Types.ACTION.DISBAND
                    || type == Types.ACTION.SEND_STARS || type == Types.ACTION.DECLARE_WAR) {
                continue;
            }

            double priority = staticActionScore(action);
            if (priority < params.greedyPriorityThreshold) {
                continue;
            }

            if (priority > bestPriority) {
                candidates.clear();
                bestPriority = priority;
            }
            if (priority == bestPriority) {
                candidates.add(action);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }
        return bestImmediateAction(gs, candidates);
    }

    private Action bestImmediateAction(GameState gs, ArrayList<Action> actions) {
        Action best = actions.get(0);
        double bestValue = -Double.MAX_VALUE;

        for (Action action : actions) {
            GameState next = gs.copy();
            next.advance(action, true);
            double value = evaluateLeaf(next);
            value = noise(value);
            if (value > bestValue) {
                bestValue = value;
                best = action;
            }
        }

        return best;
    }

    private ArrayList<Action> candidateActions(GameState state, boolean root) {
        ArrayList<Action> raw;
        if (root && params.PRIORITIZE_ROOT) {
            raw = determineActionGroup(state, rnd);
            if (raw == null) {
                raw = state.getAllAvailableActions();
            }
        } else {
            raw = state.getAllAvailableActions();
        }

        ArrayList<Action> filtered = new ArrayList<>();
        Action endTurn = null;
        for (Action action : raw) {
            Types.ACTION type = action.getActionType();
            if (type == Types.ACTION.END_TURN) {
                endTurn = action;
                filtered.add(action);
            } else if (type != Types.ACTION.DESTROY && type != Types.ACTION.DISBAND) {
                filtered.add(action);
            }
        }

        if (filtered.isEmpty()) {
            filtered.addAll(raw);
        }

        if (filtered.size() > params.prefilterActions) {
            Collections.sort(filtered, new Comparator<Action>() {
                @Override
                public int compare(Action a, Action b) {
                    return Double.compare(staticActionScore(b), staticActionScore(a));
                }
            });

            ArrayList<Action> limited = new ArrayList<>();
            for (int i = 0; i < params.prefilterActions && i < filtered.size(); i++) {
                limited.add(filtered.get(i));
            }
            if (endTurn != null && !limited.contains(endTurn)) {
                limited.set(limited.size() - 1, endTurn);
            }
            if (root && advisorAction != null && !limited.contains(advisorAction) && filtered.contains(advisorAction)) {
                limited.set(0, advisorAction);
            }
            filtered = limited;
        }

        return filtered;
    }

    private double evaluateLeaf(GameState leafState) {
        if (leafState.isGameOver()) {
            return StateFeatures.outcomeLabel(leafState, playerID, allPlayerIDs);
        }

        double heuristicValue = 0.0;
        if (heuristic != null && rootState != null) {
            heuristicValue = Math.tanh(heuristic.evaluateState(rootState, leafState) / params.heuristicScale);
        }
        double positionValue = StateFeatures.positionValue(leafState, playerID, allPlayerIDs);
        heuristicValue = (1.0 - params.positionBlend) * heuristicValue + params.positionBlend * positionValue;

        if (!valueFunction.isTrained()) {
            return heuristicValue;
        }

        double learnedValue = valueFunction.predict(leafState, playerID, allPlayerIDs);
        return StateFeatures.clamp((1.0 - params.heuristicBlend) * learnedValue
                + params.heuristicBlend * heuristicValue);
    }

    private double actionPriorLogit(GameState state, Action action, GameState nextState, int depth) {
        boolean actorIsMe = state.getActiveTribeID() == playerID;
        double orientation = actorIsMe ? 1.0 : -1.0;
        double value = evaluateLeaf(nextState);
        double advisorBoost = depth == 0 && advisorAction != null && advisorAction.equals(action) ? 5.0 : 0.0;
        double policyLogit = 0.0;
        if (policyFunction.isTrained()) {
            int actor = state.getActiveTribeID();
            double p = policyFunction.probability(state, actor, allPlayerIDs, action.getActionType());
            policyLogit = params.policyLogitWeight * Math.log(Math.max(1e-6, p));
        }
        return orientation * value + orientation * (staticActionScore(action) + advisorBoost) * 0.10 + policyLogit;
    }

    private double staticActionScore(Action action) {
        switch (action.getActionType()) {
            case CAPTURE:
                return 5.0;
            case ATTACK:
            case CONVERT:
                return 4.0;
            case LEVEL_UP:
                return 3.5;
            case RESOURCE_GATHERING:
                return 3.0;
            case SPAWN:
            case MAKE_VETERAN:
                return 2.6;
            case RESEARCH_TECH:
            case BUILD:
            case EXAMINE:
                return 2.1;
            case UPGRADE_BOAT:
            case UPGRADE_SHIP:
                return 1.7;
            case MOVE:
            case CLIMB_MOUNTAIN:
                return 1.0;
            case BUILD_ROAD:
                return 0.6;
            case DECLARE_WAR:
                return 4.2;
            case RECOVER:
            case HEAL_OTHERS:
                return 0.1;
            case END_TURN:
                return -0.2;
            case SEND_STARS:
                return -0.7;
            case DESTROY:
            case DISBAND:
                return -4.0;
            default:
                return 0.0;
        }
    }

    private double noise(double input) {
        return (input + params.epsilon) * (1.0 + params.epsilon * (rnd.nextDouble() - 0.5));
    }

    @Override
    public Agent copy() {
        return new AlphaZeroAgent(seed, params);
    }

    private class Node {
        private final Node parent;
        private final GameState state;
        private final Action actionFromParent;
        private final double prior;
        private final int depth;
        private final ArrayList<Node> children;
        private boolean expanded;
        private int visits;
        private double valueSum;

        Node(Node parent, GameState state, Action actionFromParent, double prior, int depth) {
            this.parent = parent;
            this.state = state;
            this.actionFromParent = actionFromParent;
            this.prior = prior;
            this.depth = depth;
            this.children = new ArrayList<>();
            this.expanded = false;
            this.visits = 0;
            this.valueSum = 0.0;
        }

        void expand(ArrayList<Action> providedActions) {
            if (expanded) {
                return;
            }

            ArrayList<Action> actions = providedActions != null ? providedActions : candidateActions(state, false);
            if (actions.isEmpty()) {
                expanded = true;
                return;
            }

            ArrayList<ChildSpec> specs = new ArrayList<>();
            for (Action action : actions) {
                if (fmCalls >= params.num_fmcalls && !specs.isEmpty()) {
                    break;
                }

                GameState next = state.copy();
                next.advance(action, true);
                fmCalls++;
                double logit = actionPriorLogit(state, action, next, depth);
                specs.add(new ChildSpec(action, next, logit));
            }

            Collections.sort(specs, new Comparator<ChildSpec>() {
                @Override
                public int compare(ChildSpec a, ChildSpec b) {
                    return Double.compare(b.logit, a.logit);
                }
            });

            int keep = Math.min(params.maxActionsPerNode, specs.size());
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < keep; i++) {
                max = Math.max(max, specs.get(i).logit);
            }

            double sum = 0.0;
            double[] priors = new double[keep];
            for (int i = 0; i < keep; i++) {
                double scaled = (specs.get(i).logit - max) / Math.max(0.05, params.priorTemperature);
                priors[i] = Math.exp(scaled);
                sum += priors[i];
            }

            for (int i = 0; i < keep; i++) {
                ChildSpec spec = specs.get(i);
                double p = sum > 0.0 ? priors[i] / sum : 1.0 / keep;
                children.add(new Node(this, spec.state, spec.action, p, depth + 1));
            }

            expanded = true;
        }

        Node selectChild() {
            Node selected = null;
            double best = -Double.MAX_VALUE;
            boolean actorIsMe = state.getActiveTribeID() == playerID;
            double sqrtVisits = Math.sqrt(Math.max(1, visits));

            for (Node child : children) {
                double q = child.visits == 0 ? 0.0 : child.valueSum / child.visits;
                double u = params.cpuct * child.prior * sqrtVisits / (1.0 + child.visits);
                double score = (actorIsMe ? q : -q) + u;
                score = noise(score);
                if (score > best) {
                    best = score;
                    selected = child;
                }
            }

            return selected;
        }

        void backup(double value) {
            Node n = this;
            while (n != null) {
                n.visits++;
                n.valueSum += value;
                n = n.parent;
            }
        }

        Node bestChildByVisits() {
            Node selected = null;
            int bestVisits = -1;
            double bestValue = -Double.MAX_VALUE;

            for (Node child : children) {
                double q = child.visits == 0 ? -Double.MAX_VALUE : child.valueSum / child.visits;
                if (child.visits > bestVisits || (child.visits == bestVisits && q > bestValue)) {
                    selected = child;
                    bestVisits = child.visits;
                    bestValue = q;
                }
            }

            return selected;
        }

        Node findChild(Action action) {
            if (action == null) {
                return null;
            }
            for (Node child : children) {
                if (action.equals(child.actionFromParent)) {
                    return child;
                }
            }
            return null;
        }

        double meanValue() {
            if (visits == 0) {
                return -Double.MAX_VALUE;
            }
            return valueSum / visits;
        }
    }

    private static class ChildSpec {
        final Action action;
        final GameState state;
        final double logit;

        ChildSpec(Action action, GameState state, double logit) {
            this.action = action;
            this.state = state;
            this.logit = logit;
        }
    }
}
