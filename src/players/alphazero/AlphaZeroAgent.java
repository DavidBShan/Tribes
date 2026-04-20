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
    private ValueModel valueFunction;
    private PolicyModel policyFunction;
    private ActionPolicyModel actionPolicyFunction;
    private StateHeuristic heuristic;
    private GameState rootState;
    private Action advisorAction;
    private int fmCalls;
    private int lastTick = -1;
    private int lastActiveTribe = -1;
    private int actionsThisTurn = 0;
    private double[] lastVisitPolicy;
    private double[] lastImprovedPolicy;
    private ArrayList<ActionPolicyTrainingExample> lastActionPolicy;
    private ArrayList<ActionPolicyTrainingExample> lastImprovedActionPolicy;

    public AlphaZeroAgent(long seed) {
        this(seed, new AZParams());
    }

    public AlphaZeroAgent(long seed, AZParams params) {
        super(seed);
        this.rnd = new Random(seed);
        this.params = params;
        this.advisor = new SimpleAgent(seed + 17);
        this.valueFunction = ModelFactory.loadValue(params.networkType, params.modelPath);
        this.policyFunction = ModelFactory.loadPolicy(params.networkType, params.policyPath);
        this.actionPolicyFunction = ModelFactory.loadActionPolicy(params.networkType, params.actionPolicyPath);
    }

    @Override
    public void setPlayerIDs(int playerID, ArrayList<Integer> allIds) {
        super.setPlayerIDs(playerID, allIds);
        advisor.setPlayerIDs(playerID, allIds);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        lastVisitPolicy = null;
        lastImprovedPolicy = null;
        lastActionPolicy = null;
        lastImprovedActionPolicy = null;
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        if (allActions.size() == 1) {
            return allActions.get(0);
        }

        updateTurnCounter(gs);
        this.valueFunction = ModelFactory.loadValue(params.networkType, params.modelPath);
        this.policyFunction = ModelFactory.loadPolicy(params.networkType, params.policyPath);
        this.actionPolicyFunction = ModelFactory.loadActionPolicy(params.networkType, params.actionPolicyPath);
        this.heuristic = params.getStateHeuristic(playerID, allPlayerIDs);
        this.rootState = gs;
        this.advisorAction = (params.advisorOverride || params.staticPriors) ? safeAdvisorAction(gs, ect) : null;

        if (params.tacticalShortcuts) {
            Action urgent = urgentTacticalAction(gs, allActions);
            if (urgent != null) {
                return urgent;
            }

            Action greedy = greedyPriorityAction(gs, allActions);
            if (greedy != null) {
                return greedy;
            }
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

        Node root = new Node(null, gs.copy(), null, 1.0, 0.0, 0.0, 0);
        root.expand(rootActions);

        int iterations = 0;
        if (params.rootSequentialHalving && root.children.size() > 1) {
            iterations = runRootSequentialHalving(root);
        } else {
            while (fmCalls < params.num_fmcalls) {
                runTreeSimulation(root);
                iterations++;

                if (params.stop_type == params.STOP_ITERATIONS && iterations >= params.num_iterations) {
                    break;
                }
            }
        }

        lastVisitPolicy = root.visitPolicyByActionType();
        lastActionPolicy = root.visitPolicyByAction(rootState, playerID, allPlayerIDs);
        lastImprovedPolicy = root.improvedPolicyByActionType();
        lastImprovedActionPolicy = root.improvedPolicyByAction(rootState, playerID, allPlayerIDs);
        Node selected = root.bestChildByVisits();
        Node improvedSelected = root.bestChildByImprovedScore();
        if (improvedSelected != null) {
            selected = improvedSelected;
        }
        boolean sampledSelection = false;
        if (shouldSampleFromVisits(gs)) {
            Node sampled = root.sampleChildByVisits(params.visitSamplingTemperature);
            if (sampled != null) {
                selected = sampled;
                sampledSelection = true;
            }
        }
        Node advised = root.findChild(advisorAction);
        if (params.advisorOverride && !sampledSelection && advised != null && selected != null && selected != advised) {
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

    private int runTreeSimulation(Node root) {
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
        return 1;
    }

    private int runTreeSimulationFrom(Node start) {
        Node leaf = start;
        while (leaf.expanded && !leaf.state.isGameOver()
                && leaf.depth < params.ROLLOUT_LENGTH && !leaf.children.isEmpty()) {
            leaf = leaf.selectChild();
        }

        if (!leaf.state.isGameOver() && leaf.depth < params.ROLLOUT_LENGTH && !leaf.expanded) {
            leaf.expand(null);
        }

        double value = evaluateLeaf(leaf.state);
        leaf.backup(value);
        return 1;
    }

    private int runRootSequentialHalving(Node root) {
        ArrayList<Node> active = new ArrayList<>(root.children);
        int iterations = 0;
        int rounds = Math.max(1, (int) Math.ceil(Math.log(Math.max(2, active.size())) / Math.log(2.0)));

        for (int round = 0; round < rounds && active.size() > 1
                && fmCalls < params.num_fmcalls; round++) {
            int remainingRounds = Math.max(1, rounds - round);
            int remainingBudget = Math.max(1, params.num_fmcalls - fmCalls);
            int simsPerChild = Math.max(1, remainingBudget / Math.max(1, active.size() * remainingRounds));

            for (Node child : active) {
                for (int i = 0; i < simsPerChild && fmCalls < params.num_fmcalls; i++) {
                    iterations += runTreeSimulationFrom(child);
                    if (params.stop_type == params.STOP_ITERATIONS && iterations >= params.num_iterations) {
                        return iterations;
                    }
                }
            }

            Collections.sort(active, new Comparator<Node>() {
                @Override
                public int compare(Node a, Node b) {
                    return Double.compare(rootHalvingScore(b), rootHalvingScore(a));
                }
            });
            int keep = Math.max(1, (active.size() + 1) / 2);
            while (active.size() > keep) {
                active.remove(active.size() - 1);
            }
        }

        int cursor = 0;
        while (fmCalls < params.num_fmcalls && !active.isEmpty()) {
            Node child = active.get(cursor % active.size());
            iterations += runTreeSimulationFrom(child);
            cursor++;
            if (params.stop_type == params.STOP_ITERATIONS && iterations >= params.num_iterations) {
                break;
            }
        }
        return iterations;
    }

    private double rootHalvingScore(Node child) {
        double q = child.visits == 0 ? -1.0 : child.valueSum / child.visits;
        return child.selectionLogit + params.improvedPolicyValueWeight * q;
    }

    private boolean shouldSampleFromVisits(GameState gs) {
        if (params.visitSamplingTemperature <= 0.0) {
            return false;
        }
        return params.visitSamplingUntilTick <= 0 || gs.getTick() <= params.visitSamplingUntilTick;
    }

    public double[] lastVisitPolicyTargets() {
        if (lastVisitPolicy == null) {
            return null;
        }
        return lastVisitPolicy.clone();
    }

    public ArrayList<ActionPolicyTrainingExample> lastActionPolicyExamples() {
        if (lastActionPolicy == null) {
            return null;
        }
        return new ArrayList<>(lastActionPolicy);
    }

    public double[] lastImprovedPolicyTargets() {
        if (lastImprovedPolicy == null) {
            return null;
        }
        return lastImprovedPolicy.clone();
    }

    public ArrayList<ActionPolicyTrainingExample> lastImprovedActionPolicyExamples() {
        if (lastImprovedActionPolicy == null) {
            return null;
        }
        return new ArrayList<>(lastImprovedActionPolicy);
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
            if (!next.advance(action, true, false)) {
                continue;
            }
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
            if (params.prefilterByStaticScore) {
                Collections.sort(filtered, new Comparator<Action>() {
                    @Override
                    public int compare(Action a, Action b) {
                        return Double.compare(staticActionScore(b), staticActionScore(a));
                    }
                });
            }

            ArrayList<Action> limited = new ArrayList<>();
            for (int i = 0; i < params.prefilterActions && i < filtered.size(); i++) {
                limited.add(filtered.get(i));
            }
            if (endTurn != null && !limited.contains(endTurn)) {
                limited.set(limited.size() - 1, endTurn);
            }
            if (params.staticPriors && root && advisorAction != null && !limited.contains(advisorAction) && filtered.contains(advisorAction)) {
                limited.set(0, advisorAction);
            }
            filtered = limited;
        }

        return filtered;
    }

    private double evaluateLeaf(GameState leafState) {
        return evaluateLeaf(leafState, playerID, heuristic);
    }

    private double evaluateLeafForPlayer(GameState leafState, int perspectivePlayer) {
        StateHeuristic perspectiveHeuristic = perspectivePlayer == playerID
                ? heuristic : params.getStateHeuristic(perspectivePlayer, allPlayerIDs);
        return evaluateLeaf(leafState, perspectivePlayer, perspectiveHeuristic);
    }

    private double evaluateLeaf(GameState leafState, int perspectivePlayer, StateHeuristic perspectiveHeuristic) {
        if (leafState.isGameOver()) {
            return StateFeatures.outcomeLabel(leafState, perspectivePlayer, allPlayerIDs);
        }

        double heuristicValue = 0.0;
        if (perspectiveHeuristic != null && rootState != null) {
            heuristicValue = Math.tanh(perspectiveHeuristic.evaluateState(rootState, leafState)
                    / params.heuristicScale);
        }
        double positionValue = StateFeatures.positionValue(leafState, perspectivePlayer, allPlayerIDs);
        heuristicValue = (1.0 - params.positionBlend) * heuristicValue + params.positionBlend * positionValue;

        if (!valueFunction.isTrained()) {
            return params.learnedValueOnly ? 0.0 : heuristicValue;
        }

        double learnedValue = valueFunction.predict(leafState, perspectivePlayer, allPlayerIDs);
        if (params.learnedValueOnly) {
            return learnedValue;
        }
        double blend = params.heuristicBlend;
        if (params.disagreementHeuristicBlend > blend
                && learnedValue * heuristicValue < 0.0
                && Math.abs(heuristicValue) >= params.disagreementHeuristicThreshold) {
            blend = params.disagreementHeuristicBlend;
        }
        return StateFeatures.clamp((1.0 - blend) * learnedValue + blend * heuristicValue);
    }

    private double actionPriorLogit(GameState state, Action action, GameState nextState, int depth) {
        boolean actorIsMe = state.getActiveTribeID() == playerID;
        int actor = state.getActiveTribeID();
        double orientation = actorIsMe ? 1.0 : -1.0;
        double value = params.nextStateValuePrior ? evaluateLeaf(nextState) : 0.0;
        double advisorBoost = params.staticPriors && depth == 0 && advisorAction != null && advisorAction.equals(action) ? 5.0 : 0.0;
        double staticPrior = params.staticPriors ? staticActionScore(action) + advisorBoost : 0.0;
        double targetOrientedLogit = orientation * value + orientation * staticPrior * 0.10;
        if (!actorIsMe && params.opponentSelfishWeight > 0.0) {
            double selfishWeight = Math.max(0.0, Math.min(1.0, params.opponentSelfishWeight));
            double actorValue = params.nextStateValuePrior ? evaluateLeafForPlayer(nextState, actor) : 0.0;
            double actorOrientedLogit = actorValue + staticPrior * 0.10;
            targetOrientedLogit = (1.0 - selfishWeight) * targetOrientedLogit
                    + selfishWeight * actorOrientedLogit;
        }
        double policyLogit = 0.0;
        if (policyFunction.isTrained()) {
            double p = policyFunction.probability(state, actor, allPlayerIDs, action.getActionType());
            policyLogit = params.policyLogitWeight * Math.log(Math.max(1e-6, p));
        }
        double actionPolicyLogit = 0.0;
        if (actionPolicyFunction.isTrained() && params.actionPolicyLogitWeight != 0.0) {
            actionPolicyLogit = params.actionPolicyLogitWeight
                    * actionPolicyFunction.logit(state, nextState, actor, allPlayerIDs, action);
        }
        return targetOrientedLogit + policyLogit + actionPolicyLogit;
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

    private double[] dirichletNoise(int n, double alpha) {
        double[] out = new double[n];
        double sum = 0.0;
        double shape = Math.max(1e-3, alpha);
        for (int i = 0; i < n; i++) {
            out[i] = sampleGamma(shape);
            sum += out[i];
        }

        if (sum <= 0.0) {
            double uniform = 1.0 / n;
            for (int i = 0; i < n; i++) {
                out[i] = uniform;
            }
            return out;
        }

        for (int i = 0; i < n; i++) {
            out[i] /= sum;
        }
        return out;
    }

    private double sampleGamma(double shape) {
        if (shape < 1.0) {
            double boosted = sampleGamma(shape + 1.0);
            return boosted * Math.pow(Math.max(1e-12, rnd.nextDouble()), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x = rnd.nextGaussian();
            double v = 1.0 + c * x;
            if (v <= 0.0) {
                continue;
            }
            v = v * v * v;
            double u = rnd.nextDouble();
            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v;
            }
            if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
                return d * v;
            }
        }
    }

    private double sampleGumbel() {
        double u = Math.max(1e-12, Math.min(1.0 - 1e-12, rnd.nextDouble()));
        return -Math.log(-Math.log(u));
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
        private final double priorLogit;
        private final double selectionLogit;
        private final int depth;
        private final ArrayList<Node> children;
        private boolean expanded;
        private int visits;
        private double valueSum;
        private int cachedPerspectivePlayer;
        private double cachedPerspectiveValue;

        Node(Node parent, GameState state, Action actionFromParent, double prior,
             double priorLogit, double selectionLogit, int depth) {
            this.parent = parent;
            this.state = state;
            this.actionFromParent = actionFromParent;
            this.prior = prior;
            this.priorLogit = priorLogit;
            this.selectionLogit = selectionLogit;
            this.depth = depth;
            this.children = new ArrayList<>();
            this.expanded = false;
            this.visits = 0;
            this.valueSum = 0.0;
            this.cachedPerspectivePlayer = Integer.MIN_VALUE;
            this.cachedPerspectiveValue = 0.0;
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
                if (!action.isFeasible(state)) {
                    continue;
                }
                if (fmCalls >= params.num_fmcalls && !specs.isEmpty()) {
                    break;
                }

                GameState next = state.copy();
                if (!next.advance(action, true, false)) {
                    continue;
                }
                fmCalls++;
                double priorLogit = actionPriorLogit(state, action, next, depth);
                double selectionLogit = priorLogit;
                if (depth == 0 && params.rootGumbelScale > 0.0) {
                    selectionLogit += params.rootGumbelScale * sampleGumbel();
                }
                specs.add(new ChildSpec(action, next, priorLogit, selectionLogit));
            }

            Collections.sort(specs, new Comparator<ChildSpec>() {
                @Override
                public int compare(ChildSpec a, ChildSpec b) {
                    return Double.compare(b.selectionLogit, a.selectionLogit);
                }
            });

            int keep = Math.min(params.maxActionsPerNode, specs.size());
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < keep; i++) {
                max = Math.max(max, specs.get(i).selectionLogit);
            }

            double sum = 0.0;
            double[] priors = new double[keep];
            for (int i = 0; i < keep; i++) {
                double scaled = (specs.get(i).selectionLogit - max) / Math.max(0.05, params.priorTemperature);
                priors[i] = Math.exp(scaled);
                sum += priors[i];
            }

            for (int i = 0; i < keep; i++) {
                priors[i] = sum > 0.0 ? priors[i] / sum : 1.0 / keep;
            }

            if (depth == 0 && params.rootNoiseFraction > 0.0 && keep > 1) {
                double[] noise = dirichletNoise(keep, params.rootDirichletAlpha);
                double noiseFraction = Math.max(0.0, Math.min(1.0, params.rootNoiseFraction));
                for (int i = 0; i < keep; i++) {
                    priors[i] = (1.0 - noiseFraction) * priors[i] + noiseFraction * noise[i];
                }
            }

            for (int i = 0; i < keep; i++) {
                ChildSpec spec = specs.get(i);
                children.add(new Node(this, spec.state, spec.action, priors[i],
                        spec.priorLogit, spec.selectionLogit, depth + 1));
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
                double adversaryWeight = Math.max(0.0, params.opponentAdversaryWeight);
                double valueTerm = actorIsMe ? q : -adversaryWeight * q;
                if (!actorIsMe && params.opponentSelfishWeight > 0.0) {
                    double selfishWeight = Math.max(0.0, Math.min(1.0, params.opponentSelfishWeight));
                    valueTerm += selfishWeight * child.perspectiveValue(state.getActiveTribeID());
                }
                double score = valueTerm + u;
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

        Node bestChildByImprovedScore() {
            if (params.rootValueSelectionWeight <= 0.0 || depth != 0) {
                return null;
            }

            Node selected = null;
            double bestScore = -Double.MAX_VALUE;
            for (Node child : children) {
                if (child.visits <= 0) {
                    continue;
                }
                double q = child.valueSum / child.visits;
                double score = child.selectionLogit + params.rootValueSelectionWeight * q;
                score = noise(score);
                if (score > bestScore) {
                    selected = child;
                    bestScore = score;
                }
            }
            return selected;
        }

        Node sampleChildByVisits(double temperature) {
            double temp = Math.max(1e-6, temperature);
            double total = 0.0;
            double[] weights = new double[children.size()];
            for (int i = 0; i < children.size(); i++) {
                Node child = children.get(i);
                if (child.visits <= 0) {
                    continue;
                }
                weights[i] = Math.pow(child.visits, 1.0 / temp);
                total += weights[i];
            }
            if (total <= 0.0) {
                return null;
            }

            double pick = rnd.nextDouble() * total;
            for (int i = 0; i < children.size(); i++) {
                pick -= weights[i];
                if (pick <= 0.0) {
                    return children.get(i);
                }
            }
            return children.isEmpty() ? null : children.get(children.size() - 1);
        }

        double[] visitPolicyByActionType() {
            double[] policy = new double[Types.ACTION.values().length];
            double total = 0.0;
            for (Node child : children) {
                if (child.actionFromParent == null || child.visits <= 0) {
                    continue;
                }
                int actionType = child.actionFromParent.getActionType().ordinal();
                if (actionType < 0 || actionType >= policy.length) {
                    continue;
                }
                policy[actionType] += child.visits;
                total += child.visits;
            }

            if (total <= 0.0) {
                return null;
            }
            for (int i = 0; i < policy.length; i++) {
                policy[i] /= total;
            }
            return policy;
        }

        double[] improvedPolicyByActionType() {
            double[] childWeights = improvedChildWeights();
            if (childWeights == null) {
                return null;
            }

            double[] policy = new double[Types.ACTION.values().length];
            for (int i = 0; i < children.size(); i++) {
                Node child = children.get(i);
                if (child.actionFromParent == null || child.visits <= 0) {
                    continue;
                }
                int actionType = child.actionFromParent.getActionType().ordinal();
                if (actionType >= 0 && actionType < policy.length) {
                    policy[actionType] += childWeights[i];
                }
            }
            return policy;
        }

        ArrayList<ActionPolicyTrainingExample> visitPolicyByAction(GameState parentState,
                                                                    int actor,
                                                                    ArrayList<Integer> ids) {
            ArrayList<ActionPolicyTrainingExample> examples = new ArrayList<>();
            double total = 0.0;
            for (Node child : children) {
                if (child.actionFromParent != null && child.visits > 0) {
                    total += child.visits;
                }
            }
            if (total <= 0.0) {
                return examples;
            }
            for (Node child : children) {
                if (child.actionFromParent == null) {
                    continue;
                }
                double target = child.visits <= 0 ? 0.0 : child.visits / total;
                examples.add(new ActionPolicyTrainingExample(target,
                        ActionFeatureInputs.extract(params.networkType, parentState, child.state,
                                actor, ids, child.actionFromParent)));
            }
            return examples;
        }

        ArrayList<ActionPolicyTrainingExample> improvedPolicyByAction(GameState parentState,
                                                                       int actor,
                                                                       ArrayList<Integer> ids) {
            ArrayList<ActionPolicyTrainingExample> examples = new ArrayList<>();
            double[] childWeights = improvedChildWeights();
            if (childWeights == null) {
                return examples;
            }
            for (int i = 0; i < children.size(); i++) {
                Node child = children.get(i);
                if (child.actionFromParent == null) {
                    continue;
                }
                examples.add(new ActionPolicyTrainingExample(childWeights[i],
                        ActionFeatureInputs.extract(params.networkType, parentState, child.state,
                                actor, ids, child.actionFromParent)));
            }
            return examples;
        }

        private double[] improvedChildWeights() {
            if (children.isEmpty()) {
                return null;
            }

            double max = -Double.MAX_VALUE;
            int usable = 0;
            double[] scores = new double[children.size()];
            for (int i = 0; i < children.size(); i++) {
                Node child = children.get(i);
                if (child.visits <= 0) {
                    scores[i] = -Double.MAX_VALUE;
                    continue;
                }
                double q = child.valueSum / child.visits;
                scores[i] = child.priorLogit + params.improvedPolicyValueWeight * q;
                max = Math.max(max, scores[i]);
                usable++;
            }
            if (usable <= 0) {
                return null;
            }

            double temp = Math.max(0.05, params.improvedPolicyTemperature);
            double total = 0.0;
            double[] weights = new double[children.size()];
            for (int i = 0; i < children.size(); i++) {
                if (scores[i] == -Double.MAX_VALUE) {
                    continue;
                }
                weights[i] = Math.exp((scores[i] - max) / temp);
                total += weights[i];
            }
            if (total <= 0.0) {
                return null;
            }
            for (int i = 0; i < weights.length; i++) {
                weights[i] /= total;
            }
            return weights;
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

        double perspectiveValue(int perspectivePlayer) {
            if (cachedPerspectivePlayer != perspectivePlayer) {
                cachedPerspectivePlayer = perspectivePlayer;
                cachedPerspectiveValue = evaluateLeafForPlayer(state, perspectivePlayer);
            }
            return cachedPerspectiveValue;
        }
    }

    private static class ChildSpec {
        final Action action;
        final GameState state;
        final double priorLogit;
        final double selectionLogit;

        ChildSpec(Action action, GameState state, double priorLogit, double selectionLogit) {
            this.action = action;
            this.state = state;
            this.priorLogit = priorLogit;
            this.selectionLogit = selectionLogit;
        }
    }
}
