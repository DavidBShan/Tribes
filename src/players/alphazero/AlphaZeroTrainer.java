package players.alphazero;

import core.Constants;
import core.Types;
import core.actors.Tribe;
import core.game.Game;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.RandomAgent;
import players.SimpleAgent;
import players.osla.OSLAParams;
import players.osla.OneStepLookAheadAgent;

import java.io.IOException;
import java.util.ArrayList;

import static core.Types.GAME_MODE.CAPITALS;
import static core.Types.TRIBE.IMPERIUS;
import static core.Types.TRIBE.XIN_XI;

public class AlphaZeroTrainer {

    public static void main(String[] args) {
        Options opts = Options.parse(args);
        configureHeadless(opts);

        System.out.println("AlphaZero-style value training");
        System.out.println("valueModel=" + opts.modelPath + " valueData=" + opts.dataPath);
        System.out.println("policyModel=" + opts.policyPath + " policyData=" + opts.policyDataPath);
        if (opts.recordTrajectories) {
            System.out.println("sftTrajectories=" + opts.trajectoryPath);
        }

        boolean targetReached = false;
        long startedAt = System.nanoTime();
        try (SftTrajectoryWriter trajectoryWriter = newTrajectoryWriter(opts)) {
            for (int iteration = 1; iteration <= opts.iterations; iteration++) {
                System.out.println("=== iteration " + iteration + " ===");
                runTrainingGames(opts, iteration,
                        opts.selfPlayOnly || (targetReached && opts.selfPlayAfterTarget), trajectoryWriter);

                ArrayList<ValueTrainingExample> examples = ValueDataset.load(opts.dataPath, opts.maxTrainingExamples);
                LinearValueFunction vf = LinearValueFunction.load(opts.modelPath);
                if (!examples.isEmpty()) {
                    double loss = vf.train(examples, opts.epochs, opts.learningRate, opts.l2, opts.seed + iteration);
                    vf.save(opts.modelPath);
                    System.out.printf("trained value on %d examples; mse=%.5f%n", examples.size(), loss);
                } else {
                    System.out.println("trained value on 0 examples; skipped");
                }

                ArrayList<PolicyTrainingExample> policyExamples = PolicyDataset.load(opts.policyDataPath, opts.maxTrainingExamples);
                LinearPolicyFunction policy = LinearPolicyFunction.load(opts.policyPath);
                if (!policyExamples.isEmpty()) {
                    double policyLoss = policy.train(policyExamples, opts.policyEpochs, opts.policyLearningRate, opts.l2, opts.seed + 991 * iteration);
                    policy.save(opts.policyPath);
                    System.out.printf("trained policy on %d examples; xent=%.5f%n", policyExamples.size(), policyLoss);
                } else {
                    System.out.println("trained policy on 0 examples; skipped");
                }

                MatchResult simple = evaluate(opts, "SIMPLE", opts.evalGames, opts.seed + 100000L * iteration);
                MatchResult osla = evaluate(opts, "OSLA", opts.evalGames, opts.seed + 200000L * iteration);
                System.out.println(simple);
                System.out.println(osla);
                MatchResult reference = null;
                if (opts.evalReference) {
                    reference = evaluate(opts, "REFERENCE_AZ", opts.evalGames, opts.seed + 300000L * iteration);
                    System.out.println(reference);
                }

                boolean referencePassed = reference == null || reference.winRate() >= opts.targetWinRate;
                if (simple.winRate() >= opts.targetWinRate && osla.winRate() >= opts.targetWinRate
                        && referencePassed) {
                    targetReached = true;
                    System.out.printf("target reached: SIMPLE %.2f, OSLA %.2f%s%n",
                            simple.winRate(), osla.winRate(),
                            reference == null ? "" : String.format(", REFERENCE_AZ %.2f", reference.winRate()));
                    if (!opts.continueAfterTarget) {
                        break;
                    }
                    System.out.println("continuing with self-play while keeping SIMPLE/OSLA as regression baselines");
                }
            }

            if (trajectoryWriter != null) {
                trajectoryWriter.writeManifest(opts.iterations, opts.gamesPerIteration, elapsedSeconds(startedAt));
                System.out.printf("wrote %d SFT trajectory examples to %s (%d skipped)%n",
                        trajectoryWriter.count(), opts.trajectoryPath, trajectoryWriter.skipped());
            }

            if (!targetReached) {
                System.out.printf("target not reached within %d iterations; continue with more --iterations/--games%n",
                        opts.iterations);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write SFT trajectory data", e);
        }
    }

    private static SftTrajectoryWriter newTrajectoryWriter(Options opts) throws IOException {
        if (!opts.recordTrajectories) {
            return null;
        }
        return new SftTrajectoryWriter(opts.trajectoryPath, opts.promptId,
                SftTrajectoryWriter.DEFAULT_SYSTEM_PROMPT, opts.maxPromptActions,
                opts.actionFormat, opts.trajectoryFlushEvery);
    }

    private static void configureHeadless(Options opts) {
        Constants.VISUALS = false;
        Constants.LOG_STATS = false;
        Constants.VERBOSE = false;
        Constants.TURN_TIME_LIMITED = false;
        Constants.MAX_TURNS_CAPITALS = opts.maxTurns;
    }

    private static void runTrainingGames(Options opts, int iteration, boolean selfPlayOnly,
                                         SftTrajectoryWriter trajectoryWriter) {
        int nGames = selfPlayOnly && !opts.selfPlayOnly ? opts.selfPlayGamesAfterTarget : opts.gamesPerIteration;
        for (int gameIdx = 0; gameIdx < nGames; gameIdx++) {
            long seed = opts.seed + iteration * 10000L + gameIdx * 31L;
            String opponent;
            if (selfPlayOnly) {
                opponent = "AZ";
            } else if (gameIdx % 3 == 0) {
                opponent = "SIMPLE";
            } else if (gameIdx % 3 == 1) {
                opponent = "OSLA";
            } else {
                opponent = "AZ";
            }

            int episode = trainingEpisode(opts, iteration, gameIdx);
            JSONObject setup = setupMetadata(opts, iteration, gameIdx, seed, opponent);
            Agent a = recording("AZ", newAlphaZero(seed, opts), opts, trajectoryWriter,
                    setup, episode, 0, seed + 1);
            Agent b = recording(opponent, newAgent(opponent, seed + 2, opts), opts, trajectoryWriter,
                    setup, episode, 1, seed + 3);
            runOneGame(a, b, seed, seed + 17);
            System.out.println("data game " + (gameIdx + 1) + "/" + nGames + " vs " + opponent);
        }
    }

    private static int trainingEpisode(Options opts, int iteration, int gameIdx) {
        return (iteration - 1) * Math.max(opts.gamesPerIteration, opts.selfPlayGamesAfterTarget) + gameIdx;
    }

    private static JSONObject setupMetadata(Options opts, int iteration, int gameIdx, long seed, String opponent) {
        JSONObject obj = new JSONObject();
        obj.put("episode", trainingEpisode(opts, iteration, gameIdx));
        obj.put("iteration", iteration);
        obj.put("game_index", gameIdx);
        obj.put("level_seed", seed);
        obj.put("game_seed", seed + 17);
        obj.put("game_mode", CAPITALS.name());
        obj.put("players", "az_training");
        obj.put("map", "procedural");
        obj.put("random_tribes", false);
        obj.put("value_model", opts.modelPath);
        obj.put("policy_model", opts.policyPath);
        obj.put("search_calls", opts.searchFmCalls);
        obj.put("opponent_calls", opts.opponentFmCalls);
        obj.put("search_depth", opts.searchDepth);
        obj.put("opponent", opponent);

        JSONArray bots = new JSONArray();
        bots.put("AZ");
        bots.put(opponent);
        obj.put("seat_bots", bots);

        JSONArray tribeNames = new JSONArray();
        tribeNames.put(XIN_XI.name());
        tribeNames.put(IMPERIUS.name());
        obj.put("seat_tribes", tribeNames);
        obj.put("target_tribe", "recorded_seat");
        return obj;
    }

    private static RecordingAgent recording(String botName, Agent agent, Options opts,
                                            SftTrajectoryWriter trajectoryWriter, JSONObject setupMetadata,
                                            int episode, int seat, long seed) {
        return new RecordingAgent(agent, botName, opts.dataPath, opts.policyDataPath, trajectoryWriter,
                setupMetadata, episode, seat, opts.sampleProbability, opts.maxExamplesPerGame,
                opts.trajectorySampleProbability, opts.maxTrajectoriesPerGame, seed);
    }

    private static MatchResult evaluate(Options opts, String opponent, int evalGames, long seedBase) {
        MatchResult result = new MatchResult("AZ", opponent);
        for (int i = 0; i < evalGames; i++) {
            long seed = seedBase + i * 997L;

            Game first = runOneGame(newAlphaZero(seed + 1, opts), newAgent(opponent, seed + 2, opts), seed, seed + 13);
            result.add(outcomeForTribe(first, XIN_XI));

            Game second = runOneGame(newAgent(opponent, seed + 3, opts), newAlphaZero(seed + 4, opts), seed, seed + 29);
            result.add(outcomeForTribe(second, IMPERIUS));
        }
        return result;
    }

    private static Game runOneGame(Agent a, Agent b, long levelSeed, long gameSeed) {
        ArrayList<Agent> players = new ArrayList<>();
        players.add(a);
        players.add(b);

        Game game = new Game();
        game.init(players, levelSeed, new Types.TRIBE[]{XIN_XI, IMPERIUS}, gameSeed, CAPITALS);
        game.run(null, null);
        return game;
    }

    private static double elapsedSeconds(long startedAt) {
        return (System.nanoTime() - startedAt) / 1_000_000_000.0;
    }

    private static EvalOutcome outcomeForTribe(Game game, Types.TRIBE tribeType) {
        Types.RESULT[] results = game.getWinnerStatus();
        Tribe[] tribes = game.getBoard().getTribes();
        int[] scores = game.getScores();
        for (int i = 0; i < tribes.length; i++) {
            if (tribes[i].getType() == tribeType) {
                int bestOther = Integer.MIN_VALUE;
                for (int j = 0; j < scores.length; j++) {
                    if (j != i) {
                        bestOther = Math.max(bestOther, scores[j]);
                    }
                }
                if (bestOther == Integer.MIN_VALUE) {
                    bestOther = 0;
                }
                return new EvalOutcome(results[i], scores[i], bestOther);
            }
        }
        return new EvalOutcome(Types.RESULT.LOSS, 0, 0);
    }

    private static Agent newAgent(String name, long seed, Options opts) {
        if ("AZ".equalsIgnoreCase(name) || "ALPHAZERO".equalsIgnoreCase(name)) {
            return newAlphaZero(seed, opts);
        }
        if ("REFERENCE_AZ".equalsIgnoreCase(name)) {
            return newReferenceAlphaZero(seed, opts);
        }
        if ("SIMPLE".equalsIgnoreCase(name)) {
            return new SimpleAgent(seed);
        }
        if ("OSLA".equalsIgnoreCase(name)) {
            OSLAParams params = new OSLAParams();
            params.stop_type = params.STOP_FMCALLS;
            params.heuristic_method = params.DIFF_HEURISTIC;
            params.num_fmcalls = opts.opponentFmCalls;
            return new OneStepLookAheadAgent(seed, params);
        }
        if ("RANDOM".equalsIgnoreCase(name)) {
            return new RandomAgent(seed);
        }
        throw new IllegalArgumentException("Unknown opponent: " + name);
    }

    private static AlphaZeroAgent newAlphaZero(long seed, Options opts) {
        AZParams params = new AZParams();
        params.modelPath = opts.modelPath;
        params.policyPath = opts.policyPath;
        params.num_fmcalls = opts.searchFmCalls;
        params.ROLLOUT_LENGTH = opts.searchDepth;
        params.maxActionsPerNode = opts.maxActionsPerNode;
        params.prefilterActions = opts.prefilterActions;
        params.heuristicBlend = opts.heuristicBlend;
        return new AlphaZeroAgent(seed, params);
    }

    private static AlphaZeroAgent newReferenceAlphaZero(long seed, Options opts) {
        AZParams params = new AZParams();
        params.modelPath = opts.referenceModelPath;
        params.policyPath = opts.referencePolicyPath;
        params.num_fmcalls = opts.referenceSearchFmCalls > 0 ? opts.referenceSearchFmCalls : opts.searchFmCalls;
        params.ROLLOUT_LENGTH = opts.searchDepth;
        params.maxActionsPerNode = opts.maxActionsPerNode;
        params.prefilterActions = opts.prefilterActions;
        params.heuristicBlend = opts.heuristicBlend;
        return new AlphaZeroAgent(seed, params);
    }

    private static class EvalOutcome {
        final Types.RESULT result;
        final int score;
        final int opponentScore;

        EvalOutcome(Types.RESULT result, int score, int opponentScore) {
            this.result = result;
            this.score = score;
            this.opponentScore = opponentScore;
        }

        int margin() {
            return score - opponentScore;
        }
    }

    private static class MatchResult {
        final String agent;
        final String opponent;
        int wins;
        int losses;
        int incomplete;
        int scoreTotal;
        int opponentScoreTotal;
        int marginTotal;

        MatchResult(String agent, String opponent) {
            this.agent = agent;
            this.opponent = opponent;
        }

        void add(EvalOutcome outcome) {
            scoreTotal += outcome.score;
            opponentScoreTotal += outcome.opponentScore;
            marginTotal += outcome.margin();
            if (outcome.result == Types.RESULT.WIN) {
                wins++;
            } else if (outcome.result == Types.RESULT.LOSS) {
                losses++;
            } else {
                incomplete++;
            }
        }

        double winRate() {
            int n = wins + losses + incomplete;
            return n == 0 ? 0.0 : (double) wins / n;
        }

        @Override
        public String toString() {
            int n = wins + losses + incomplete;
            double avgScore = n == 0 ? 0.0 : (double) scoreTotal / n;
            double avgOpponentScore = n == 0 ? 0.0 : (double) opponentScoreTotal / n;
            double avgMargin = n == 0 ? 0.0 : (double) marginTotal / n;
            return String.format("eval %s vs %s: W=%d L=%d I=%d N=%d winRate=%.3f avgScore=%.1f avgOppScore=%.1f avgMargin=%.1f",
                    agent, opponent, wins, losses, incomplete, n, winRate(),
                    avgScore, avgOpponentScore, avgMargin);
        }
    }

    private static class Options {
        String modelPath = "models/alphazero-value.tsv";
        String policyPath = "models/alphazero-policy.tsv";
        String dataPath = "training/alphazero-value-data.tsv";
        String policyDataPath = "training/alphazero-policy-data.tsv";
        String trajectoryPath = "training/alphazero-sft-trajectories.jsonl";
        String referenceModelPath = "models/alphazero-value.tsv";
        String referencePolicyPath = "models/alphazero-policy.tsv";
        String promptId = "alphazero-training-sft-v1";
        String actionFormat = "compact-full";
        int iterations = 4;
        int gamesPerIteration = 6;
        int selfPlayGamesAfterTarget = 6;
        int evalGames = 3;
        int epochs = 12;
        int policyEpochs = 8;
        int maxTrainingExamples = 25000;
        int maxExamplesPerGame = 160;
        int maxTurns = 50;
        int searchFmCalls = 900;
        int opponentFmCalls = 1200;
        int searchDepth = 18;
        int maxActionsPerNode = 32;
        int prefilterActions = 80;
        int maxPromptActions = 96;
        int maxTrajectoriesPerGame = 240;
        int trajectoryFlushEvery = 64;
        int referenceSearchFmCalls = 0;
        double learningRate = 0.015;
        double policyLearningRate = 0.010;
        double l2 = 0.0001;
        double targetWinRate = 0.55;
        double sampleProbability = 0.35;
        double trajectorySampleProbability = 1.0;
        double heuristicBlend = 0.35;
        long seed = 20260416L;
        boolean continueAfterTarget = false;
        boolean selfPlayAfterTarget = true;
        boolean selfPlayOnly = false;
        boolean recordTrajectories = true;
        boolean evalReference = false;

        static Options parse(String[] args) {
            Options opts = new Options();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if (!arg.startsWith("--")) {
                    continue;
                }
                String key = arg.substring(2);
                String value = i + 1 < args.length ? args[++i] : "";

                if ("model".equals(key)) opts.modelPath = value;
                else if ("policy".equals(key)) opts.policyPath = value;
                else if ("data".equals(key)) opts.dataPath = value;
                else if ("policy-data".equals(key)) opts.policyDataPath = value;
                else if ("trajectory-data".equals(key)) opts.trajectoryPath = value;
                else if ("reference-model".equals(key)) opts.referenceModelPath = value;
                else if ("reference-policy".equals(key)) opts.referencePolicyPath = value;
                else if ("prompt-id".equals(key)) opts.promptId = value;
                else if ("action-format".equals(key)) opts.actionFormat = value;
                else if ("iterations".equals(key)) opts.iterations = Integer.parseInt(value);
                else if ("games".equals(key)) opts.gamesPerIteration = Integer.parseInt(value);
                else if ("self-play-games".equals(key)) opts.selfPlayGamesAfterTarget = Integer.parseInt(value);
                else if ("eval-games".equals(key)) opts.evalGames = Integer.parseInt(value);
                else if ("epochs".equals(key)) opts.epochs = Integer.parseInt(value);
                else if ("policy-epochs".equals(key)) opts.policyEpochs = Integer.parseInt(value);
                else if ("max-examples".equals(key)) opts.maxTrainingExamples = Integer.parseInt(value);
                else if ("max-turns".equals(key)) opts.maxTurns = Integer.parseInt(value);
                else if ("search-calls".equals(key)) opts.searchFmCalls = Integer.parseInt(value);
                else if ("opponent-calls".equals(key)) opts.opponentFmCalls = Integer.parseInt(value);
                else if ("depth".equals(key)) opts.searchDepth = Integer.parseInt(value);
                else if ("max-actions".equals(key)) opts.maxActionsPerNode = Integer.parseInt(value);
                else if ("prefilter".equals(key)) opts.prefilterActions = Integer.parseInt(value);
                else if ("max-prompt-actions".equals(key)) opts.maxPromptActions = Integer.parseInt(value);
                else if ("max-trajectories-per-game".equals(key)) opts.maxTrajectoriesPerGame = Integer.parseInt(value);
                else if ("trajectory-flush-every".equals(key)) opts.trajectoryFlushEvery = Integer.parseInt(value);
                else if ("reference-calls".equals(key)) opts.referenceSearchFmCalls = Integer.parseInt(value);
                else if ("lr".equals(key)) opts.learningRate = Double.parseDouble(value);
                else if ("policy-lr".equals(key)) opts.policyLearningRate = Double.parseDouble(value);
                else if ("l2".equals(key)) opts.l2 = Double.parseDouble(value);
                else if ("target".equals(key)) opts.targetWinRate = Double.parseDouble(value);
                else if ("sample".equals(key)) opts.sampleProbability = Double.parseDouble(value);
                else if ("trajectory-sample".equals(key)) opts.trajectorySampleProbability = Double.parseDouble(value);
                else if ("heuristic-blend".equals(key)) opts.heuristicBlend = Double.parseDouble(value);
                else if ("seed".equals(key)) opts.seed = Long.parseLong(value);
                else if ("continue-after-target".equals(key)) opts.continueAfterTarget = Boolean.parseBoolean(value);
                else if ("self-play-after-target".equals(key)) opts.selfPlayAfterTarget = Boolean.parseBoolean(value);
                else if ("self-play-only".equals(key)) opts.selfPlayOnly = Boolean.parseBoolean(value);
                else if ("record-trajectories".equals(key)) opts.recordTrajectories = Boolean.parseBoolean(value);
                else if ("eval-reference".equals(key)) opts.evalReference = Boolean.parseBoolean(value);
                else {
                    throw new IllegalArgumentException("Unknown option --" + key);
                }
            }
            return opts;
        }
    }
}
