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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import static core.Types.GAME_MODE.CAPITALS;
import static core.Types.TRIBE.IMPERIUS;
import static core.Types.TRIBE.XIN_XI;

public class AlphaZeroTrainer {

    public static void main(String[] args) {
        Options opts = Options.parse(args);
        configureHeadless(opts);

        System.out.println("AlphaZero-style value training");
        System.out.println("networkType=" + opts.networkType);
        if (!opts.referenceNetworkType.isEmpty()) {
            System.out.println("referenceNetworkType=" + opts.referenceNetworkType);
        }
        System.out.println("valueModel=" + opts.modelPath + " valueData=" + opts.dataPath);
        System.out.println("policyModel=" + opts.policyPath + " policyData=" + opts.policyDataPath);
        if (!opts.actionPolicyPath.isEmpty()) {
            System.out.println("actionPolicyModel=" + opts.actionPolicyPath
                    + " actionPolicyData=" + opts.actionPolicyDataPath
                    + " actionPolicyLogitWeight=" + opts.actionPolicyLogitWeight);
        }
        System.out.println("policyTargets=" + opts.policyTargetMode);
        System.out.println("bestValueModel=" + opts.bestModelPath + " bestPolicyModel=" + opts.bestPolicyPath
                + " restoreBestOnRegression=" + opts.restoreBestOnRegression);
        System.out.println("baselineInitialCheckpoint=" + opts.baselineInitialCheckpoint);
        System.out.println("gatePlayerCountFloor=" + opts.gatePlayerCountFloor);
        System.out.println("referenceRefreshInterval=" + opts.referenceRefreshInterval);
        if (!opts.leagueDir.isEmpty()) {
            System.out.println("leagueDir=" + opts.leagueDir
                    + " leagueMaxSnapshots=" + opts.leagueMaxSnapshots
                    + " leagueSnapshotInterval=" + opts.leagueSnapshotInterval);
        }
        System.out.println("cpuct=" + opts.cpuct + " priorTemperature=" + opts.priorTemperature);
        System.out.println("trainingRootNoise=" + opts.rootNoiseFraction
                + " rootDirichletAlpha=" + opts.rootDirichletAlpha
                + " rootGumbelScale=" + opts.rootGumbelScale);
        System.out.println("rootValueSelectionWeight=" + opts.rootValueSelectionWeight
                + " improvedPolicyValueWeight=" + opts.improvedPolicyValueWeight
                + " improvedPolicyTemperature=" + opts.improvedPolicyTemperature);
        System.out.println("trainingVisitSamplingTemp=" + opts.visitSamplingTemperature
                + " untilTick=" + opts.visitSamplingUntilTick);
        System.out.println("randomLevelSeeds=" + opts.randomLevelSeeds
                + " mapSnapshotDir=" + opts.mapSnapshotDir);
        System.out.println("randomTribes=" + opts.randomTribes
                + " randomPlayerCount=" + opts.randomPlayerCount
                + " playerRange=" + opts.minPlayers + "-" + opts.maxPlayers
                + " stratifiedTrainingPlayerCounts=" + opts.stratifiedTrainingPlayerCounts
                + " stratifiedEvalPlayerCounts=" + opts.stratifiedEvalPlayerCounts);
        System.out.println("trainingOpponentMode=" + opts.trainingOpponentMode);
        System.out.println("valuePositionBlend=" + opts.valuePositionBlend
                + " terminalPositionBlend=" + opts.terminalPositionBlend
                + " rankValueBlend=" + opts.rankValueBlend
                + " survivalValueBlend=" + opts.survivalValueBlend
                + " searchPositionBlend=" + opts.positionBlend
                + " advisorMargin=" + opts.advisorOverrideMargin
                + " opponentAdversaryWeight=" + opts.opponentAdversaryWeight);
        System.out.println("pureAz=" + opts.pureAz
                + " tacticalShortcuts=" + opts.tacticalShortcuts
                + " advisorOverride=" + opts.advisorOverride
                + " staticPriors=" + opts.staticPriors
                + " nextStateValuePrior=" + opts.nextStateValuePrior
                + " learnedValueOnly=" + opts.learnedValueOnly
                + " prefilterByStaticScore=" + opts.prefilterByStaticScore);
        if (opts.recordTrajectories) {
            System.out.println("sftTrajectories=" + opts.trajectoryPath);
        }
        System.out.println("valueRecordingBots=" + opts.valueRecordingBotsCsv);

        boolean targetReached = false;
        CheckpointScore bestCheckpoint = null;
        long startedAt = System.nanoTime();
        try (SftTrajectoryWriter trajectoryWriter = newTrajectoryWriter(opts)) {
            if (opts.baselineInitialCheckpoint) {
                MatchResult simple = evaluate(opts, "SIMPLE", opts.evalGames, opts.seed + 910000000L);
                MatchResult osla = evaluate(opts, "OSLA", opts.evalGames, opts.seed + 920000000L);
                System.out.println("initial " + simple);
                System.out.println("initial " + osla);
                MatchResult reference = null;
                if (opts.evalReference) {
                    reference = evaluate(opts, "REFERENCE_AZ", opts.evalGames, opts.seed + 930000000L);
                    System.out.println("initial " + reference);
                }
                bestCheckpoint = CheckpointScore.from(simple, osla, reference, opts.gatePlayerCountFloor);
                saveBestCheckpoint(opts);
                saveLeagueSnapshot(opts, "initial", bestCheckpoint);
                System.out.println(bestCheckpoint.format("initial best checkpoint"));

                boolean referencePassed = reference == null || reference.winRate() >= opts.targetWinRate;
                targetReached = simple.winRate() >= opts.targetWinRate && osla.winRate() >= opts.targetWinRate
                        && referencePassed;
                if (targetReached) {
                    System.out.println("initial checkpoint reached target; training starts in self-play mode");
                }
            }

            for (int iteration = 1; iteration <= opts.iterations; iteration++) {
                System.out.println("=== iteration " + iteration + " ===");
                runTrainingGames(opts, iteration,
                        opts.selfPlayOnly || (targetReached && opts.selfPlayAfterTarget), trajectoryWriter);

                ArrayList<ValueTrainingExample> examples = ValueDataset.load(opts.dataPath, opts.maxTrainingExamples);
                ValueModel vf = ModelFactory.loadValue(opts.networkType, opts.modelPath);
                if (!examples.isEmpty()) {
                    double loss = vf.train(examples, opts.epochs, opts.learningRate, opts.l2, opts.seed + iteration);
                    vf.save(opts.modelPath);
                    System.out.printf("trained value on %d examples; mse=%.5f%n", examples.size(), loss);
                } else {
                    System.out.println("trained value on 0 examples; skipped");
                }

                ArrayList<PolicyTrainingExample> policyExamples = PolicyDataset.load(opts.policyDataPath, opts.maxTrainingExamples);
                PolicyModel policy = ModelFactory.loadPolicy(opts.networkType, opts.policyPath);
                if (!policyExamples.isEmpty()) {
                    double policyLoss = policy.train(policyExamples, opts.policyEpochs, opts.policyLearningRate, opts.l2, opts.seed + 991 * iteration);
                    policy.save(opts.policyPath);
                    System.out.printf("trained policy on %d examples; xent=%.5f%n", policyExamples.size(), policyLoss);
                } else {
                    System.out.println("trained policy on 0 examples; skipped");
                }

                if (!opts.actionPolicyPath.isEmpty()) {
                    ArrayList<ActionPolicyTrainingExample> actionPolicyExamples =
                            ActionPolicyDataset.load(opts.actionPolicyDataPath, opts.maxTrainingExamples);
                    ActionPolicyModel actionPolicy = ModelFactory.loadActionPolicy(opts.actionPolicyPath);
                    if (!actionPolicyExamples.isEmpty()) {
                        double actionPolicyLoss = actionPolicy.train(actionPolicyExamples, opts.policyEpochs,
                                opts.policyLearningRate, opts.l2, opts.seed + 1999L * iteration);
                        actionPolicy.save(opts.actionPolicyPath);
                        System.out.printf("trained action policy on %d examples; xent=%.5f%n",
                                actionPolicyExamples.size(), actionPolicyLoss);
                    } else {
                        System.out.println("trained action policy on 0 examples; skipped");
                    }
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

                CheckpointScore checkpoint = CheckpointScore.from(simple, osla, reference, opts.gatePlayerCountFloor);
                System.out.println(checkpoint.format("checkpoint candidate"));
                if (bestCheckpoint == null || checkpoint.betterThan(bestCheckpoint)) {
                    saveBestCheckpoint(opts);
                    bestCheckpoint = checkpoint;
                    saveLeagueSnapshot(opts, String.format("iter-%03d-best", iteration), checkpoint);
                    System.out.println(checkpoint.format("best checkpoint updated"));
                } else if (opts.restoreBestOnRegression) {
                    restoreBestCheckpoint(opts);
                    System.out.printf("restored best checkpoint: currentScore=%.3f currentMarginFloor=%.1f "
                                    + "bestScore=%.3f bestMarginFloor=%.1f%n",
                            checkpoint.score, checkpoint.marginFloor, bestCheckpoint.score,
                            bestCheckpoint.marginFloor);
                }
                if (opts.referenceRefreshInterval > 0 && iteration % opts.referenceRefreshInterval == 0) {
                    refreshReferenceCheckpoint(opts);
                    System.out.printf("refreshed reference checkpoint: iteration=%d interval=%d%n",
                            iteration, opts.referenceRefreshInterval);
                }
                if (opts.leagueSnapshotInterval > 0 && iteration % opts.leagueSnapshotInterval == 0) {
                    saveLeagueSnapshot(opts, String.format("iter-%03d-periodic", iteration), checkpoint);
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
            throw new RuntimeException("AlphaZero trainer I/O failure", e);
        }
    }

    private static void saveBestCheckpoint(Options opts) throws IOException {
        copyFileIfConfigured(opts.modelPath, opts.bestModelPath);
        copyFileIfConfigured(opts.policyPath, opts.bestPolicyPath);
        copyFileIfConfigured(opts.actionPolicyPath, opts.bestActionPolicyPath);
    }

    private static void restoreBestCheckpoint(Options opts) throws IOException {
        if (Files.exists(Path.of(opts.bestModelPath))) {
            copyFile(opts.bestModelPath, opts.modelPath);
        }
        if (Files.exists(Path.of(opts.bestPolicyPath))) {
            copyFile(opts.bestPolicyPath, opts.policyPath);
        }
        if (!opts.actionPolicyPath.isEmpty() && Files.exists(Path.of(opts.bestActionPolicyPath))) {
            copyFile(opts.bestActionPolicyPath, opts.actionPolicyPath);
        }
    }

    private static void refreshReferenceCheckpoint(Options opts) throws IOException {
        if (Files.exists(Path.of(opts.bestModelPath))) {
            copyFile(opts.bestModelPath, opts.referenceModelPath);
        }
        if (Files.exists(Path.of(opts.bestPolicyPath))) {
            copyFile(opts.bestPolicyPath, opts.referencePolicyPath);
        }
        if (!opts.actionPolicyPath.isEmpty() && Files.exists(Path.of(opts.bestActionPolicyPath))) {
            copyFile(opts.bestActionPolicyPath, opts.referenceActionPolicyPath);
        }
    }

    private static void saveLeagueSnapshot(Options opts, String tag, CheckpointScore score) throws IOException {
        if (opts.leagueDir == null || opts.leagueDir.trim().isEmpty()) {
            return;
        }
        Path dir = Path.of(opts.leagueDir);
        Files.createDirectories(dir);

        String base = safeFileName(tag) + "-" + System.currentTimeMillis();
        Path value = dir.resolve(base + "-value.tsv");
        Path policy = dir.resolve(base + "-policy.tsv");
        Path actionPolicy = dir.resolve(base + "-action-policy.tsv");

        copyFile(opts.modelPath, value.toString());
        copyFile(opts.policyPath, policy.toString());
        if (!opts.actionPolicyPath.isEmpty() && Files.exists(Path.of(opts.actionPolicyPath))) {
            copyFile(opts.actionPolicyPath, actionPolicy.toString());
        }

        JSONObject manifest = new JSONObject();
        manifest.put("created_at_ms", System.currentTimeMillis());
        manifest.put("tag", tag);
        manifest.put("network_type", opts.networkType);
        manifest.put("value_model", value.getFileName().toString());
        manifest.put("policy_model", policy.getFileName().toString());
        if (Files.exists(actionPolicy)) {
            manifest.put("action_policy_model", actionPolicy.getFileName().toString());
        }
        if (score != null) {
            manifest.put("score", score.score);
            manifest.put("margin_floor", score.marginFloor);
            manifest.put("margin_average", score.marginAverage);
            manifest.put("simple_win_rate", score.simpleWinRate);
            manifest.put("osla_win_rate", score.oslaWinRate);
            if (!Double.isNaN(score.referenceWinRate)) {
                manifest.put("reference_win_rate", score.referenceWinRate);
                manifest.put("reference_margin", score.referenceMargin);
            }
        }
        Files.write(dir.resolve("league-manifest.jsonl"), (manifest.toString() + "\n").getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        pruneLeagueSnapshots(opts, dir);
        System.out.printf("saved league snapshot: %s%n", value.getFileName());
    }

    private static void pruneLeagueSnapshots(Options opts, Path dir) throws IOException {
        if (opts.leagueMaxSnapshots <= 0) {
            return;
        }
        ArrayList<Path> values = leagueValueFiles(dir);
        values.sort((a, b) -> {
            try {
                return Files.getLastModifiedTime(a).compareTo(Files.getLastModifiedTime(b));
            } catch (IOException e) {
                return a.getFileName().toString().compareTo(b.getFileName().toString());
            }
        });
        while (values.size() > opts.leagueMaxSnapshots) {
            Path value = values.remove(0);
            String base = stripSuffix(value.getFileName().toString(), "-value.tsv");
            Files.deleteIfExists(value);
            Files.deleteIfExists(dir.resolve(base + "-policy.tsv"));
            Files.deleteIfExists(dir.resolve(base + "-action-policy.tsv"));
        }
    }

    private static ArrayList<Path> leagueValueFiles(Path dir) throws IOException {
        ArrayList<Path> values = new ArrayList<>();
        if (!Files.isDirectory(dir)) {
            return values;
        }
        try (java.util.stream.Stream<Path> stream = Files.list(dir)) {
            stream.filter(path -> path.getFileName().toString().endsWith("-value.tsv"))
                    .forEach(values::add);
        }
        return values;
    }

    private static String safeFileName(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "snapshot";
        }
        return value.replaceAll("[^A-Za-z0-9_.-]", "_");
    }

    private static String stripSuffix(String value, String suffix) {
        return value.endsWith(suffix) ? value.substring(0, value.length() - suffix.length()) : value;
    }

    private static void copyFileIfConfigured(String source, String target) throws IOException {
        if (source == null || source.trim().isEmpty() || target == null || target.trim().isEmpty()) {
            return;
        }
        if (!Files.exists(Path.of(source))) {
            return;
        }
        copyFile(source, target);
    }

    private static void copyFile(String source, String target) throws IOException {
        Path sourcePath = Path.of(source);
        Path targetPath = Path.of(target);
        if (sourcePath.toAbsolutePath().normalize().equals(targetPath.toAbsolutePath().normalize())) {
            return;
        }
        Path parent = targetPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
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
            long levelSeed = opts.nextLevelSeed(seed);
            long gameSeed = seed + 17;
            String opponent;
            if (selfPlayOnly) {
                opponent = "AZ";
            } else {
                opponent = opts.trainingOpponent(gameIdx);
            }

            int episode = trainingEpisode(opts, iteration, gameIdx);
            int forcedPlayerCount = opts.stratifiedTrainingPlayerCounts
                    ? opts.trainingPlayerCountForIndex(gameIdx) : 0;
            GameSetup gameSetup = trainingSetup(opts, seed, opponent, selfPlayOnly, forcedPlayerCount);
            JSONObject setup = setupMetadata(opts, iteration, gameIdx, episode, levelSeed, gameSeed,
                    gameSetup, "train");
            ArrayList<Agent> players = new ArrayList<>();
            for (int seat = 0; seat < gameSetup.bots.length; seat++) {
                String bot = gameSetup.bots[seat];
                players.add(recording(bot, newAgent(bot, seed + 2 + seat * 17L, opts, true),
                        opts, trajectoryWriter, setup, episode, seat, seed + 3 + seat * 19L));
            }
            runOneGame(players, gameSetup.tribes, levelSeed, gameSeed, opts, "train",
                    episode, gameIdx, gameSetup.opponentLabel(), setup);
            System.out.println("data game " + (gameIdx + 1) + "/" + nGames
                    + " vs " + gameSetup.opponentLabel()
                    + " players=" + gameSetup.bots.length
                    + " tribes=" + gameSetup.tribeLabel());
        }
    }

    private static int trainingEpisode(Options opts, int iteration, int gameIdx) {
        return (iteration - 1) * Math.max(opts.gamesPerIteration, opts.selfPlayGamesAfterTarget) + gameIdx;
    }

    private static JSONObject setupMetadata(Options opts, int iteration, int gameIdx, int episode,
                                            long levelSeed, long gameSeed,
                                            GameSetup setup, String mapSplit) {
        JSONObject obj = new JSONObject();
        obj.put("episode", episode);
        obj.put("iteration", iteration);
        obj.put("game_index", gameIdx);
        obj.put("level_seed", levelSeed);
        obj.put("game_seed", gameSeed);
        obj.put("game_mode", CAPITALS.name());
        obj.put("players", opts.randomPlayerCount ? "randomized" : "az_training");
        obj.put("player_count", setup.bots.length);
        obj.put("map", "procedural");
        obj.put("random_level_seeds", opts.randomLevelSeeds);
        obj.put("map_snapshot_dir", opts.mapSnapshotDir);
        obj.put("map_snapshot_split", mapSplit);
        obj.put("map_snapshot_file",
                MapSnapshotWriter.fileName(mapSplit, episode, gameIdx, levelSeed, gameSeed));
        obj.put("random_tribes", opts.randomTribes);
        obj.put("random_player_count", opts.randomPlayerCount);
        obj.put("stratified_training_player_counts", opts.stratifiedTrainingPlayerCounts);
        obj.put("stratified_eval_player_counts", opts.stratifiedEvalPlayerCounts);
        obj.put("value_model", opts.modelPath);
        obj.put("policy_model", opts.policyPath);
        obj.put("search_calls", opts.searchFmCalls);
        obj.put("opponent_calls", opts.opponentFmCalls);
        obj.put("search_depth", opts.searchDepth);
        obj.put("opponent", setup.opponentLabel());
        obj.put("training_opponents", opts.trainingOpponentsCsv);
        obj.put("training_opponent_mode", opts.trainingOpponentMode);
        obj.put("policy_targets", opts.policyTargetMode);
        obj.put("value_recording_bots", opts.valueRecordingBotsCsv);
        obj.put("policy_logit_weight", opts.policyLogitWeight);
        obj.put("root_noise_fraction", opts.rootNoiseFraction);
        obj.put("root_dirichlet_alpha", opts.rootDirichletAlpha);
        obj.put("root_gumbel_scale", opts.rootGumbelScale);
        obj.put("root_value_selection_weight", opts.rootValueSelectionWeight);
        obj.put("improved_policy_value_weight", opts.improvedPolicyValueWeight);
        obj.put("improved_policy_temperature", opts.improvedPolicyTemperature);
        obj.put("visit_sampling_temperature", opts.visitSamplingTemperature);
        obj.put("visit_sampling_until_tick", opts.visitSamplingUntilTick);
        obj.put("value_position_blend", opts.valuePositionBlend);
        obj.put("terminal_position_blend", opts.terminalPositionBlend);
        obj.put("rank_value_blend", opts.rankValueBlend);
        obj.put("survival_value_blend", opts.survivalValueBlend);
        obj.put("search_position_blend", opts.positionBlend);
        obj.put("advisor_override_margin", opts.advisorOverrideMargin);
        obj.put("opponent_adversary_weight", opts.opponentAdversaryWeight);
        obj.put("disagreement_heuristic_blend", opts.disagreementHeuristicBlend);
        obj.put("disagreement_heuristic_threshold", opts.disagreementHeuristicThreshold);
        obj.put("reference_refresh_interval", opts.referenceRefreshInterval);
        obj.put("league_dir", opts.leagueDir);
        obj.put("league_max_snapshots", opts.leagueMaxSnapshots);
        obj.put("league_snapshot_interval", opts.leagueSnapshotInterval);
        obj.put("gate_player_count_floor", opts.gatePlayerCountFloor);

        JSONArray bots = new JSONArray();
        for (String bot : setup.bots) {
            bots.put(bot);
        }
        obj.put("seat_bots", bots);

        JSONArray tribeNames = new JSONArray();
        for (Types.TRIBE tribe : setup.tribes) {
            tribeNames.put(tribe.name());
        }
        obj.put("seat_tribes", tribeNames);
        obj.put("target_az_seat", setup.targetSeat);
        obj.put("target_tribe", setup.targetTribe().name());
        return obj;
    }

    private static GameSetup trainingSetup(Options opts, long seed, String scheduledOpponent,
                                           boolean selfPlayOnly) {
        return trainingSetup(opts, seed, scheduledOpponent, selfPlayOnly, 0);
    }

    private static GameSetup trainingSetup(Options opts, long seed, String scheduledOpponent,
                                           boolean selfPlayOnly, int forcedPlayerCount) {
        if (forcedPlayerCount <= 0 && !opts.randomPlayerCount && !opts.randomTribes) {
            return new GameSetup(new String[]{"AZ", scheduledOpponent},
                    new Types.TRIBE[]{XIN_XI, IMPERIUS}, 0);
        }

        Random rnd = new Random(seed ^ 0x9E3779B97F4A7C15L);
        int playerCount = forcedPlayerCount > 0 ? forcedPlayerCount
                : opts.randomPlayerCount ? opts.randomPlayerCount(rnd) : 2;
        Types.TRIBE[] tribes = opts.randomTribes ? randomTribes(playerCount, rnd)
                : fixedTribes(playerCount);
        String[] bots = new String[playerCount];
        int targetSeat = rnd.nextInt(playerCount);
        boolean scheduledPlaced = false;
        for (int seat = 0; seat < playerCount; seat++) {
            if (selfPlayOnly || seat == targetSeat) {
                bots[seat] = "AZ";
            } else if (opts.useScheduledTrainingOpponentForAllSeats()
                    || "AZ".equalsIgnoreCase(scheduledOpponent)) {
                bots[seat] = scheduledOpponent;
            } else if (opts.useHybridTrainingOpponentMode() && !scheduledPlaced) {
                bots[seat] = scheduledOpponent;
                scheduledPlaced = true;
            } else {
                bots[seat] = opts.randomTrainingOpponent(rnd);
            }
        }
        return new GameSetup(bots, tribes, targetSeat);
    }

    private static GameSetup evalSetup(Options opts, String opponent, long seed) {
        return evalSetup(opts, opponent, seed, 0);
    }

    private static GameSetup evalSetup(Options opts, String opponent, long seed, int forcedPlayerCount) {
        if (forcedPlayerCount <= 0 && !opts.randomPlayerCount && !opts.randomTribes) {
            return new GameSetup(new String[]{"AZ", opponent},
                    new Types.TRIBE[]{XIN_XI, IMPERIUS}, 0);
        }

        Random rnd = new Random(seed ^ 0xD1B54A32D192ED03L);
        int playerCount = forcedPlayerCount > 0 ? forcedPlayerCount
                : opts.randomPlayerCount ? opts.randomPlayerCount(rnd) : 2;
        Types.TRIBE[] tribes = opts.randomTribes ? randomTribes(playerCount, rnd)
                : fixedTribes(playerCount);
        String[] bots = new String[playerCount];
        int targetSeat = rnd.nextInt(playerCount);
        for (int seat = 0; seat < playerCount; seat++) {
            bots[seat] = seat == targetSeat ? "AZ" : opponent;
        }
        return new GameSetup(bots, tribes, targetSeat);
    }

    private static Types.TRIBE[] fixedTribes(int count) {
        Types.TRIBE[] available = Types.TRIBE.values();
        Types.TRIBE[] tribes = new Types.TRIBE[count];
        for (int i = 0; i < count; i++) {
            tribes[i] = available[i % available.length];
        }
        return tribes;
    }

    private static Types.TRIBE[] randomTribes(int count, Random rnd) {
        ArrayList<Types.TRIBE> tribes = new ArrayList<>();
        Collections.addAll(tribes, Types.TRIBE.values());
        Collections.shuffle(tribes, rnd);
        Types.TRIBE[] out = new Types.TRIBE[count];
        for (int i = 0; i < count; i++) {
            out[i] = tribes.get(i);
        }
        return out;
    }

    private static RecordingAgent recording(String botName, Agent agent, Options opts,
                                            SftTrajectoryWriter trajectoryWriter, JSONObject setupMetadata,
                                            int episode, int seat, long seed) {
        return new RecordingAgent(agent, botName, opts.dataPath, opts.policyDataPath, opts.actionPolicyDataPath,
                trajectoryWriter, setupMetadata, episode, seat, opts.sampleProbability, opts.maxExamplesPerGame,
                opts.trajectorySampleProbability, opts.maxTrajectoriesPerGame, opts.policyTargetMode,
                opts.valuePositionBlend, opts.terminalPositionBlend, opts.rankValueBlend,
                opts.survivalValueBlend, opts.shouldRecordValueFor(botName), seed);
    }

    private static MatchResult evaluate(Options opts, String opponent, int evalGames, long seedBase) {
        MatchResult result = new MatchResult("AZ", opponent);
        for (int i = 0; i < evalGames; i++) {
            long seed = seedBase + i * 997L;
            int forcedPlayerCount = opts.stratifiedEvalPlayerCounts
                    ? opts.evalPlayerCountForIndex(i) : 0;

            long firstLevelSeed = opts.nextLevelSeed(seed);
            GameSetup firstSetup = evalSetup(opts, opponent, seed + 13, forcedPlayerCount);
            JSONObject firstMetadata = setupMetadata(opts, 0, i * 2, i,
                    firstLevelSeed, seed + 13, firstSetup, "eval-" + opponent + "-az-first");
            Game first = runOneGame(firstSetup.agents(seed + 1, opts), firstSetup.tribes,
                    firstLevelSeed, seed + 13, opts, "eval-" + opponent + "-az-first",
                    i, i * 2, opponent, firstMetadata);
            result.add(outcomeForTribe(first, firstSetup.targetTribe()), firstSetup.bots.length);

            long secondLevelSeed = opts.randomLevelSeeds ? opts.nextLevelSeed(seed + 1) : seed;
            GameSetup secondSetup;
            if (forcedPlayerCount > 0 || opts.randomPlayerCount || opts.randomTribes) {
                secondSetup = evalSetup(opts, opponent, seed + 29, forcedPlayerCount);
            } else {
                secondSetup = new GameSetup(new String[]{opponent, "AZ"},
                        new Types.TRIBE[]{XIN_XI, IMPERIUS}, 1);
            }
            JSONObject secondMetadata = setupMetadata(opts, 0, i * 2 + 1, i,
                    secondLevelSeed, seed + 29, secondSetup, "eval-" + opponent + "-az-second");
            Game second = runOneGame(secondSetup.agents(seed + 3, opts), secondSetup.tribes,
                    secondLevelSeed, seed + 29, opts, "eval-" + opponent + "-az-second",
                    i, i * 2 + 1, opponent, secondMetadata);
            result.add(outcomeForTribe(second, secondSetup.targetTribe()), secondSetup.bots.length);
        }
        return result;
    }

    private static Game runOneGame(Agent a, Agent b, long levelSeed, long gameSeed,
                                   Options opts, String mapSplit, int episode,
                                   int gameIndex, String opponent) {
        ArrayList<Agent> players = new ArrayList<>();
        players.add(a);
        players.add(b);
        return runOneGame(players, new Types.TRIBE[]{XIN_XI, IMPERIUS}, levelSeed, gameSeed,
                opts, mapSplit, episode, gameIndex, opponent, null);
    }

    private static Game runOneGame(ArrayList<Agent> players, Types.TRIBE[] tribes, long levelSeed,
                                   long gameSeed, Options opts, String mapSplit, int episode,
                                   int gameIndex, String opponent, JSONObject setupMetadata) {
        Game game = new Game();
        game.init(players, levelSeed, tribes, gameSeed, CAPITALS);
        MapSnapshotWriter.write(opts.mapSnapshotDir, mapSplit, episode, gameIndex,
                levelSeed, gameSeed, opponent, game, setupMetadata);
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
        return newAgent(name, seed, opts, false);
    }

    private static Agent newAgent(String name, long seed, Options opts, boolean training) {
        if ("AZ".equalsIgnoreCase(name) || "ALPHAZERO".equalsIgnoreCase(name)) {
            return newAlphaZero(seed, opts, training);
        }
        if ("REFERENCE_AZ".equalsIgnoreCase(name)) {
            return newReferenceAlphaZero(seed, opts);
        }
        if ("LEAGUE_AZ".equalsIgnoreCase(name)) {
            return newLeagueAlphaZero(seed, opts);
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
        return newAlphaZero(seed, opts, false);
    }

    private static AlphaZeroAgent newAlphaZero(long seed, Options opts, boolean training) {
        AZParams params = new AZParams();
        params.modelPath = opts.modelPath;
        params.policyPath = opts.policyPath;
        params.actionPolicyPath = opts.actionPolicyPath;
        params.networkType = opts.networkType;
        params.num_fmcalls = opts.searchFmCalls;
        params.ROLLOUT_LENGTH = opts.searchDepth;
        params.cpuct = opts.cpuct;
        params.priorTemperature = opts.priorTemperature;
        params.maxActionsPerNode = opts.maxActionsPerNode;
        params.prefilterActions = opts.prefilterActions;
        params.heuristicBlend = opts.heuristicBlend;
        params.disagreementHeuristicBlend = opts.disagreementHeuristicBlend;
        params.disagreementHeuristicThreshold = opts.disagreementHeuristicThreshold;
        params.positionBlend = opts.positionBlend;
        params.advisorOverrideMargin = opts.advisorOverrideMargin;
        params.opponentAdversaryWeight = opts.opponentAdversaryWeight;
        params.policyLogitWeight = opts.policyLogitWeight;
        params.actionPolicyLogitWeight = opts.actionPolicyLogitWeight;
        params.tacticalShortcuts = opts.tacticalShortcuts;
        params.advisorOverride = opts.advisorOverride;
        params.staticPriors = opts.staticPriors;
        params.nextStateValuePrior = opts.nextStateValuePrior;
        params.learnedValueOnly = opts.learnedValueOnly;
        params.prefilterByStaticScore = opts.prefilterByStaticScore;
        params.rootNoiseFraction = training ? opts.rootNoiseFraction : 0.0;
        params.rootDirichletAlpha = opts.rootDirichletAlpha;
        params.rootGumbelScale = training ? opts.rootGumbelScale : 0.0;
        params.rootValueSelectionWeight = opts.rootValueSelectionWeight;
        params.improvedPolicyValueWeight = opts.improvedPolicyValueWeight;
        params.improvedPolicyTemperature = opts.improvedPolicyTemperature;
        params.visitSamplingTemperature = training ? opts.visitSamplingTemperature : 0.0;
        params.visitSamplingUntilTick = opts.visitSamplingUntilTick;
        return new AlphaZeroAgent(seed, params);
    }

    private static AlphaZeroAgent newReferenceAlphaZero(long seed, Options opts) {
        return newReferenceAlphaZero(seed, opts, opts.referenceModelPath, opts.referencePolicyPath,
                opts.referenceActionPolicyPath);
    }

    private static AlphaZeroAgent newLeagueAlphaZero(long seed, Options opts) {
        LeagueSnapshot snapshot = selectLeagueSnapshot(opts, seed);
        if (snapshot == null) {
            return newReferenceAlphaZero(seed, opts);
        }
        return newReferenceAlphaZero(seed, opts, snapshot.valuePath, snapshot.policyPath,
                snapshot.actionPolicyPath);
    }

    private static AlphaZeroAgent newReferenceAlphaZero(long seed, Options opts,
                                                       String valuePath, String policyPath,
                                                       String actionPolicyPath) {
        AZParams params = new AZParams();
        params.modelPath = valuePath;
        params.policyPath = policyPath;
        params.actionPolicyPath = actionPolicyPath;
        params.networkType = opts.referenceNetworkType.isEmpty() ? opts.networkType : opts.referenceNetworkType;
        params.num_fmcalls = opts.referenceSearchFmCalls > 0 ? opts.referenceSearchFmCalls : opts.searchFmCalls;
        params.ROLLOUT_LENGTH = opts.searchDepth;
        params.cpuct = opts.cpuct;
        params.priorTemperature = opts.priorTemperature;
        params.maxActionsPerNode = opts.maxActionsPerNode;
        params.prefilterActions = opts.prefilterActions;
        params.heuristicBlend = opts.heuristicBlend;
        params.disagreementHeuristicBlend = opts.disagreementHeuristicBlend;
        params.disagreementHeuristicThreshold = opts.disagreementHeuristicThreshold;
        params.positionBlend = opts.positionBlend;
        params.advisorOverrideMargin = opts.advisorOverrideMargin;
        params.opponentAdversaryWeight = opts.opponentAdversaryWeight;
        params.policyLogitWeight = opts.policyLogitWeight;
        params.actionPolicyLogitWeight = opts.actionPolicyLogitWeight;
        params.tacticalShortcuts = opts.tacticalShortcuts;
        params.advisorOverride = opts.advisorOverride;
        params.staticPriors = opts.staticPriors;
        params.nextStateValuePrior = opts.nextStateValuePrior;
        params.learnedValueOnly = opts.learnedValueOnly;
        params.prefilterByStaticScore = opts.prefilterByStaticScore;
        params.rootNoiseFraction = 0.0;
        params.rootDirichletAlpha = opts.rootDirichletAlpha;
        params.rootGumbelScale = 0.0;
        params.rootValueSelectionWeight = opts.rootValueSelectionWeight;
        params.improvedPolicyValueWeight = opts.improvedPolicyValueWeight;
        params.improvedPolicyTemperature = opts.improvedPolicyTemperature;
        params.visitSamplingTemperature = 0.0;
        params.visitSamplingUntilTick = opts.visitSamplingUntilTick;
        return new AlphaZeroAgent(seed, params);
    }

    private static LeagueSnapshot selectLeagueSnapshot(Options opts, long seed) {
        if (opts.leagueDir == null || opts.leagueDir.trim().isEmpty()) {
            return null;
        }
        Path dir = Path.of(opts.leagueDir);
        ArrayList<Path> values;
        try {
            values = leagueValueFiles(dir);
        } catch (IOException e) {
            return null;
        }
        ArrayList<LeagueSnapshot> snapshots = new ArrayList<>();
        for (Path value : values) {
            String base = stripSuffix(value.getFileName().toString(), "-value.tsv");
            Path policy = dir.resolve(base + "-policy.tsv");
            if (!Files.exists(policy)) {
                continue;
            }
            Path actionPolicy = dir.resolve(base + "-action-policy.tsv");
            snapshots.add(new LeagueSnapshot(value.toString(), policy.toString(),
                    Files.exists(actionPolicy) ? actionPolicy.toString() : opts.referenceActionPolicyPath));
        }
        if (snapshots.isEmpty()) {
            return null;
        }
        Random rnd = new Random(seed ^ 0xA0761D6478BD642FL);
        return snapshots.get(rnd.nextInt(snapshots.size()));
    }

    private static class LeagueSnapshot {
        final String valuePath;
        final String policyPath;
        final String actionPolicyPath;

        LeagueSnapshot(String valuePath, String policyPath, String actionPolicyPath) {
            this.valuePath = valuePath;
            this.policyPath = policyPath;
            this.actionPolicyPath = actionPolicyPath;
        }
    }

    private static class GameSetup {
        final String[] bots;
        final Types.TRIBE[] tribes;
        final int targetSeat;

        GameSetup(String[] bots, Types.TRIBE[] tribes, int targetSeat) {
            this.bots = bots;
            this.tribes = tribes;
            this.targetSeat = targetSeat;
        }

        Types.TRIBE targetTribe() {
            return tribes[targetSeat];
        }

        ArrayList<Agent> agents(long seed, Options opts) {
            ArrayList<Agent> agents = new ArrayList<>();
            for (int i = 0; i < bots.length; i++) {
                agents.add(newAgent(bots[i], seed + i * 17L, opts));
            }
            return agents;
        }

        String opponentLabel() {
            ArrayList<String> labels = new ArrayList<>();
            for (int i = 0; i < bots.length; i++) {
                if (i != targetSeat) {
                    labels.add(bots[i]);
                }
            }
            return String.join("+", labels);
        }

        String tribeLabel() {
            ArrayList<String> labels = new ArrayList<>();
            for (Types.TRIBE tribe : tribes) {
                labels.add(tribe.name());
            }
            return String.join("+", labels);
        }
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
        final int[] playerCountBuckets = new int[Types.TRIBE.values().length + 1];
        final int[] playerCountWins = new int[Types.TRIBE.values().length + 1];
        final int[] playerCountLosses = new int[Types.TRIBE.values().length + 1];
        final int[] playerCountIncomplete = new int[Types.TRIBE.values().length + 1];
        final int[] playerCountMarginTotal = new int[Types.TRIBE.values().length + 1];

        MatchResult(String agent, String opponent) {
            this.agent = agent;
            this.opponent = opponent;
        }

        void add(EvalOutcome outcome) {
            add(outcome, 0);
        }

        void add(EvalOutcome outcome, int playerCount) {
            scoreTotal += outcome.score;
            opponentScoreTotal += outcome.opponentScore;
            marginTotal += outcome.margin();
            if (playerCount >= 0 && playerCount < playerCountBuckets.length) {
                playerCountBuckets[playerCount]++;
                playerCountMarginTotal[playerCount] += outcome.margin();
                if (outcome.result == Types.RESULT.WIN) {
                    playerCountWins[playerCount]++;
                } else if (outcome.result == Types.RESULT.LOSS) {
                    playerCountLosses[playerCount]++;
                } else {
                    playerCountIncomplete[playerCount]++;
                }
            }
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

        double avgMargin() {
            int n = wins + losses + incomplete;
            return n == 0 ? 0.0 : (double) marginTotal / n;
        }

        double playerCountFloorWinRate() {
            double floor = Double.POSITIVE_INFINITY;
            for (int i = 0; i < playerCountBuckets.length; i++) {
                int n = playerCountBuckets[i];
                if (n > 0) {
                    floor = Math.min(floor, (double) playerCountWins[i] / n);
                }
            }
            return floor == Double.POSITIVE_INFINITY ? winRate() : floor;
        }

        double playerCountFloorMargin() {
            double floor = Double.POSITIVE_INFINITY;
            for (int i = 0; i < playerCountBuckets.length; i++) {
                int n = playerCountBuckets[i];
                if (n > 0) {
                    floor = Math.min(floor, (double) playerCountMarginTotal[i] / n);
                }
            }
            return floor == Double.POSITIVE_INFINITY ? avgMargin() : floor;
        }

        @Override
        public String toString() {
            int n = wins + losses + incomplete;
            double avgScore = n == 0 ? 0.0 : (double) scoreTotal / n;
            double avgOpponentScore = n == 0 ? 0.0 : (double) opponentScoreTotal / n;
            return String.format("eval %s vs %s: W=%d L=%d I=%d N=%d players=%s winRate=%.3f avgScore=%.1f avgOppScore=%.1f avgMargin=%.1f",
                    agent, opponent, wins, losses, incomplete, n, playerCountSummary(),
                    winRate(), avgScore, avgOpponentScore, avgMargin());
        }

        private String playerCountSummary() {
            ArrayList<String> parts = new ArrayList<>();
            for (int i = 0; i < playerCountBuckets.length; i++) {
                if (playerCountBuckets[i] > 0) {
                    parts.add(i + "p:" + playerCountBuckets[i]);
                }
            }
            return parts.isEmpty() ? "none" : String.join(",", parts);
        }
    }

    private static class CheckpointScore {
        final double score;
        final double marginFloor;
        final double marginAverage;
        final double simpleWinRate;
        final double oslaWinRate;
        final double referenceWinRate;
        final double simpleMargin;
        final double oslaMargin;
        final double referenceMargin;
        final boolean playerCountFloorGate;
        final double simpleGateWinRate;
        final double oslaGateWinRate;
        final double referenceGateWinRate;
        final double simpleGateMargin;
        final double oslaGateMargin;
        final double referenceGateMargin;
        static final double WIN_RATE_NOISE_BAND = 0.125;

        CheckpointScore(double score, double marginFloor, double marginAverage,
                        double simpleWinRate, double oslaWinRate, double referenceWinRate,
                        double simpleMargin, double oslaMargin, double referenceMargin,
                        boolean playerCountFloorGate,
                        double simpleGateWinRate, double oslaGateWinRate, double referenceGateWinRate,
                        double simpleGateMargin, double oslaGateMargin, double referenceGateMargin) {
            this.score = score;
            this.marginFloor = marginFloor;
            this.marginAverage = marginAverage;
            this.simpleWinRate = simpleWinRate;
            this.oslaWinRate = oslaWinRate;
            this.referenceWinRate = referenceWinRate;
            this.simpleMargin = simpleMargin;
            this.oslaMargin = oslaMargin;
            this.referenceMargin = referenceMargin;
            this.playerCountFloorGate = playerCountFloorGate;
            this.simpleGateWinRate = simpleGateWinRate;
            this.oslaGateWinRate = oslaGateWinRate;
            this.referenceGateWinRate = referenceGateWinRate;
            this.simpleGateMargin = simpleGateMargin;
            this.oslaGateMargin = oslaGateMargin;
            this.referenceGateMargin = referenceGateMargin;
        }

        static CheckpointScore from(MatchResult simple, MatchResult osla, MatchResult reference,
                                    boolean playerCountFloorGate) {
            double simpleGateWin = playerCountFloorGate ? simple.playerCountFloorWinRate() : simple.winRate();
            double oslaGateWin = playerCountFloorGate ? osla.playerCountFloorWinRate() : osla.winRate();
            double simpleGateMargin = playerCountFloorGate ? simple.playerCountFloorMargin() : simple.avgMargin();
            double oslaGateMargin = playerCountFloorGate ? osla.playerCountFloorMargin() : osla.avgMargin();
            double score = Math.min(simpleGateWin, oslaGateWin);
            double marginFloor = Math.min(simpleGateMargin, oslaGateMargin);
            double marginAverage = (simple.avgMargin() + osla.avgMargin()) / 2.0;
            double referenceWinRate = Double.NaN;
            double referenceMargin = Double.NaN;
            double referenceGateWin = Double.NaN;
            double referenceGateMargin = Double.NaN;
            if (reference != null) {
                referenceWinRate = reference.winRate();
                referenceMargin = reference.avgMargin();
                referenceGateWin = playerCountFloorGate ? reference.playerCountFloorWinRate() : referenceWinRate;
                referenceGateMargin = playerCountFloorGate ? reference.playerCountFloorMargin() : referenceMargin;
                score = Math.min(score, referenceGateWin);
                marginFloor = Math.min(marginFloor, referenceGateMargin);
                marginAverage = (simple.avgMargin() + osla.avgMargin() + referenceMargin) / 3.0;
            }
            return new CheckpointScore(score, marginFloor, marginAverage,
                    simple.winRate(), osla.winRate(), referenceWinRate,
                    simple.avgMargin(), osla.avgMargin(), referenceMargin,
                    playerCountFloorGate, simpleGateWin, oslaGateWin, referenceGateWin,
                    simpleGateMargin, oslaGateMargin, referenceGateMargin);
        }

        boolean betterThan(CheckpointScore other) {
            double scoreDiff = score - other.score;
            if (Math.abs(scoreDiff) > WIN_RATE_NOISE_BAND) {
                return scoreDiff > 0.0;
            }
            if (marginFloor > other.marginFloor + 1e-9) {
                return true;
            }
            if (marginFloor < other.marginFloor - 1e-9) {
                return false;
            }
            if (scoreDiff > 1e-9) {
                return true;
            }
            if (scoreDiff < -1e-9) {
                return false;
            }
            return marginAverage > other.marginAverage + 1e-9;
        }

        String format(String prefix) {
            return String.format("%s: score=%.3f marginFloor=%.1f marginAvg=%.1f "
                            + "simpleWin=%.3f oslaWin=%.3f referenceWin=%s "
                            + "simpleMargin=%.1f oslaMargin=%.1f referenceMargin=%s%s",
                    prefix, score, marginFloor, marginAverage,
                    simpleWinRate, oslaWinRate, formatMaybe(referenceWinRate),
                    simpleMargin, oslaMargin, formatMaybe(referenceMargin), gateSuffix());
        }

        private String gateSuffix() {
            if (!playerCountFloorGate) {
                return "";
            }
            return String.format(" gate=playerCountFloor simpleGate=%.3f oslaGate=%.3f referenceGate=%s "
                            + "simpleGateMargin=%.1f oslaGateMargin=%.1f referenceGateMargin=%s",
                    simpleGateWinRate, oslaGateWinRate, formatMaybe(referenceGateWinRate),
                    simpleGateMargin, oslaGateMargin, formatMaybe(referenceGateMargin));
        }

        private static String formatMaybe(double value) {
            if (Double.isNaN(value)) {
                return "NaN";
            }
            return String.format("%.3f", value);
        }
    }

    private static class Options {
        String modelPath = "models/alphazero-value.tsv";
        String policyPath = "models/alphazero-policy.tsv";
        String actionPolicyPath = "";
        String networkType = ModelFactory.LINEAR;
        String referenceNetworkType = "";
        String bestModelPath = "";
        String bestPolicyPath = "";
        String bestActionPolicyPath = "";
        String dataPath = "training/alphazero-value-data.tsv";
        String policyDataPath = "training/alphazero-policy-data.tsv";
        String actionPolicyDataPath = "";
        String trajectoryPath = "training/alphazero-sft-trajectories.jsonl";
        String mapSnapshotDir = "";
        String referenceModelPath = "models/alphazero-value.tsv";
        String referencePolicyPath = "models/alphazero-policy.tsv";
        String referenceActionPolicyPath = "";
        String leagueDir = "";
        String trainingOpponentsCsv = "SIMPLE,OSLA,AZ";
        String valueRecordingBotsCsv = "ALL";
        String policyTargetMode = "action";
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
        int referenceRefreshInterval = 0;
        int leagueMaxSnapshots = 8;
        int leagueSnapshotInterval = 0;
        double learningRate = 0.015;
        double policyLearningRate = 0.010;
        double l2 = 0.0001;
        double targetWinRate = 0.55;
        double sampleProbability = 0.35;
        double trajectorySampleProbability = 1.0;
        double heuristicBlend = 0.35;
        double disagreementHeuristicBlend = -1.0;
        double disagreementHeuristicThreshold = 0.35;
        double positionBlend = 0.20;
        double advisorOverrideMargin = 0.08;
        double opponentAdversaryWeight = 1.0;
        double policyLogitWeight = 0.06;
        double actionPolicyLogitWeight = 0.0;
        double valuePositionBlend = 0.0;
        double terminalPositionBlend = 0.0;
        double rankValueBlend = 0.0;
        double survivalValueBlend = 0.0;
        double cpuct = 1.50;
        double priorTemperature = 0.75;
        double rootNoiseFraction = 0.0;
        double rootDirichletAlpha = 0.30;
        double rootGumbelScale = 0.0;
        double rootValueSelectionWeight = 0.0;
        double improvedPolicyValueWeight = 1.0;
        double improvedPolicyTemperature = 1.0;
        double visitSamplingTemperature = 0.0;
        int visitSamplingUntilTick = 0;
        long seed = 20260416L;
        private Random levelSeedRandom = new Random(seed);
        boolean continueAfterTarget = false;
        boolean selfPlayAfterTarget = true;
        boolean selfPlayOnly = false;
        boolean randomLevelSeeds = false;
        boolean randomTribes = false;
        boolean randomPlayerCount = false;
        int minPlayers = 2;
        int maxPlayers = 2;
        boolean stratifiedTrainingPlayerCounts = false;
        boolean stratifiedEvalPlayerCounts = false;
        boolean stratifiedEvalPlayerCountsConfigured = false;
        String trainingOpponentMode = "mixed";
        boolean recordTrajectories = true;
        boolean evalReference = false;
        boolean restoreBestOnRegression = true;
        boolean baselineInitialCheckpoint = false;
        boolean gatePlayerCountFloor = false;
        boolean pureAz = false;
        boolean tacticalShortcuts = true;
        boolean advisorOverride = true;
        boolean staticPriors = true;
        boolean nextStateValuePrior = true;
        boolean learnedValueOnly = false;
        boolean prefilterByStaticScore = true;

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
                else if ("action-policy".equals(key)) opts.actionPolicyPath = value;
                else if ("network-type".equals(key)) opts.networkType = value;
                else if ("reference-network-type".equals(key)) opts.referenceNetworkType = value;
                else if ("best-model".equals(key)) opts.bestModelPath = value;
                else if ("best-policy".equals(key)) opts.bestPolicyPath = value;
                else if ("best-action-policy".equals(key)) opts.bestActionPolicyPath = value;
                else if ("data".equals(key)) opts.dataPath = value;
                else if ("policy-data".equals(key)) opts.policyDataPath = value;
                else if ("action-policy-data".equals(key)) opts.actionPolicyDataPath = value;
                else if ("trajectory-data".equals(key)) opts.trajectoryPath = value;
                else if ("map-snapshot-dir".equals(key)) opts.mapSnapshotDir = value;
                else if ("reference-model".equals(key)) opts.referenceModelPath = value;
                else if ("reference-policy".equals(key)) opts.referencePolicyPath = value;
                else if ("reference-action-policy".equals(key)) opts.referenceActionPolicyPath = value;
                else if ("league-dir".equals(key)) opts.leagueDir = value;
                else if ("training-opponents".equals(key)) opts.trainingOpponentsCsv = value;
                else if ("value-recording-bots".equals(key)) opts.valueRecordingBotsCsv = value;
                else if ("policy-targets".equals(key)) opts.policyTargetMode = value;
                else if ("prompt-id".equals(key)) opts.promptId = value;
                else if ("action-format".equals(key)) opts.actionFormat = value;
                else if ("iterations".equals(key)) opts.iterations = Integer.parseInt(value);
                else if ("games".equals(key)) opts.gamesPerIteration = Integer.parseInt(value);
                else if ("self-play-games".equals(key)) opts.selfPlayGamesAfterTarget = Integer.parseInt(value);
                else if ("eval-games".equals(key)) opts.evalGames = Integer.parseInt(value);
                else if ("epochs".equals(key)) opts.epochs = Integer.parseInt(value);
                else if ("policy-epochs".equals(key)) opts.policyEpochs = Integer.parseInt(value);
                else if ("max-examples".equals(key)) opts.maxTrainingExamples = Integer.parseInt(value);
                else if ("max-examples-per-game".equals(key)) opts.maxExamplesPerGame = Integer.parseInt(value);
                else if ("max-turns".equals(key)) opts.maxTurns = Integer.parseInt(value);
                else if ("search-calls".equals(key)) opts.searchFmCalls = Integer.parseInt(value);
                else if ("opponent-calls".equals(key)) opts.opponentFmCalls = Integer.parseInt(value);
                else if ("depth".equals(key)) opts.searchDepth = Integer.parseInt(value);
                else if ("max-actions".equals(key)) opts.maxActionsPerNode = Integer.parseInt(value);
                else if ("prefilter".equals(key)) opts.prefilterActions = Integer.parseInt(value);
                else if ("cpuct".equals(key)) opts.cpuct = Double.parseDouble(value);
                else if ("prior-temp".equals(key)) opts.priorTemperature = Double.parseDouble(value);
                else if ("max-prompt-actions".equals(key)) opts.maxPromptActions = Integer.parseInt(value);
                else if ("max-trajectories-per-game".equals(key)) opts.maxTrajectoriesPerGame = Integer.parseInt(value);
                else if ("trajectory-flush-every".equals(key)) opts.trajectoryFlushEvery = Integer.parseInt(value);
                else if ("reference-calls".equals(key)) opts.referenceSearchFmCalls = Integer.parseInt(value);
                else if ("reference-refresh-interval".equals(key)) opts.referenceRefreshInterval = Integer.parseInt(value);
                else if ("league-max-snapshots".equals(key)) opts.leagueMaxSnapshots = Integer.parseInt(value);
                else if ("league-snapshot-interval".equals(key)) opts.leagueSnapshotInterval = Integer.parseInt(value);
                else if ("lr".equals(key)) opts.learningRate = Double.parseDouble(value);
                else if ("policy-lr".equals(key)) opts.policyLearningRate = Double.parseDouble(value);
                else if ("l2".equals(key)) opts.l2 = Double.parseDouble(value);
                else if ("target".equals(key)) opts.targetWinRate = Double.parseDouble(value);
                else if ("sample".equals(key)) opts.sampleProbability = Double.parseDouble(value);
                else if ("trajectory-sample".equals(key)) opts.trajectorySampleProbability = Double.parseDouble(value);
                else if ("heuristic-blend".equals(key)) opts.heuristicBlend = Double.parseDouble(value);
                else if ("disagreement-heuristic-blend".equals(key)) opts.disagreementHeuristicBlend = Double.parseDouble(value);
                else if ("disagreement-threshold".equals(key)) opts.disagreementHeuristicThreshold = Double.parseDouble(value);
                else if ("position-blend".equals(key)) opts.positionBlend = Double.parseDouble(value);
                else if ("advisor-margin".equals(key)) opts.advisorOverrideMargin = Double.parseDouble(value);
                else if ("opponent-adversary-weight".equals(key)) opts.opponentAdversaryWeight = Double.parseDouble(value);
                else if ("policy-logit-weight".equals(key)) opts.policyLogitWeight = Double.parseDouble(value);
                else if ("action-policy-logit-weight".equals(key)) opts.actionPolicyLogitWeight = Double.parseDouble(value);
                else if ("value-position-blend".equals(key)) opts.valuePositionBlend = Double.parseDouble(value);
                else if ("terminal-position-blend".equals(key)) opts.terminalPositionBlend = Double.parseDouble(value);
                else if ("rank-value-blend".equals(key)) opts.rankValueBlend = Double.parseDouble(value);
                else if ("survival-value-blend".equals(key)) opts.survivalValueBlend = Double.parseDouble(value);
                else if ("root-noise".equals(key)) opts.rootNoiseFraction = Double.parseDouble(value);
                else if ("root-alpha".equals(key)) opts.rootDirichletAlpha = Double.parseDouble(value);
                else if ("root-gumbel-scale".equals(key)) opts.rootGumbelScale = Double.parseDouble(value);
                else if ("root-value-selection-weight".equals(key)) opts.rootValueSelectionWeight = Double.parseDouble(value);
                else if ("improved-policy-value-weight".equals(key)) opts.improvedPolicyValueWeight = Double.parseDouble(value);
                else if ("improved-policy-temp".equals(key)) opts.improvedPolicyTemperature = Double.parseDouble(value);
                else if ("visit-sampling-temp".equals(key)) opts.visitSamplingTemperature = Double.parseDouble(value);
                else if ("visit-sampling-until".equals(key)) opts.visitSamplingUntilTick = Integer.parseInt(value);
                else if ("seed".equals(key)) opts.seed = Long.parseLong(value);
                else if ("continue-after-target".equals(key)) opts.continueAfterTarget = Boolean.parseBoolean(value);
                else if ("self-play-after-target".equals(key)) opts.selfPlayAfterTarget = Boolean.parseBoolean(value);
                else if ("self-play-only".equals(key)) opts.selfPlayOnly = Boolean.parseBoolean(value);
                else if ("random-level-seeds".equals(key)) opts.randomLevelSeeds = Boolean.parseBoolean(value);
                else if ("random-tribes".equals(key)) opts.randomTribes = Boolean.parseBoolean(value);
                else if ("random-player-count".equals(key)) opts.randomPlayerCount = Boolean.parseBoolean(value);
                else if ("min-players".equals(key)) opts.minPlayers = Integer.parseInt(value);
                else if ("max-players".equals(key)) opts.maxPlayers = Integer.parseInt(value);
                else if ("stratified-training-player-counts".equals(key)) {
                    opts.stratifiedTrainingPlayerCounts = Boolean.parseBoolean(value);
                }
                else if ("stratified-eval-player-counts".equals(key)) {
                    opts.stratifiedEvalPlayerCounts = Boolean.parseBoolean(value);
                    opts.stratifiedEvalPlayerCountsConfigured = true;
                }
                else if ("training-opponent-mode".equals(key)) opts.trainingOpponentMode = value;
                else if ("record-trajectories".equals(key)) opts.recordTrajectories = Boolean.parseBoolean(value);
                else if ("eval-reference".equals(key)) opts.evalReference = Boolean.parseBoolean(value);
                else if ("restore-best-on-regression".equals(key)) opts.restoreBestOnRegression = Boolean.parseBoolean(value);
                else if ("baseline-initial-checkpoint".equals(key)) opts.baselineInitialCheckpoint = Boolean.parseBoolean(value);
                else if ("gate-player-count-floor".equals(key)) opts.gatePlayerCountFloor = Boolean.parseBoolean(value);
                else if ("pure-az".equals(key)) opts.pureAz = Boolean.parseBoolean(value);
                else if ("tactical-shortcuts".equals(key)) opts.tacticalShortcuts = Boolean.parseBoolean(value);
                else if ("advisor-override".equals(key)) opts.advisorOverride = Boolean.parseBoolean(value);
                else if ("static-priors".equals(key)) opts.staticPriors = Boolean.parseBoolean(value);
                else if ("next-state-value-prior".equals(key)) opts.nextStateValuePrior = Boolean.parseBoolean(value);
                else if ("learned-value-only".equals(key)) opts.learnedValueOnly = Boolean.parseBoolean(value);
                else if ("prefilter-by-static-score".equals(key)) opts.prefilterByStaticScore = Boolean.parseBoolean(value);
                else {
                    throw new IllegalArgumentException("Unknown option --" + key);
                }
            }
            opts.applyModeDefaults();
            opts.resolveDerivedPaths();
            opts.resetLevelSeedRandom();
            return opts;
        }

        private void applyModeDefaults() {
            if (!pureAz) {
                return;
            }
            tacticalShortcuts = false;
            advisorOverride = false;
            staticPriors = false;
            nextStateValuePrior = false;
            learnedValueOnly = true;
            prefilterByStaticScore = false;
            heuristicBlend = 0.0;
            positionBlend = 0.0;
            advisorOverrideMargin = -1.0;
            policyLogitWeight = Math.max(policyLogitWeight, 1.0);
        }

        private void resolveDerivedPaths() {
            if (bestModelPath == null || bestModelPath.trim().isEmpty()) {
                bestModelPath = derivedBestPath(modelPath);
            }
            if (bestPolicyPath == null || bestPolicyPath.trim().isEmpty()) {
                bestPolicyPath = derivedBestPath(policyPath);
            }
            if (actionPolicyPath != null && !actionPolicyPath.trim().isEmpty()) {
                if (bestActionPolicyPath == null || bestActionPolicyPath.trim().isEmpty()) {
                    bestActionPolicyPath = derivedBestPath(actionPolicyPath);
                }
                if (actionPolicyDataPath == null || actionPolicyDataPath.trim().isEmpty()) {
                    actionPolicyDataPath = derivedActionDataPath(policyDataPath);
                }
                if (referenceActionPolicyPath == null || referenceActionPolicyPath.trim().isEmpty()) {
                    referenceActionPolicyPath = actionPolicyPath;
                }
            }
            minPlayers = Math.max(2, Math.min(Types.TRIBE.values().length, minPlayers));
            maxPlayers = Math.max(minPlayers, Math.min(Types.TRIBE.values().length, maxPlayers));
            if (!stratifiedEvalPlayerCountsConfigured && gatePlayerCountFloor && randomPlayerCount) {
                stratifiedEvalPlayerCounts = true;
            }
            if (stratifiedEvalPlayerCounts) {
                evalGames = Math.max(evalGames, maxPlayers - minPlayers + 1);
            }
            if (stratifiedTrainingPlayerCounts) {
                gamesPerIteration = Math.max(gamesPerIteration, maxPlayers - minPlayers + 1);
            }
            trainingOpponentMode = trainingOpponentMode == null ? "mixed" : trainingOpponentMode.trim();
            if (trainingOpponentMode.isEmpty()) {
                trainingOpponentMode = "mixed";
            }
            if (!useMixedTrainingOpponentMode()
                    && !useScheduledTrainingOpponentForAllSeats()
                    && !useHybridTrainingOpponentMode()) {
                throw new IllegalArgumentException("Unknown --training-opponent-mode " + trainingOpponentMode
                        + " (expected mixed, scheduled, or hybrid)");
            }
            rankValueBlend = Math.max(0.0, Math.min(1.0, rankValueBlend));
            survivalValueBlend = Math.max(0.0, Math.min(1.0, survivalValueBlend));
            leagueMaxSnapshots = Math.max(0, leagueMaxSnapshots);
            leagueSnapshotInterval = Math.max(0, leagueSnapshotInterval);
        }

        private static String derivedBestPath(String path) {
            int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
            int dot = path.lastIndexOf('.');
            if (dot > slash) {
                return path.substring(0, dot) + "-best" + path.substring(dot);
            }
            return path + "-best";
        }

        private static String derivedActionDataPath(String path) {
            int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
            String dir = slash >= 0 ? path.substring(0, slash + 1) : "";
            return dir + "alphazero-action-policy-data.tsv";
        }

        synchronized long nextLevelSeed(long deterministicSeed) {
            if (!randomLevelSeeds) {
                return deterministicSeed;
            }
            long value = levelSeedRandom.nextLong();
            return value == Long.MIN_VALUE ? 0L : Math.abs(value);
        }

        private void resetLevelSeedRandom() {
            long mixed = seed ^ System.nanoTime() ^ (System.currentTimeMillis() << 21);
            levelSeedRandom = new Random(mixed);
        }

        String trainingOpponent(int gameIdx) {
            ArrayList<String> opponents = csv(trainingOpponentsCsv);
            if (opponents.isEmpty()) {
                opponents.add("AZ");
            }
            return opponents.get(Math.floorMod(gameIdx, opponents.size()));
        }

        String randomTrainingOpponent(Random rnd) {
            ArrayList<String> opponents = csv(trainingOpponentsCsv);
            if (opponents.isEmpty()) {
                opponents.add("AZ");
            }
            return opponents.get(rnd.nextInt(opponents.size()));
        }

        boolean shouldRecordValueFor(String botName) {
            ArrayList<String> bots = csv(valueRecordingBotsCsv);
            if (bots.isEmpty()) {
                return true;
            }
            for (String bot : bots) {
                if ("ALL".equalsIgnoreCase(bot)) {
                    return true;
                }
                if (bot.equalsIgnoreCase(botName)) {
                    return true;
                }
            }
            return false;
        }

        int randomPlayerCount(Random rnd) {
            if (maxPlayers <= minPlayers) {
                return minPlayers;
            }
            return minPlayers + rnd.nextInt(maxPlayers - minPlayers + 1);
        }

        int trainingPlayerCountForIndex(int gameIndex) {
            if (!stratifiedTrainingPlayerCounts) {
                return 0;
            }
            int width = Math.max(1, maxPlayers - minPlayers + 1);
            return minPlayers + Math.floorMod(gameIndex, width);
        }

        int evalPlayerCountForIndex(int evalIndex) {
            if (!stratifiedEvalPlayerCounts) {
                return 0;
            }
            int width = Math.max(1, maxPlayers - minPlayers + 1);
            return minPlayers + Math.floorMod(evalIndex, width);
        }

        boolean useMixedTrainingOpponentMode() {
            return "mixed".equalsIgnoreCase(trainingOpponentMode);
        }

        boolean useScheduledTrainingOpponentForAllSeats() {
            return "scheduled".equalsIgnoreCase(trainingOpponentMode);
        }

        boolean useHybridTrainingOpponentMode() {
            return "hybrid".equalsIgnoreCase(trainingOpponentMode);
        }

        private static ArrayList<String> csv(String value) {
            ArrayList<String> out = new ArrayList<>();
            for (String part : value.split(",")) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    out.add(trimmed);
                }
            }
            return out;
        }
    }
}
