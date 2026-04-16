package players.alphazero;

import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.ResourceGathering;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.DeclareWar;
import core.actions.tribeactions.ResearchTech;
import core.actions.tribeactions.SendStars;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Capture;
import core.actions.unitactions.Move;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.Vector2d;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class SftTrajectoryWriter implements Closeable {

    public static final String DEFAULT_SYSTEM_PROMPT =
            "You are playing Polytopia in CAPITALS mode. Choose exactly one legal candidate action index.\n\n"
                    + "Return only JSON:\n"
                    + "{\"action_index\": 0, \"reason\": \"short\"}\n\n"
                    + "No prose. No markdown. Keep any reasoning private.\n\n"
                    + "Main objective:\n"
                    + "1. Capture enemy capital.\n"
                    + "2. Keep your own capital.\n"
                    + "3. Capture villages and cities.\n"
                    + "4. Convert stars into growth, units, and useful tech.\n"
                    + "5. Move units closer to villages, cities, and capitals.\n"
                    + "6. End only when useful actions are gone.\n\n"
                    + "Critical rule: choose from the legal list literally. CAPTURE is the only immediate capture. "
                    + "MOVE only changes position. Do not invent an action outside the table.";

    private final BufferedWriter out;
    private final File outputFile;
    private final PromptFormatter formatter;
    private final String promptId;
    private final String promptSha256;
    private final String actionFormat;
    private final int flushEvery;
    private int count = 0;
    private int skipped = 0;

    public SftTrajectoryWriter(String outputPath, String promptId, String systemPrompt,
                               int maxPromptActions, String actionFormat, int flushEvery) throws IOException {
        this.outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        this.out = new BufferedWriter(new FileWriter(outputFile, true));
        this.promptId = promptId;
        this.promptSha256 = sha256(systemPrompt);
        this.actionFormat = actionFormat;
        this.flushEvery = Math.max(1, flushEvery);
        this.formatter = new PromptFormatter(maxPromptActions, systemPrompt);
    }

    public synchronized int count() {
        return count;
    }

    public synchronized int skipped() {
        return skipped;
    }

    public synchronized void writeSample(int episode, int localActionIndex, String expertBot,
                                         int seat, int playerID, JSONObject setupMetadata,
                                         GameState gs, Action expertAction, long elapsedMicros,
                                         ArrayList<Integer> allPlayerIDs) {
        if (expertAction == null || !isUsable(gs, expertAction)) {
            skipped++;
            return;
        }

        formatter.setPlayer(playerID, allPlayerIDs);
        ArrayList<Action> candidates = formatter.promptCandidates(gs, gs.getAllAvailableActions(), expertAction);
        int chosenIndex = formatter.indexOfAction(candidates, expertAction);
        if (chosenIndex < 0) {
            skipped++;
            return;
        }

        JSONObject assistant = new JSONObject()
                .put("action_index", chosenIndex)
                .put("reason", "selected");
        JSONArray messages = new JSONArray()
                .put(new JSONObject().put("role", "system").put("content", formatter.systemPrompt))
                .put(new JSONObject().put("role", "user").put("content", formatter.userPrompt(gs, candidates)))
                .put(new JSONObject().put("role", "assistant").put("content", assistant.toString()));

        JSONObject obj = new JSONObject();
        obj.put("messages", messages);
        obj.put("prompt_id", promptId);
        obj.put("prompt_sha256", promptSha256);
        obj.put("prompt_source", "players.alphazero.SftTrajectoryWriter");
        obj.put("expert_bot", expertBot);
        obj.put("target_policy", "imitate_" + expertBot);
        obj.put("episode", episode);
        obj.put("local_action_index", localActionIndex);
        obj.put("tick", gs.getTick());
        obj.put("seat", seat);
        obj.put("player_id", playerID);
        obj.put("elapsed_micros", elapsedMicros);
        obj.put("candidate_count", candidates.size());
        obj.put("action_format", actionFormat);
        obj.put("chosen_action_index", chosenIndex);
        obj.put("chosen_action_type", expertAction.getActionType().name());
        obj.put("chosen_action", expertAction.toString());
        obj.put("setup", new JSONObject(setupMetadata.toString()));

        try {
            out.write(obj.toString());
            out.newLine();
            count++;
            if (count % flushEvery == 0) {
                out.flush();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public synchronized void writeManifest(int iterations, int gamesPerIteration, double elapsedSeconds) {
        File manifest = new File(outputFile.getAbsolutePath() + ".manifest.json");
        JSONObject obj = new JSONObject();
        obj.put("created_at", Instant.now().toString());
        obj.put("dataset", outputFile.getAbsolutePath());
        obj.put("examples_this_run", count);
        obj.put("skipped_this_run", skipped);
        obj.put("iterations", iterations);
        obj.put("games_per_iteration", gamesPerIteration);
        obj.put("elapsed_seconds", elapsedSeconds);
        obj.put("prompt_id", promptId);
        obj.put("prompt_sha256", promptSha256);
        obj.put("prompt_source", "players.alphazero.SftTrajectoryWriter");
        obj.put("action_format", actionFormat);
        try {
            Files.write(manifest.toPath(), (obj.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public synchronized void close() throws IOException {
        out.flush();
        out.close();
    }

    private static boolean isUsable(GameState gs, Action action) {
        try {
            return action != null && action.isFeasible(gs);
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hashed) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static class PromptFormatter {
        private final int maxPromptActions;
        private final String systemPrompt;
        private int playerID;
        private ArrayList<Integer> allPlayerIDs = new ArrayList<>();

        PromptFormatter(int maxPromptActions, String systemPrompt) {
            this.maxPromptActions = maxPromptActions;
            this.systemPrompt = systemPrompt;
        }

        void setPlayer(int playerID, ArrayList<Integer> allPlayerIDs) {
            this.playerID = playerID;
            this.allPlayerIDs = allPlayerIDs == null ? new ArrayList<>() : new ArrayList<>(allPlayerIDs);
        }

        ArrayList<Action> promptCandidates(GameState gs, ArrayList<Action> allActions, Action expertAction) {
            ArrayList<Action> candidates = new ArrayList<>();
            for (Action action : allActions) {
                if (isUsable(gs, action)) {
                    candidates.add(action);
                }
            }
            sortPromptCandidates(gs, candidates);
            if (maxPromptActions <= 0 || candidates.size() <= maxPromptActions) {
                return candidates;
            }

            ArrayList<Action> limited = new ArrayList<>(candidates.subList(0, Math.max(1, maxPromptActions)));
            if (indexOfAction(limited, expertAction) < 0) {
                for (Action action : candidates) {
                    if (sameAction(action, expertAction)) {
                        limited.set(limited.size() - 1, action);
                        break;
                    }
                }
                sortPromptCandidates(gs, limited);
            }
            return limited;
        }

        void sortPromptCandidates(GameState gs, ArrayList<Action> candidates) {
            candidates.sort((a, b) -> {
                int byPriority = Integer.compare(promptActionPriority(b), promptActionPriority(a));
                if (byPriority != 0) {
                    return byPriority;
                }
                int byFeature = Double.compare(promptActionTieBreakScore(gs, b), promptActionTieBreakScore(gs, a));
                if (byFeature != 0) {
                    return byFeature;
                }
                return a.toString().compareTo(b.toString());
            });
        }

        String userPrompt(GameState gs, ArrayList<Action> candidates) {
            StringBuilder sb = new StringBuilder();
            sb.append("You are player ").append(playerID)
                    .append(". Active player: ").append(gs.getActiveTribeID())
                    .append(". Tick: ").append(gs.getTick())
                    .append(". Board size: ").append(gs.getBoard().getSize())
                    .append(". Legal candidates shown: ").append(candidates.size())
                    .append(".\n\n");
            sb.append(compactPlayerSummary(gs));
            sb.append("\n").append(compactBoardPrompt(gs));
            sb.append("\n").append(compactVisibleCapitals(gs));
            sb.append("\nCandidate actions are sorted by tactical category and board features, not by any opponent or teacher bot. ");
            sb.append(maxPromptActions <= 0 ? "Every legal action is shown. " : "A legal action subset is shown. ");
            sb.append(compactActionSchema());
            for (int i = 0; i < candidates.size(); i++) {
                sb.append(compactActionRow(gs, i, candidates.get(i))).append('\n');
            }
            sb.append("\nReturn only JSON with action_index and a short reason.");
            return sb.toString();
        }

        String compactActionSchema() {
            return "Choose action_index from the compact legal action table below.\n"
                    + "Each row is one legal action. Columns are separated by | and blank means not applicable.\n"
                    + "idx: the integer to return as action_index.\n"
                    + "k: short tactical category used only for ordering. C=capture, V=survival_upgrade, G=city_growth, S=unit_spawn, A=tactical_unit, M=movement_pressure, R=mobility_build, T=tech_unlock, B=economy_build, D=diplomacy_low, E=end_turn, X=bad destructive action, O=other.\n"
                    + "t: compact engine action type. M=MOVE, AT=ATTACK, C=CAPTURE, SP=SPAWN, RT=RESEARCH_TECH, B=BUILD, RG=RESOURCE_GATHERING, LU=LEVEL_UP, E=END_TURN, SS=SEND_STARS, DW=DECLARE_WAR, BR=BUILD_ROAD, DB=DISBAND, DST=DESTROY.\n"
                    + "actor: acting unit id for unit actions.\n"
                    + "from/to: x,y board coordinates before and after the action when meaningful.\n"
                    + "target: target unit id, target city id, target player, or capture target.\n"
                    + "city: city id paying for, spawning, growing, or receiving the action.\n"
                    + "item: unit, tech, resource, building, or city bonus involved.\n"
                    + "x: compact factual hints. d is movement target_delta, n is nearest target after move, g is adjacent unseen count, cap says capital capture, need says population still needed to level, ah/th/at/df describe combat hp/attack/defense.\n"
                    + "Return only the idx as action_index; do not invent an action outside this table.\n"
                    + "idx|k|t|actor|from|to|target|city|item|x\n";
        }

        String compactActionRow(GameState gs, int index, Action action) {
            String cat = promptActionCategory(action);
            String type = compactType(action.getActionType());
            String actor = "";
            String from = "";
            String to = "";
            String target = "";
            String city = "";
            String item = "";
            String facts = "";
            try {
                switch (action.getActionType()) {
                    case MOVE: {
                        Move move = (Move) action;
                        Unit unit = (Unit) gs.getActor(move.getUnitId());
                        actor = "u" + move.getUnitId();
                        if (unit != null) {
                            from = pos(unit.getPosition());
                            double before = nearestStrategicTargetDistance(gs, unit.getPosition());
                            double after = nearestStrategicTargetDistance(gs, move.getDestination());
                            facts = "d" + fmtShort(before - after) + ",n" + fmtShort(after)
                                    + ",g" + adjacentUnseenCount(gs, move.getDestination());
                        }
                        to = pos(move.getDestination());
                        break;
                    }
                    case ATTACK: {
                        Attack attack = (Attack) action;
                        Unit attacker = (Unit) gs.getActor(attack.getUnitId());
                        Unit defender = (Unit) gs.getActor(attack.getTargetId());
                        actor = "u" + attack.getUnitId();
                        target = "u" + attack.getTargetId();
                        if (attacker != null) {
                            from = pos(attacker.getPosition());
                        }
                        if (defender != null) {
                            to = pos(defender.getPosition());
                            facts = "ah" + (attacker == null ? "?" : attacker.getCurrentHP())
                                    + ",th" + defender.getCurrentHP()
                                    + ",at" + (attacker == null ? "?" : attacker.ATK)
                                    + ",df" + defender.DEF;
                        }
                        break;
                    }
                    case CAPTURE: {
                        Capture capture = (Capture) action;
                        target = capture.getCaptureType() + ":" + capture.getTargetCity();
                        city = "c" + capture.getTargetCity();
                        facts = "cap=" + isCapitalCapture(gs, capture);
                        City targetCity = (City) gs.getActor(capture.getTargetCity());
                        if (targetCity != null) {
                            to = pos(targetCity.getPosition());
                        }
                        break;
                    }
                    case SPAWN: {
                        Spawn spawn = (Spawn) action;
                        city = "c" + spawn.getCityId();
                        item = spawn.getUnitType().name();
                        City spawnCity = (City) gs.getActor(spawn.getCityId());
                        if (spawnCity != null) {
                            to = pos(spawnCity.getPosition());
                            facts = "u=" + spawnCity.getNumUnits();
                        }
                        break;
                    }
                    case RESOURCE_GATHERING: {
                        ResourceGathering gathering = (ResourceGathering) action;
                        city = "c" + gathering.getCityId();
                        item = gathering.getResource().name();
                        City resourceCity = (City) gs.getActor(gathering.getCityId());
                        facts = "need=" + (resourceCity == null ? "?" : neededToLevel(resourceCity));
                        break;
                    }
                    case LEVEL_UP: {
                        LevelUp levelUp = (LevelUp) action;
                        city = "c" + levelUp.getCityId();
                        item = levelUp.getBonus().name();
                        break;
                    }
                    case RESEARCH_TECH: {
                        ResearchTech research = (ResearchTech) action;
                        item = research.getTech().name();
                        facts = "plan=" + researchPlanLabel(research.getTech());
                        break;
                    }
                    case BUILD: {
                        Build build = (Build) action;
                        city = "c" + build.getCityId();
                        item = build.getBuildingType().name();
                        to = pos(build.getTargetPos());
                        break;
                    }
                    case BUILD_ROAD: {
                        BuildRoad road = (BuildRoad) action;
                        to = pos(road.getPosition());
                        facts = "nt=" + fmt(nearestStrategicTargetDistance(gs, road.getPosition()));
                        break;
                    }
                    case END_TURN:
                        facts = "only";
                        break;
                    case SEND_STARS:
                        SendStars sendStars = (SendStars) action;
                        target = "p" + sendStars.getTargetID();
                        item = Integer.toString(sendStars.getNumStars());
                        facts = "low";
                        break;
                    case DECLARE_WAR:
                        DeclareWar declareWar = (DeclareWar) action;
                        target = "p" + declareWar.getTargetID();
                        facts = "low";
                        break;
                    default:
                        facts = abbreviate(action.toString(), 36);
                        break;
                }
            } catch (Throwable ignored) {
                facts = "?";
            }
            return index + "|" + clean(compactCategory(cat)) + "|" + clean(type) + "|" + clean(actor)
                    + "|" + clean(from) + "|" + clean(to) + "|" + clean(target)
                    + "|" + clean(city) + "|" + clean(item) + "|" + clean(facts);
        }

        String compactCategory(String category) {
            switch (category) {
                case "capture":
                    return "C";
                case "survival_upgrade":
                    return "V";
                case "city_growth":
                    return "G";
                case "unit_spawn":
                    return "S";
                case "tactical_unit":
                    return "A";
                case "movement_pressure":
                    return "M";
                case "mobility_build":
                    return "R";
                case "tech_unlock":
                    return "T";
                case "economy_build":
                    return "B";
                case "diplomacy":
                    return "D";
                case "end_turn":
                    return "E";
                case "avoid":
                    return "X";
                default:
                    return "O";
            }
        }

        String compactType(Types.ACTION type) {
            switch (type) {
                case MOVE:
                    return "M";
                case ATTACK:
                    return "AT";
                case CAPTURE:
                    return "C";
                case SPAWN:
                    return "SP";
                case RESEARCH_TECH:
                    return "RT";
                case BUILD:
                    return "B";
                case RESOURCE_GATHERING:
                    return "RG";
                case LEVEL_UP:
                    return "LU";
                case END_TURN:
                    return "E";
                case SEND_STARS:
                    return "SS";
                case DECLARE_WAR:
                    return "DW";
                case BUILD_ROAD:
                    return "BR";
                case DISBAND:
                    return "DB";
                case DESTROY:
                    return "DST";
                case MAKE_VETERAN:
                    return "VET";
                case CONVERT:
                    return "CV";
                case HEAL_OTHERS:
                    return "HO";
                case UPGRADE_BOAT:
                    return "UB";
                case UPGRADE_SHIP:
                    return "US";
                case RECOVER:
                    return "RCV";
                case GROW_FOREST:
                    return "GF";
                case BURN_FOREST:
                    return "BF";
                case CLEAR_FOREST:
                    return "CF";
                default:
                    return type.name();
            }
        }

        int promptActionPriority(Action action) {
            switch (action.getActionType()) {
                case CAPTURE:
                    return 100;
                case MAKE_VETERAN:
                    return 95;
                case LEVEL_UP:
                    return 90;
                case RESOURCE_GATHERING:
                    return 85;
                case SPAWN:
                    return 80;
                case ATTACK:
                case CONVERT:
                case HEAL_OTHERS:
                    return 75;
                case MOVE:
                case UPGRADE_BOAT:
                case UPGRADE_SHIP:
                    return 70;
                case BUILD_ROAD:
                    return 65;
                case RESEARCH_TECH:
                    return 60;
                case BUILD:
                case BURN_FOREST:
                case CLEAR_FOREST:
                    return 55;
                case RECOVER:
                    return 45;
                case GROW_FOREST:
                    return 20;
                case DECLARE_WAR:
                case SEND_STARS:
                    return 0;
                case END_TURN:
                    return -100;
                case DESTROY:
                case DISBAND:
                    return -200;
                default:
                    return 0;
            }
        }

        String promptActionCategory(Action action) {
            switch (action.getActionType()) {
                case CAPTURE:
                    return "capture";
                case MAKE_VETERAN:
                    return "survival_upgrade";
                case LEVEL_UP:
                case RESOURCE_GATHERING:
                    return "city_growth";
                case SPAWN:
                    return "unit_spawn";
                case ATTACK:
                case CONVERT:
                case HEAL_OTHERS:
                    return "tactical_unit";
                case MOVE:
                case UPGRADE_BOAT:
                case UPGRADE_SHIP:
                    return "movement_pressure";
                case BUILD_ROAD:
                    return "mobility_build";
                case RESEARCH_TECH:
                    return "tech_unlock";
                case BUILD:
                case BURN_FOREST:
                case CLEAR_FOREST:
                    return "economy_build";
                case SEND_STARS:
                case DECLARE_WAR:
                    return "diplomacy";
                case END_TURN:
                    return "end_turn";
                case DESTROY:
                case DISBAND:
                    return "avoid";
                default:
                    return "other";
            }
        }

        double promptActionTieBreakScore(GameState gs, Action action) {
            try {
                switch (action.getActionType()) {
                    case CAPTURE:
                        return captureTieBreakScore(gs, (Capture) action);
                    case LEVEL_UP:
                        return levelUpTieBreakScore((LevelUp) action);
                    case RESOURCE_GATHERING:
                        return resourceTieBreakScore(gs, (ResourceGathering) action);
                    case SPAWN:
                        return spawnTieBreakScore((Spawn) action);
                    case ATTACK:
                        return attackTieBreakScore(gs, (Attack) action);
                    case MOVE:
                        return moveTieBreakScore(gs, (Move) action);
                    case RESEARCH_TECH:
                        return researchTieBreakScore((ResearchTech) action);
                    case BUILD:
                        return buildTieBreakScore((Build) action);
                    case BUILD_ROAD:
                        return roadTieBreakScore(gs, (BuildRoad) action);
                    case END_TURN:
                        return -10_000.0;
                    default:
                        return 0.0;
                }
            } catch (Throwable ignored) {
                return 0.0;
            }
        }

        String compactPlayerSummary(GameState gs) {
            StringBuilder sb = new StringBuilder();
            sb.append("Players compact rows:\n");
            sb.append("id|role|tribe|result|stars|score|prod|tech_count|techs|cities|city_levels|units|kills|capital_ok\n");
            for (int id : playerIDs(gs, allPlayerIDs)) {
                Tribe tribe = gs.getTribe(id);
                int cityLevels = 0;
                for (City city : gs.getCities(id)) {
                    cityLevels += city.getLevel();
                }
                sb.append(id).append('|')
                        .append(id == playerID ? "SELF" : "OPP").append('|')
                        .append(clean(tribe.getName())).append('|')
                        .append(gs.getTribeWinStatus(id)).append('|')
                        .append(tribe.getStars()).append('|')
                        .append(gs.getScore(id)).append('|')
                        .append(gs.getTribeProduction(id)).append('|')
                        .append(gs.getTribeTechTree(id).getNumResearched()).append('|')
                        .append(clean(researchedTechs(gs, id))).append('|')
                        .append(gs.getCities(id).size()).append('|')
                        .append(cityLevels).append('|')
                        .append(gs.getUnits(id).size()).append('|')
                        .append(gs.getNKills(id)).append('|')
                        .append(tribe.controlsCapital())
                        .append('\n');
            }
            return sb.toString();
        }

        String compactVisibleCapitals(GameState gs) {
            StringBuilder sb = new StringBuilder();
            sb.append("Visible capitals compact rows:\n");
            sb.append("player|city|owner|pos|level|self_controlled\n");
            for (int id : playerIDs(gs, allPlayerIDs)) {
                City capital = (City) gs.getActor(gs.getTribe(id).getCapitalID());
                if (capital != null) {
                    sb.append(id).append('|')
                            .append('c').append(capital.getActorId()).append('|')
                            .append(capital.getTribeId()).append('|')
                            .append(pos(capital.getPosition())).append('|')
                            .append(capital.getLevel()).append('|')
                            .append(capital.getTribeId() == playerID)
                            .append('\n');
                }
            }
            return sb.toString();
        }

        String compactBoardPrompt(GameState gs) {
            StringBuilder sb = new StringBuilder();
            Board board = gs.getBoard();
            int size = board.getSize();
            sb.append("Board compact grid:\n")
                    .append("Legend . plain, f forest, m mountain, s shallow, d deep, v village, K# capital, C# city, U# unit owner, resource/building suffixes may appear.\n");
            for (int y = 0; y < size; y++) {
                sb.append("y").append(y).append('|');
                for (int x = 0; x < size; x++) {
                    if (x > 0) {
                        sb.append(' ');
                    }
                    sb.append(tileToken(gs, board, x, y));
                }
                sb.append('\n');
            }
            sb.append("City compact rows:\n");
            sb.append("id|owner|cap|pos|lvl|pop/need|prod|walls|units\n");
            for (int id : playerIDs(gs, allPlayerIDs)) {
                for (City city : gs.getCities(id)) {
                    sb.append('c').append(city.getActorId()).append('|')
                            .append(id).append('|')
                            .append(city.isCapital()).append('|')
                            .append(pos(city.getPosition())).append('|')
                            .append(city.getLevel()).append('|')
                            .append(city.getPopulation()).append('/').append(city.getPopulation_need()).append('|')
                            .append(city.getProduction()).append('|')
                            .append(city.hasWalls()).append('|')
                            .append(city.getNumUnits())
                            .append('\n');
                }
            }
            sb.append("Unit compact rows:\n");
            sb.append("id|owner|type|pos|hp/max|status|kills|vet\n");
            for (int id : playerIDs(gs, allPlayerIDs)) {
                for (Unit unit : gs.getUnits(id)) {
                    sb.append('u').append(unit.getActorId()).append('|')
                            .append(id).append('|')
                            .append(unit.getType()).append('|')
                            .append(pos(unit.getPosition())).append('|')
                            .append(unit.getCurrentHP()).append('/').append(unit.getMaxHP()).append('|')
                            .append(unit.getStatus()).append('|')
                            .append(unit.getKills()).append('|')
                            .append(unit.isVeteran())
                            .append('\n');
                }
            }
            return sb.toString();
        }

        String tileToken(GameState gs, Board board, int x, int y) {
            Types.TERRAIN terrain = board.getTerrainAt(x, y);
            String token = String.valueOf(terrain.getMapChar());
            int cityId = board.getCityIdAt(x, y);
            if (terrain == Types.TERRAIN.CITY && cityId > 0) {
                City city = (City) gs.getActor(cityId);
                if (city != null) {
                    token = (city.isCapital() ? "K" : "C") + city.getTribeId();
                }
            }
            Types.RESOURCE resource = board.getResourceAt(x, y);
            if (resource != null) {
                token += resource.getMapChar();
            }
            Types.BUILDING building = board.getBuildingAt(x, y);
            if (building != null) {
                token += buildingToken(building);
            }
            int[][] units = board.getUnits();
            if (x >= 0 && y >= 0 && x < units.length && y < units[x].length && units[x][y] > 0) {
                Unit unit = (Unit) gs.getActor(units[x][y]);
                if (unit != null) {
                    token += "U" + unit.getTribeId();
                }
            }
            return token;
        }

        String buildingToken(Types.BUILDING building) {
            String name = building.name();
            return name.length() <= 3 ? name : name.substring(0, 3);
        }

        String researchedTechs(GameState gs, int id) {
            ArrayList<String> names = new ArrayList<>();
            for (Types.TECHNOLOGY tech : Types.TECHNOLOGY.values()) {
                if (gs.getTribeTechTree(id).isResearched(tech)) {
                    names.add(tech.name());
                }
            }
            return names.toString();
        }

        double captureTieBreakScore(GameState gs, Capture capture) {
            if (isCapitalCapture(gs, capture)) {
                return 1_000.0;
            }
            return capture.getCaptureType() == Types.TERRAIN.CITY ? 650.0 : 400.0;
        }

        boolean isCapitalCapture(GameState gs, Capture capture) {
            if (capture.getCaptureType() != Types.TERRAIN.CITY) {
                return false;
            }
            City target = (City) gs.getActor(capture.getTargetCity());
            return target != null
                    && target.getTribeId() != playerID
                    && gs.getTribe(target.getTribeId()).getCapitalID() == target.getActorId();
        }

        double levelUpTieBreakScore(LevelUp action) {
            switch (action.getBonus()) {
                case SUPERUNIT:
                    return 500.0;
                case WORKSHOP:
                case RESOURCES:
                    return 450.0;
                case BORDER_GROWTH:
                    return 360.0;
                case CITY_WALL:
                    return 220.0;
                default:
                    return 100.0;
            }
        }

        double resourceTieBreakScore(GameState gs, ResourceGathering action) {
            City city = (City) gs.getActor(action.getCityId());
            int needed = city == null ? 5 : neededToLevel(city);
            double score = 300.0 - 45.0 * needed;
            if (action.getResource() == Types.RESOURCE.ANIMAL) {
                score += 35.0;
            } else if (action.getResource() == Types.RESOURCE.FISH || action.getResource() == Types.RESOURCE.WHALES) {
                score += 25.0;
            } else if (action.getResource() == Types.RESOURCE.FRUIT) {
                score += 20.0;
            }
            return score;
        }

        int neededToLevel(City city) {
            return city.getLevel() + 1 - city.getProduction();
        }

        double spawnTieBreakScore(Spawn action) {
            switch (action.getUnitType()) {
                case KNIGHT:
                    return 420.0;
                case SWORDMAN:
                    return 390.0;
                case DEFENDER:
                    return 360.0;
                case RIDER:
                    return 340.0;
                case ARCHER:
                    return 320.0;
                case WARRIOR:
                    return 300.0;
                case CATAPULT:
                    return 260.0;
                default:
                    return 200.0;
            }
        }

        double attackTieBreakScore(GameState gs, Attack action) {
            Unit attacker = (Unit) gs.getActor(action.getUnitId());
            Unit defender = (Unit) gs.getActor(action.getTargetId());
            if (attacker == null || defender == null) {
                return 0.0;
            }
            double score = 200.0;
            if (attacker.getCurrentHP() >= defender.getCurrentHP()) {
                score += 60.0;
            }
            if (attacker.ATK > defender.DEF) {
                score += 70.0;
            }
            if (defender.getType() == Types.UNIT.SUPERUNIT || defender.getType() == Types.UNIT.KNIGHT
                    || defender.getType() == Types.UNIT.CATAPULT) {
                score += 45.0;
            }
            return score - 0.5 * defender.getCurrentHP();
        }

        double moveTieBreakScore(GameState gs, Move action) {
            Unit unit = (Unit) gs.getActor(action.getUnitId());
            if (unit == null) {
                return 0.0;
            }
            Vector2d from = unit.getPosition();
            Vector2d to = action.getDestination();
            double before = nearestStrategicTargetDistance(gs, from);
            double after = nearestStrategicTargetDistance(gs, to);
            double score = 120.0 * (before - after) - 8.0 * after + 8.0 * adjacentUnseenCount(gs, to);
            if (after == 0.0) {
                score += 250.0;
            }
            return score;
        }

        double nearestStrategicTargetDistance(GameState gs, Vector2d pos) {
            if (pos == null) {
                return Double.POSITIVE_INFINITY;
            }
            double best = Double.POSITIVE_INFINITY;
            Board board = gs.getBoard();
            int size = board.getSize();
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    if (board.getTerrainAt(x, y) == Types.TERRAIN.VILLAGE) {
                        best = Math.min(best, Vector2d.chebychevDistance(pos, new Vector2d(x, y)));
                    }
                }
            }
            for (int id : playerIDs(gs, allPlayerIDs)) {
                if (id == playerID) {
                    continue;
                }
                for (City city : gs.getCities(id)) {
                    double distance = Vector2d.chebychevDistance(pos, city.getPosition());
                    if (city.isCapital()) {
                        distance -= 1.5;
                    }
                    best = Math.min(best, distance);
                }
            }
            City ownCapital = (City) gs.getActor(gs.getTribe(playerID).getCapitalID());
            if (ownCapital != null && enemyNear(gs, ownCapital.getPosition(), 3)) {
                best = Math.min(best, Vector2d.chebychevDistance(pos, ownCapital.getPosition()) - 1.0);
            }
            return best;
        }

        boolean enemyNear(GameState gs, Vector2d pos, int radius) {
            for (int id : playerIDs(gs, allPlayerIDs)) {
                if (id == playerID) {
                    continue;
                }
                for (Unit unit : gs.getUnits(id)) {
                    if (Vector2d.chebychevDistance(pos, unit.getPosition()) <= radius) {
                        return true;
                    }
                }
            }
            return false;
        }

        int adjacentUnseenCount(GameState gs, Vector2d pos) {
            if (pos == null) {
                return 0;
            }
            boolean[][] obsGrid = gs.getTribe(playerID).getObsGrid();
            if (obsGrid == null) {
                return 0;
            }
            int count = 0;
            for (Vector2d neigh : pos.neighborhood(1, 0, gs.getBoard().getSize())) {
                if (neigh.x >= 0 && neigh.y >= 0 && neigh.x < obsGrid.length
                        && neigh.y < obsGrid[neigh.x].length && !obsGrid[neigh.x][neigh.y]) {
                    count++;
                }
            }
            return count;
        }

        double researchTieBreakScore(ResearchTech action) {
            switch (action.getTech()) {
                case ORGANIZATION:
                    return 360.0;
                case FISHING:
                    return 350.0;
                case RIDING:
                    return 330.0;
                case CLIMBING:
                    return 320.0;
                case FORESTRY:
                case FARMING:
                case MINING:
                    return 300.0;
                case SAILING:
                case ROADS:
                case ARCHERY:
                case SHIELDS:
                    return 260.0;
                default:
                    return 180.0;
            }
        }

        String researchPlanLabel(Types.TECHNOLOGY tech) {
            switch (tech) {
                case ORGANIZATION:
                    return "fruit_and_early_economy";
                case FISHING:
                    return "fish_water_access_chain";
                case CLIMBING:
                    return "mountain_access_and_mines";
                case RIDING:
                    return "fast_expansion_pressure";
                case ROADS:
                    return "capital_routes_after_expansion";
                case SAILING:
                    return "cross_water_after_ports";
                default:
                    return "general_unlock";
            }
        }

        double buildTieBreakScore(Build action) {
            switch (action.getBuildingType()) {
                case FARM:
                case MINE:
                    return 300.0;
                case PORT:
                    return 270.0;
                case LUMBER_HUT:
                    return 250.0;
                case FORGE:
                case SAWMILL:
                case WINDMILL:
                    return 240.0;
                default:
                    return 120.0;
            }
        }

        double roadTieBreakScore(GameState gs, BuildRoad action) {
            double distance = nearestStrategicTargetDistance(gs, action.getPosition());
            return Double.isInfinite(distance) ? 0.0 : 200.0 - 15.0 * distance;
        }

        int indexOfAction(ArrayList<Action> candidates, Action target) {
            if (target == null) {
                return -1;
            }
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.get(i) == target) {
                    return i;
                }
            }
            for (int i = 0; i < candidates.size(); i++) {
                if (sameAction(candidates.get(i), target)) {
                    return i;
                }
            }
            return -1;
        }

        boolean sameAction(Action left, Action right) {
            if (left == right) {
                return true;
            }
            return left != null && right != null && left.toString().equals(right.toString());
        }

        List<Integer> playerIDs(GameState gs, ArrayList<Integer> ids) {
            if (ids != null && !ids.isEmpty()) {
                return ids;
            }
            ArrayList<Integer> out = new ArrayList<>();
            for (int i = 0; i < gs.getTribes().length; i++) {
                out.add(i);
            }
            return out;
        }

        String pos(Vector2d pos) {
            return pos == null ? "" : pos.x + "," + pos.y;
        }

        String clean(String value) {
            if (value == null) {
                return "";
            }
            return value.replace('|', '/')
                    .replace('\n', ' ')
                    .replace('\r', ' ')
                    .trim();
        }

        String abbreviate(String value, int maxLength) {
            String cleaned = clean(value);
            return cleaned.length() <= maxLength ? cleaned : cleaned.substring(0, maxLength);
        }

        String fmt(double value) {
            if (Double.isInfinite(value) || Double.isNaN(value)) {
                return "none";
            }
            return String.format(java.util.Locale.ROOT, "%.1f", value);
        }

        String fmtShort(double value) {
            String formatted = fmt(value);
            return formatted.endsWith(".0") ? formatted.substring(0, formatted.length() - 2) : formatted;
        }
    }
}
