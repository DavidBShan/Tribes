package players.alphazero;

import core.Types;
import core.actors.City;
import core.game.Board;
import core.game.Game;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

public final class MapSnapshotWriter {

    private MapSnapshotWriter() {
    }

    public static String fileName(String split, int episode, int gameIndex, long levelSeed, long gameSeed) {
        return String.format("%s-episode%05d-game%03d-level%d-game%d.csv",
                safe(split), episode, gameIndex, levelSeed, gameSeed);
    }

    public static synchronized void write(String dir, String split, int episode, int gameIndex,
                                          long levelSeed, long gameSeed, String opponent, Game game,
                                          JSONObject setupMetadata) {
        if (dir == null || dir.trim().isEmpty() || game == null) {
            return;
        }

        try {
            File root = new File(dir);
            root.mkdirs();

            File csv = new File(root, fileName(split, episode, gameIndex, levelSeed, gameSeed));
            JSONArray rows = writeCsv(csv, game.getBoard());

            JSONObject manifest = new JSONObject();
            manifest.put("split", split);
            manifest.put("episode", episode);
            manifest.put("game_index", gameIndex);
            manifest.put("level_seed", levelSeed);
            manifest.put("game_seed", gameSeed);
            manifest.put("opponent", opponent);
            manifest.put("csv", csv.getName());
            if (setupMetadata != null) {
                Iterator<String> keys = setupMetadata.keys();
                while (keys.hasNext()) {
                    String key = keys.next();
                    if (!manifest.has(key)) {
                        manifest.put(key, setupMetadata.get(key));
                    }
                }
                manifest.put("setup", setupMetadata);
            }
            manifest.put("rows", rows);

            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(root, "manifest.jsonl"), true));
            writer.write(manifest.toString());
            writer.newLine();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException("Could not write map snapshot to " + dir, e);
        }
    }

    private static JSONArray writeCsv(File csv, Board board) throws IOException {
        JSONArray rows = new JSONArray();
        BufferedWriter writer = new BufferedWriter(new FileWriter(csv));
        int size = board.getSize();
        for (int x = 0; x < size; x++) {
            StringBuilder row = new StringBuilder();
            for (int y = 0; y < size; y++) {
                if (y > 0) {
                    row.append(',');
                }
                row.append(levelToken(board, x, y));
            }
            String line = row.toString();
            rows.put(line);
            writer.write(line);
            writer.newLine();
        }
        writer.close();
        return rows;
    }

    private static String levelToken(Board board, int x, int y) {
        if (board.getTerrainAt(x, y) == Types.TERRAIN.CITY) {
            int cityId = board.getCityIdAt(x, y);
            if (cityId > 0 && board.getActor(cityId) instanceof City) {
                City city = (City) board.getActor(cityId);
                int tribeKey = board.getTribe(city.getTribeId()).getType().getKey();
                return String.valueOf(Types.TERRAIN.CITY.getMapChar()) + ':' + tribeKey;
            }
        }

        String token = String.valueOf(board.getTerrainAt(x, y).getMapChar()) + ':';
        Types.RESOURCE resource = board.getResourceAt(x, y);
        return token + (resource == null ? " " : String.valueOf(resource.getMapChar()));
    }

    private static String safe(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "game";
        }
        return value.replaceAll("[^A-Za-z0-9_.-]", "_");
    }
}
