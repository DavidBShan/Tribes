package core.levelgen;

import core.Types;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Research-derived Polytopia-style generator for original Tribes CSV levels.
 *
 * It intentionally recreates map-generation patterns rather than any fixed
 * official map layout: size families, wetness presets, settlement spacing,
 * tribe-biased terrain, settlement-banded resources, and ruin counts.
 */
public class PolytopiaStyleLevelGenerator {

    public enum MapType {
        DRYLANDS("drylands", false, false, false, 0.0),
        LAKES("lakes", true, true, false, 0.3),
        CONTINENTS("continents", false, false, true, 0.0),
        PANGEA("pangea", false, false, true, 0.0),
        ARCHIPELAGO("archipelago", true, true, false, 0.3),
        WATER_WORLD("water_world", true, false, true, 0.1);

        private final String slug;
        private final boolean usesQuadrants;
        private final boolean usesSuburbs;
        private final boolean usesTinyIslandVillages;
        private final double preTerrainVillageCoefficient;

        MapType(String slug, boolean usesQuadrants, boolean usesSuburbs,
                boolean usesTinyIslandVillages, double preTerrainVillageCoefficient) {
            this.slug = slug;
            this.usesQuadrants = usesQuadrants;
            this.usesSuburbs = usesSuburbs;
            this.usesTinyIslandVillages = usesTinyIslandVillages;
            this.preTerrainVillageCoefficient = preTerrainVillageCoefficient;
        }

        public String slug() {
            return slug;
        }
    }

    public static class Config {
        public final int size;
        public final int players;
        public final MapType mapType;
        public final Types.TRIBE[] tribes;

        public Config(int size, int players, MapType mapType, Types.TRIBE[] tribes) {
            if (size < 11) {
                throw new IllegalArgumentException("Map size must be at least 11.");
            }
            if (players < 1 || players > 12) {
                throw new IllegalArgumentException("Tribes supports 1 to 12 playable tribes.");
            }
            if (tribes != null && tribes.length != players) {
                throw new IllegalArgumentException("Tribe array length must match player count.");
            }
            this.size = size;
            this.players = players;
            this.mapType = mapType;
            this.tribes = tribes == null ? randomTribes(players) : tribes.clone();
        }
    }

    public static class GeneratedLevel {
        private final String[] lines;
        private final int size;
        private final int players;
        private final MapType mapType;
        private final Types.TRIBE[] tribes;

        private GeneratedLevel(String[] lines, int size, int players, MapType mapType, Types.TRIBE[] tribes) {
            this.lines = lines;
            this.size = size;
            this.players = players;
            this.mapType = mapType;
            this.tribes = tribes.clone();
        }

        public String[] lines() {
            return lines.clone();
        }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append(mapType.slug()).append('\t').append(size).append('x').append(size)
                    .append('\t').append(players).append("p").append('\t');
            for (int i = 0; i < tribes.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(tribes[i].getName());
            }
            return sb.toString();
        }

        public void writeCsv(File file) throws IOException {
            File parent = file.getParentFile();
            if (parent != null && !parent.exists() && !parent.mkdirs()) {
                throw new IOException("Could not create directory: " + parent);
            }
            try (FileWriter writer = new FileWriter(file)) {
                for (int i = 0; i < lines.length; i++) {
                    if (i > 0) {
                        writer.write('\n');
                    }
                    writer.write(lines[i]);
                }
            }
        }
    }

    private static final char PLAIN = '.';
    private static final char SHALLOW_WATER = 's';
    private static final char DEEP_WATER = 'd';
    private static final char MOUNTAIN = 'm';
    private static final char VILLAGE = 'v';
    private static final char CITY = 'c';
    private static final char FOREST = 'f';
    private static final char NONE = 0;

    private static final char FISH = 'h';
    private static final char FRUIT = 'f';
    private static final char ANIMAL = 'a';
    private static final char WHALE = 'w';
    private static final char ORE = 'o';
    private static final char CROPS = 'c';
    private static final char RUINS = 'r';

    private static final double FIELD_RATE = 0.48;
    private static final double FOREST_RATE = 0.38;
    private static final double MOUNTAIN_RATE = 0.14;
    private static final double INNER_FIELD_RESOURCE_RATE = 0.18;
    private static final double OUTER_FIELD_RESOURCE_RATE = 0.06;
    private static final double INNER_ANIMAL_RATE = 0.19;
    private static final double OUTER_ANIMAL_RATE = 0.06;
    private static final double INNER_ORE_RATE = 0.11;
    private static final double OUTER_ORE_RATE = 0.03;
    private static final double BORDER_EXPANSION = 1.0 / 3.0;

    private static final EnumMap<Types.TRIBE, TribeRates> TRIBE_RATES = new EnumMap<>(Types.TRIBE.class);

    static {
        for (Types.TRIBE tribe : Types.TRIBE.values()) {
            TRIBE_RATES.put(tribe, new TribeRates());
        }
        rates(Types.TRIBE.XIN_XI).mountain = 1.5;
        rates(Types.TRIBE.XIN_XI).ore = 1.5;
        rates(Types.TRIBE.IMPERIUS).animal = 0.5;
        rates(Types.TRIBE.IMPERIUS).fruit = 2.0;
        rates(Types.TRIBE.BARDUR).forest = 0.8;
        rates(Types.TRIBE.BARDUR).crop = 0.0;
        rates(Types.TRIBE.OUMAJI).forest = 0.2;
        rates(Types.TRIBE.OUMAJI).animal = 0.2;
        rates(Types.TRIBE.OUMAJI).mountain = 0.5;
        rates(Types.TRIBE.KICKOO).mountain = 0.5;
        rates(Types.TRIBE.KICKOO).fish = 1.5;
        rates(Types.TRIBE.HOODRICK).mountain = 0.5;
        rates(Types.TRIBE.HOODRICK).forest = 1.5;
        rates(Types.TRIBE.VENGIR).ore = 2.0;
        rates(Types.TRIBE.VENGIR).animal = 0.1;
        rates(Types.TRIBE.VENGIR).fruit = 0.1;
        rates(Types.TRIBE.VENGIR).fish = 0.1;
        rates(Types.TRIBE.ZEBASI).mountain = 0.5;
        rates(Types.TRIBE.ZEBASI).forest = 0.5;
        rates(Types.TRIBE.ZEBASI).fruit = 0.5;
        rates(Types.TRIBE.AI_MO).mountain = 1.5;
        rates(Types.TRIBE.AI_MO).crop = 0.1;
        rates(Types.TRIBE.QUETZALI).fruit = 2.0;
        rates(Types.TRIBE.QUETZALI).crop = 0.1;
        rates(Types.TRIBE.YADAKK).mountain = 0.5;
        rates(Types.TRIBE.YADAKK).forest = 0.5;
        rates(Types.TRIBE.YADAKK).fruit = 1.5;
    }

    private final Config config;
    private final int size;
    private final Tile[] tiles;
    private final LinkedHashMap<Integer, Types.TRIBE> capitals = new LinkedHashMap<>();
    private final Set<Integer> villages = new HashSet<>();

    public PolytopiaStyleLevelGenerator(Config config) {
        this.config = config;
        this.size = config.size;
        this.tiles = new Tile[size * size];
        for (int i = 0; i < tiles.length; i++) {
            tiles[i] = new Tile();
        }
    }

    public static GeneratedLevel generate(Config config) {
        PolytopiaStyleLevelGenerator generator = new PolytopiaStyleLevelGenerator(config);
        return generator.generate();
    }

    public static Types.TRIBE[] randomTribes(int players) {
        ArrayList<Types.TRIBE> all = new ArrayList<>();
        Collections.addAll(all, Types.TRIBE.values());
        shuffle(all);
        Types.TRIBE[] selected = new Types.TRIBE[players];
        for (int i = 0; i < players; i++) {
            selected[i] = all.get(i);
        }
        return selected;
    }

    private GeneratedLevel generate() {
        generateLandAndWater();
        convertCoastalWater();

        if (config.mapType.usesQuadrants) {
            placeQuadrantCapitals();
            if (config.mapType.usesSuburbs) {
                placeSuburbs();
            }
            placePreTerrainVillages();
        } else {
            placePangeaOrContinentVillages();
            convertVillagesToCapitals();
        }

        assignTerritories();
        addTerrainFeatures();
        convertCoastalWater();
        placePostTerrainVillages();
        if (config.mapType.usesTinyIslandVillages) {
            placeTinyIslandVillages();
        }
        assignTerritories();
        addResources();
        addRuins();
        ensureStartingResources();
        convertCoastalWater();

        return new GeneratedLevel(toCsvLines(), size, config.players, config.mapType, config.tribes);
    }

    private void generateLandAndWater() {
        switch (config.mapType) {
            case DRYLANDS:
                fillLand();
                scatterWater(0.045, 1);
                break;
            case LAKES:
                fillLand();
                scatterWater(0.24, 2);
                forceLandBorder();
                break;
            case CONTINENTS:
                randomLand(0.52);
                smoothLand(4, 5);
                ensureMinimumLand(0.38);
                break;
            case PANGEA:
                radialPangea();
                smoothLand(3, 5);
                forceCenterLand();
                ensureMinimumLand(0.45);
                break;
            case ARCHIPELAGO:
                randomLand(0.34);
                smoothLand(2, 4);
                addIslandChains();
                ensureMinimumLand(0.25);
                break;
            case WATER_WORLD:
                fillWater();
                addSmallIslands(config.players * 4 + Math.max(4, size / 2), 1);
                ensureMinimumLand(0.10);
                break;
            default:
                throw new IllegalStateException("Unhandled map type: " + config.mapType);
        }
    }

    private void fillLand() {
        for (Tile tile : tiles) {
            tile.terrain = PLAIN;
        }
    }

    private void fillWater() {
        for (Tile tile : tiles) {
            tile.terrain = DEEP_WATER;
        }
    }

    private void randomLand(double probability) {
        for (Tile tile : tiles) {
            tile.terrain = Math.random() < probability ? PLAIN : DEEP_WATER;
        }
    }

    private void radialPangea() {
        double center = (size - 1) / 2.0;
        double maxDistance = center;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double distance = Math.max(Math.abs(x - center), Math.abs(y - center)) / maxDistance;
                double probability = 0.92 - distance * 0.85 + (Math.random() - 0.5) * 0.18;
                tile(index(x, y)).terrain = Math.random() < probability ? PLAIN : DEEP_WATER;
            }
        }
    }

    private void forceCenterLand() {
        int center = index(size / 2, size / 2);
        for (int cell : disk(center, Math.max(2, size / 7))) {
            tile(cell).terrain = PLAIN;
        }
    }

    private void scatterWater(double probability, int expansionPasses) {
        ArrayList<Integer> seeds = new ArrayList<>();
        for (int i = 0; i < tiles.length; i++) {
            if (Math.random() < probability) {
                seeds.add(i);
                tile(i).terrain = DEEP_WATER;
            }
        }
        for (int pass = 0; pass < expansionPasses; pass++) {
            ArrayList<Integer> expansion = new ArrayList<>();
            for (int seed : seeds) {
                for (int neighbor : crossNeighbors(seed)) {
                    if (Math.random() < 0.35) {
                        expansion.add(neighbor);
                    }
                }
            }
            for (int cell : expansion) {
                tile(cell).terrain = DEEP_WATER;
            }
            seeds = expansion;
        }
    }

    private void smoothLand(int passes, int landThresholdWithSelf) {
        for (int pass = 0; pass < passes; pass++) {
            char[] next = new char[tiles.length];
            for (int i = 0; i < tiles.length; i++) {
                int land = 0;
                for (int cell : disk(i, 1)) {
                    if (isLandTerrain(tile(cell).terrain)) {
                        land++;
                    }
                }
                next[i] = land >= landThresholdWithSelf ? PLAIN : DEEP_WATER;
            }
            for (int i = 0; i < tiles.length; i++) {
                tile(i).terrain = next[i];
            }
        }
    }

    private void addIslandChains() {
        int chains = Math.max(2, size / 5);
        for (int i = 0; i < chains; i++) {
            int cell = randomCell();
            int length = randomInt(Math.max(3, size / 4), Math.max(5, size / 2));
            for (int step = 0; step < length; step++) {
                tile(cell).terrain = PLAIN;
                if (Math.random() < 0.35) {
                    for (int neighbor : crossNeighbors(cell)) {
                        if (Math.random() < 0.55) {
                            tile(neighbor).terrain = PLAIN;
                        }
                    }
                }
                List<Integer> next = crossNeighbors(cell);
                cell = next.get(randomInt(0, next.size()));
            }
        }
    }

    private void addSmallIslands(int count, int radius) {
        for (int i = 0; i < count; i++) {
            int cell = randomCellAwayFromEdge(1);
            forceLandPatch(cell, radius);
            if (Math.random() < 0.4) {
                List<Integer> neighbors = crossNeighbors(cell);
                forceLandPatch(neighbors.get(randomInt(0, neighbors.size())), 0);
            }
        }
    }

    private void ensureMinimumLand(double targetLandRatio) {
        int target = (int) Math.round(tiles.length * targetLandRatio);
        while (countLand() < target) {
            forceLandPatch(randomCellAwayFromEdge(1), 1);
        }
    }

    private void forceLandBorder() {
        for (int i = 0; i < size; i++) {
            tile(index(i, 0)).terrain = PLAIN;
            tile(index(i, size - 1)).terrain = PLAIN;
            tile(index(0, i)).terrain = PLAIN;
            tile(index(size - 1, i)).terrain = PLAIN;
        }
    }

    private void convertCoastalWater() {
        ArrayList<Integer> toShallow = new ArrayList<>();
        for (int i = 0; i < tiles.length; i++) {
            if (tile(i).terrain == DEEP_WATER) {
                for (int neighbor : crossNeighbors(i)) {
                    if (isLandTerrain(tile(neighbor).terrain)) {
                        toShallow.add(i);
                        break;
                    }
                }
            }
        }
        for (int cell : toShallow) {
            tile(cell).terrain = SHALLOW_WATER;
        }
    }

    private void placeQuadrantCapitals() {
        int domainsPerSide = domainsPerSide(config.players);
        ArrayList<Integer> domains = new ArrayList<>();
        for (int i = 0; i < domainsPerSide * domainsPerSide; i++) {
            domains.add(i);
        }
        shuffle(domains);

        for (int i = 0; i < config.players; i++) {
            int domain = domains.get(i);
            int cell = chooseCapitalCellInDomain(domain, domainsPerSide);
            forceLandPatch(cell, config.mapType == MapType.WATER_WORLD ? 2 : 1);
            addCapital(cell, config.tribes[i]);
        }
    }

    private int chooseCapitalCellInDomain(int domain, int domainsPerSide) {
        int domainX = domain % domainsPerSide;
        int domainY = domain / domainsPerSide;
        int startX = (int) Math.floor(domainX * size / (double) domainsPerSide);
        int endX = (int) Math.floor((domainX + 1) * size / (double) domainsPerSide) - 1;
        int startY = (int) Math.floor(domainY * size / (double) domainsPerSide);
        int endY = (int) Math.floor((domainY + 1) * size / (double) domainsPerSide) - 1;

        ArrayList<Integer> candidates = new ArrayList<>();
        for (int y = Math.max(2, startY); y <= Math.min(size - 3, endY); y++) {
            for (int x = Math.max(2, startX); x <= Math.min(size - 3, endX); x++) {
                int cell = index(x, y);
                if (isLandTerrain(tile(cell).terrain) && farFromSettlements(cell, 4)) {
                    candidates.add(cell);
                }
            }
        }
        if (candidates.isEmpty()) {
            for (int y = Math.max(2, startY); y <= Math.min(size - 3, endY); y++) {
                for (int x = Math.max(2, startX); x <= Math.min(size - 3, endX); x++) {
                    candidates.add(index(x, y));
                }
            }
        }
        if (candidates.isEmpty()) {
            return randomCellAwayFromEdge(2);
        }
        return candidates.get(randomInt(0, candidates.size()));
    }

    private void placeSuburbs() {
        ArrayList<Integer> capitalCells = new ArrayList<>(capitals.keySet());
        for (int capital : capitalCells) {
            int suburbCount = Math.random() < 0.85 ? 2 : 1;
            for (int i = 0; i < suburbCount; i++) {
                ArrayList<Integer> candidates = new ArrayList<>();
                for (int radius = 2; radius <= Math.max(3, size / 4); radius++) {
                    for (int cell : circle(capital, radius)) {
                        if (validVillageCandidate(cell, 1, true)) {
                            candidates.add(cell);
                        }
                    }
                }
                if (!candidates.isEmpty()) {
                    int cell = candidates.get(randomInt(0, candidates.size()));
                    forceLandPatch(cell, 0);
                    addVillage(cell);
                }
            }
        }
    }

    private void placePreTerrainVillages() {
        double coeff = config.mapType.preTerrainVillageCoefficient;
        if (coeff <= 0.0) {
            return;
        }
        int base = (size / 3) * (size / 3);
        int target = Math.max(0, (int) Math.round((base - settlementCount()) * coeff));
        for (int i = 0; i < target; i++) {
            int cell = chooseVillageCandidate(1, true);
            if (cell < 0 && config.mapType == MapType.WATER_WORLD) {
                cell = randomCellAwayFromEdge(1);
                forceLandPatch(cell, 1);
            }
            if (cell >= 0) {
                addVillage(cell);
            }
        }
    }

    private void placePangeaOrContinentVillages() {
        if (config.mapType == MapType.CONTINENTS) {
            placeOneVillagePerLandmass();
        }
        int guard = tiles.length;
        while (guard-- > 0) {
            int cell = chooseVillageCandidate(1, true);
            if (cell < 0) {
                break;
            }
            addVillage(cell);
        }
        while (villages.size() < config.players) {
            int cell = randomLandCellAwayFromEdge(1);
            addVillage(cell);
        }
    }

    private void placeOneVillagePerLandmass() {
        for (List<Integer> component : landComponents()) {
            if (component.size() < 6) {
                continue;
            }
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int cell : component) {
                if (validVillageCandidate(cell, 1, true)) {
                    candidates.add(cell);
                }
            }
            if (!candidates.isEmpty()) {
                addVillage(candidates.get(randomInt(0, candidates.size())));
            }
        }
    }

    private void convertVillagesToCapitals() {
        ArrayList<Integer> available = new ArrayList<>(villages);
        if (available.isEmpty()) {
            available.add(randomLandCellAwayFromEdge(2));
        }
        for (int i = 0; i < config.players; i++) {
            if (available.isEmpty()) {
                int forced = randomLandCellAwayFromEdge(2);
                addVillage(forced);
                available.add(forced);
            }
            int selected = chooseCapitalFromVillages(available);
            villages.remove(selected);
            available.remove((Integer) selected);
            addCapital(selected, config.tribes[i]);
        }
    }

    private int chooseCapitalFromVillages(ArrayList<Integer> candidates) {
        int best = candidates.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int cell : candidates) {
            double minDistance = capitals.isEmpty() ? size : distanceToNearestCapital(cell);
            double coastBonus = adjacentToWater(cell) ? 2.0 : 0.0;
            double continentBonus = differentLandmassBonus(cell);
            double score = minDistance + coastBonus + continentBonus + Math.random();
            if (score > bestScore) {
                bestScore = score;
                best = cell;
            }
        }
        return best;
    }

    private double differentLandmassBonus(int cell) {
        if (config.mapType != MapType.CONTINENTS || capitals.isEmpty()) {
            return 0.0;
        }
        for (int capital : capitals.keySet()) {
            if (sameLandmass(cell, capital)) {
                return 0.0;
            }
        }
        return 3.0;
    }

    private void addTerrainFeatures() {
        for (int i = 0; i < tiles.length; i++) {
            Tile tile = tile(i);
            if (tile.terrain != PLAIN || isSettlement(i)) {
                continue;
            }
            TribeRates rates = rates(tile.owner);
            double mountain = MOUNTAIN_RATE * rates.mountain;
            double forest = FOREST_RATE * rates.forest;
            if (config.mapType == MapType.DRYLANDS) {
                forest *= 0.9;
            } else if (config.mapType == MapType.ARCHIPELAGO || config.mapType == MapType.WATER_WORLD) {
                forest *= 0.85;
            }
            double total = mountain + forest;
            if (total > 0.92) {
                mountain *= 0.92 / total;
                forest *= 0.92 / total;
            }
            double roll = Math.random();
            if (roll < mountain) {
                tile.terrain = MOUNTAIN;
            } else if (roll < mountain + forest) {
                tile.terrain = FOREST;
            } else {
                tile.terrain = PLAIN;
            }
        }
    }

    private void placePostTerrainVillages() {
        int guard = tiles.length;
        while (guard-- > 0) {
            int cell = chooseVillageCandidate(2, false);
            if (cell < 0) {
                break;
            }
            addVillage(cell);
        }
    }

    private void placeTinyIslandVillages() {
        int target = tinyIslandVillageCount(size);
        for (int i = 0; i < target; i++) {
            int cell = chooseIslandVillageCell();
            if (cell < 0) {
                return;
            }
            tile(cell).terrain = PLAIN;
            addVillage(cell);
            for (int neighbor : crossNeighbors(cell)) {
                if (!isSettlement(neighbor)) {
                    tile(neighbor).terrain = SHALLOW_WATER;
                    tile(neighbor).resource = NONE;
                }
            }
        }
    }

    private int chooseIslandVillageCell() {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < tiles.length; i++) {
            if (edgeDistance(i) < 2 || tile(i).terrain != DEEP_WATER || !farFromSettlements(i, 4)) {
                continue;
            }
            boolean farFromLand = true;
            for (int cell : disk(i, 1)) {
                if (cell != i && isLandTerrain(tile(cell).terrain)) {
                    farFromLand = false;
                    break;
                }
            }
            if (farFromLand) {
                candidates.add(i);
            }
        }
        if (candidates.isEmpty()) {
            return -1;
        }
        return candidates.get(randomInt(0, candidates.size()));
    }

    private void addResources() {
        for (int i = 0; i < tiles.length; i++) {
            if (isSettlement(i)) {
                tile(i).resource = NONE;
                continue;
            }
            int settlementDistance = distanceToNearestSettlement(i);
            if (settlementDistance > 2) {
                continue;
            }

            boolean inner = settlementDistance <= 1;
            double borderFactor = inner ? 1.0 : BORDER_EXPANSION;
            TribeRates rates = rates(tile(i).owner);
            switch (tile(i).terrain) {
                case PLAIN:
                    addFieldResource(i, inner, rates);
                    break;
                case FOREST:
                    if (Math.random() < (inner ? INNER_ANIMAL_RATE : OUTER_ANIMAL_RATE) * rates.animal) {
                        tile(i).resource = ANIMAL;
                    }
                    break;
                case MOUNTAIN:
                    if (Math.random() < (inner ? INNER_ORE_RATE : OUTER_ORE_RATE) * rates.ore) {
                        tile(i).resource = ORE;
                    }
                    break;
                case SHALLOW_WATER:
                    if (Math.random() < 0.50 * rates.fish * borderFactor) {
                        tile(i).resource = FISH;
                    }
                    break;
                case DEEP_WATER:
                    if (Math.random() < 0.18 * borderFactor) {
                        tile(i).resource = WHALE;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private void addFieldResource(int cell, boolean inner, TribeRates rates) {
        double fruit = (inner ? INNER_FIELD_RESOURCE_RATE : OUTER_FIELD_RESOURCE_RATE) * rates.fruit;
        double crop = (inner ? INNER_FIELD_RESOURCE_RATE : OUTER_FIELD_RESOURCE_RATE) * rates.crop;
        double total = fruit + crop;
        if (total > 0.75) {
            fruit *= 0.75 / total;
            crop *= 0.75 / total;
        }
        double roll = Math.random();
        if (roll < fruit) {
            tile(cell).resource = FRUIT;
        } else if (roll < fruit + crop) {
            tile(cell).resource = CROPS;
        }
    }

    private void addRuins() {
        int target = ruinCount(size);
        int waterLimit = config.mapType == MapType.LAKES ? Math.max(1, target / 3) : target;
        int waterRuins = 0;
        for (int placed = 0; placed < target; placed++) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < tiles.length; i++) {
                if (isSettlement(i) || tile(i).resource != NONE || adjacentToResource(i, RUINS)) {
                    continue;
                }
                if (distanceToNearestSettlement(i) <= 1) {
                    continue;
                }
                if (tile(i).terrain == PLAIN || tile(i).terrain == FOREST || tile(i).terrain == MOUNTAIN
                        || tile(i).terrain == DEEP_WATER) {
                    if (tile(i).terrain != DEEP_WATER || waterRuins < waterLimit) {
                        candidates.add(i);
                    }
                }
            }
            if (candidates.isEmpty()) {
                return;
            }
            int cell = candidates.get(randomInt(0, candidates.size()));
            if (tile(cell).terrain == DEEP_WATER) {
                waterRuins++;
            }
            tile(cell).resource = RUINS;
        }
    }

    private void ensureStartingResources() {
        for (Map.Entry<Integer, Types.TRIBE> entry : capitals.entrySet()) {
            int capital = entry.getKey();
            Types.TRIBE tribe = entry.getValue();
            switch (tribe) {
                case IMPERIUS:
                case QUETZALI:
                case YADAKK:
                    ensureResourceAround(capital, FRUIT, PLAIN, 2);
                    break;
                case BARDUR:
                case HOODRICK:
                    ensureResourceAround(capital, ANIMAL, FOREST, 2);
                    break;
                case KICKOO:
                    ensureFishAround(capital, 2);
                    break;
                case ZEBASI:
                    ensureResourceAround(capital, CROPS, PLAIN, 1);
                    break;
                case XIN_XI:
                case AI_MO:
                case VENGIR:
                    ensureResourceAround(capital, ORE, MOUNTAIN, 1);
                    break;
                default:
                    break;
            }
            ensureAtLeastTwoUsefulResources(capital);
        }
    }

    private void ensureAtLeastTwoUsefulResources(int capital) {
        int useful = 0;
        for (int cell : circle(capital, 1)) {
            char resource = tile(cell).resource;
            if (resource == FRUIT || resource == CROPS || resource == ANIMAL
                    || resource == FISH || resource == ORE) {
                useful++;
            }
        }
        while (useful < 2) {
            if (!ensureResourceAround(capital, FRUIT, PLAIN, useful + 1)) {
                break;
            }
            useful++;
        }
    }

    private boolean ensureResourceAround(int capital, char resource, char terrain, int quantity) {
        int count = countResourceAround(capital, resource);
        int guard = 24;
        while (count < quantity && guard-- > 0) {
            ArrayList<Integer> candidates = editableCapitalRing(capital);
            if (candidates.isEmpty()) {
                return false;
            }
            int cell = candidates.get(randomInt(0, candidates.size()));
            tile(cell).terrain = terrain;
            tile(cell).resource = resource;
            count = countResourceAround(capital, resource);
        }
        return count >= quantity;
    }

    private void ensureFishAround(int capital, int quantity) {
        int count = countResourceAround(capital, FISH);
        int guard = 24;
        while (count < quantity && guard-- > 0) {
            ArrayList<Integer> candidates = new ArrayList<>();
            for (int cell : crossNeighbors(capital)) {
                if (!isSettlement(cell)) {
                    candidates.add(cell);
                }
            }
            if (candidates.isEmpty()) {
                return;
            }
            int cell = candidates.get(randomInt(0, candidates.size()));
            tile(cell).terrain = SHALLOW_WATER;
            tile(cell).resource = FISH;
            for (int neighbor : crossNeighbors(cell)) {
                if (tile(neighbor).terrain == DEEP_WATER) {
                    tile(neighbor).terrain = SHALLOW_WATER;
                }
            }
            count = countResourceAround(capital, FISH);
        }
    }

    private ArrayList<Integer> editableCapitalRing(int capital) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int cell : circle(capital, 1)) {
            if (!isSettlement(cell)) {
                candidates.add(cell);
            }
        }
        return candidates;
    }

    private int countResourceAround(int capital, char resource) {
        int count = 0;
        for (int cell : circle(capital, 1)) {
            if (tile(cell).resource == resource) {
                count++;
            }
        }
        return count;
    }

    private int chooseVillageCandidate(int edgeBuffer, boolean beforeTerrain) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < tiles.length; i++) {
            if (validVillageCandidate(i, edgeBuffer, beforeTerrain)) {
                candidates.add(i);
            }
        }
        if (candidates.isEmpty()) {
            return -1;
        }
        return candidates.get(randomInt(0, candidates.size()));
    }

    private boolean validVillageCandidate(int cell, int edgeBuffer, boolean beforeTerrain) {
        if (edgeDistance(cell) < edgeBuffer || isSettlement(cell) || !farFromSettlements(cell, 3)) {
            return false;
        }
        char terrain = tile(cell).terrain;
        if (!isLandTerrain(terrain)) {
            return false;
        }
        return beforeTerrain || terrain == PLAIN || terrain == FOREST;
    }

    private void addVillage(int cell) {
        if (capitals.containsKey(cell)) {
            return;
        }
        tile(cell).terrain = VILLAGE;
        tile(cell).resource = NONE;
        villages.add(cell);
    }

    private void addCapital(int cell, Types.TRIBE tribe) {
        villages.remove(cell);
        tile(cell).terrain = CITY;
        tile(cell).resource = NONE;
        tile(cell).owner = tribe;
        capitals.put(cell, tribe);
    }

    private void assignTerritories() {
        if (capitals.isEmpty()) {
            return;
        }
        for (int i = 0; i < tiles.length; i++) {
            Types.TRIBE owner = null;
            double best = Double.POSITIVE_INFINITY;
            for (Map.Entry<Integer, Types.TRIBE> entry : capitals.entrySet()) {
                double score = distance(i, entry.getKey()) + Math.random() * 0.30;
                if (score < best) {
                    best = score;
                    owner = entry.getValue();
                }
            }
            tile(i).owner = owner;
        }
        for (Map.Entry<Integer, Types.TRIBE> entry : capitals.entrySet()) {
            tile(entry.getKey()).owner = entry.getValue();
        }
    }

    private List<List<Integer>> landComponents() {
        ArrayList<List<Integer>> components = new ArrayList<>();
        boolean[] seen = new boolean[tiles.length];
        for (int i = 0; i < tiles.length; i++) {
            if (seen[i] || !isLandTerrain(tile(i).terrain)) {
                continue;
            }
            ArrayList<Integer> component = new ArrayList<>();
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(i);
            seen[i] = true;
            while (!queue.isEmpty()) {
                int cell = queue.removeFirst();
                component.add(cell);
                for (int neighbor : crossNeighbors(cell)) {
                    if (!seen[neighbor] && isLandTerrain(tile(neighbor).terrain)) {
                        seen[neighbor] = true;
                        queue.add(neighbor);
                    }
                }
            }
            components.add(component);
        }
        return components;
    }

    private boolean sameLandmass(int a, int b) {
        if (!isLandTerrain(tile(a).terrain) || !isLandTerrain(tile(b).terrain)) {
            return false;
        }
        boolean[] seen = new boolean[tiles.length];
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(a);
        seen[a] = true;
        while (!queue.isEmpty()) {
            int cell = queue.removeFirst();
            if (cell == b) {
                return true;
            }
            for (int neighbor : crossNeighbors(cell)) {
                if (!seen[neighbor] && isLandTerrain(tile(neighbor).terrain)) {
                    seen[neighbor] = true;
                    queue.add(neighbor);
                }
            }
        }
        return false;
    }

    private boolean farFromSettlements(int cell, int minimumDistance) {
        for (int settlement : capitals.keySet()) {
            if (distance(cell, settlement) < minimumDistance) {
                return false;
            }
        }
        for (int settlement : villages) {
            if (distance(cell, settlement) < minimumDistance) {
                return false;
            }
        }
        return true;
    }

    private boolean adjacentToWater(int cell) {
        for (int neighbor : crossNeighbors(cell)) {
            if (isWaterTerrain(tile(neighbor).terrain)) {
                return true;
            }
        }
        return false;
    }

    private boolean adjacentToResource(int cell, char resource) {
        for (int neighbor : disk(cell, 1)) {
            if (neighbor != cell && tile(neighbor).resource == resource) {
                return true;
            }
        }
        return false;
    }

    private int distanceToNearestSettlement(int cell) {
        int best = size * 2;
        for (int capital : capitals.keySet()) {
            best = Math.min(best, distance(cell, capital));
        }
        for (int village : villages) {
            best = Math.min(best, distance(cell, village));
        }
        return best;
    }

    private int distanceToNearestCapital(int cell) {
        int best = size * 2;
        for (int capital : capitals.keySet()) {
            best = Math.min(best, distance(cell, capital));
        }
        return best;
    }

    private boolean isSettlement(int cell) {
        return capitals.containsKey(cell) || villages.contains(cell);
    }

    private int settlementCount() {
        return capitals.size() + villages.size();
    }

    private int countLand() {
        int count = 0;
        for (Tile tile : tiles) {
            if (isLandTerrain(tile.terrain)) {
                count++;
            }
        }
        return count;
    }

    private int randomLandCellAwayFromEdge(int edgeBuffer) {
        ArrayList<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < tiles.length; i++) {
            if (edgeDistance(i) >= edgeBuffer && isLandTerrain(tile(i).terrain) && !isSettlement(i)) {
                candidates.add(i);
            }
        }
        if (candidates.isEmpty()) {
            int cell = randomCellAwayFromEdge(edgeBuffer);
            forceLandPatch(cell, 1);
            return cell;
        }
        return candidates.get(randomInt(0, candidates.size()));
    }

    private int randomCellAwayFromEdge(int edgeBuffer) {
        int x = randomInt(edgeBuffer, size - edgeBuffer);
        int y = randomInt(edgeBuffer, size - edgeBuffer);
        return index(x, y);
    }

    private int randomCell() {
        return randomInt(0, tiles.length);
    }

    private int randomInt(int minInclusive, int maxExclusive) {
        if (maxExclusive <= minInclusive) {
            return minInclusive;
        }
        return (int) Math.floor(minInclusive + Math.random() * (maxExclusive - minInclusive));
    }

    private void forceLandPatch(int center, int radius) {
        for (int cell : disk(center, radius)) {
            if (!isSettlement(cell)) {
                tile(cell).terrain = PLAIN;
                tile(cell).resource = NONE;
            }
        }
    }

    private int edgeDistance(int cell) {
        int x = x(cell);
        int y = y(cell);
        return Math.min(Math.min(x, y), Math.min(size - 1 - x, size - 1 - y));
    }

    private ArrayList<Integer> circle(int center, int radius) {
        ArrayList<Integer> ring = new ArrayList<>();
        int row = y(center);
        int column = x(center);
        int top = row - radius;
        if (top >= 0 && top < size) {
            for (int j = column - radius; j < column + radius; j++) {
                if (j >= 0 && j < size) {
                    ring.add(index(j, top));
                }
            }
        }
        int bottom = row + radius;
        if (bottom >= 0 && bottom < size) {
            for (int j = column + radius; j > column - radius; j--) {
                if (j >= 0 && j < size) {
                    ring.add(index(j, bottom));
                }
            }
        }
        int left = column - radius;
        if (left >= 0 && left < size) {
            for (int i = row + radius; i > row - radius; i--) {
                if (i >= 0 && i < size) {
                    ring.add(index(left, i));
                }
            }
        }
        int right = column + radius;
        if (right >= 0 && right < size) {
            for (int i = row - radius; i < row + radius; i++) {
                if (i >= 0 && i < size) {
                    ring.add(index(right, i));
                }
            }
        }
        return ring;
    }

    private ArrayList<Integer> disk(int center, int radius) {
        ArrayList<Integer> cells = new ArrayList<>();
        for (int r = 1; r <= radius; r++) {
            cells.addAll(circle(center, r));
        }
        cells.add(center);
        return cells;
    }

    private List<Integer> crossNeighbors(int center) {
        ArrayList<Integer> neighbors = new ArrayList<>();
        int x = x(center);
        int y = y(center);
        if (x > 0) {
            neighbors.add(center - 1);
        }
        if (x < size - 1) {
            neighbors.add(center + 1);
        }
        if (y > 0) {
            neighbors.add(center - size);
        }
        if (y < size - 1) {
            neighbors.add(center + size);
        }
        return neighbors;
    }

    private int distance(int a, int b) {
        return Math.max(Math.abs(x(a) - x(b)), Math.abs(y(a) - y(b)));
    }

    private int x(int cell) {
        return cell % size;
    }

    private int y(int cell) {
        return cell / size;
    }

    private int index(int x, int y) {
        return y * size + x;
    }

    private Tile tile(int cell) {
        return tiles[cell];
    }

    private String[] toCsvLines() {
        String[] lines = new String[size];
        for (int y = 0; y < size; y++) {
            StringBuilder line = new StringBuilder();
            for (int x = 0; x < size; x++) {
                if (x > 0) {
                    line.append(',');
                }
                int cell = index(x, y);
                line.append(token(cell));
            }
            lines[y] = line.toString();
        }
        return lines;
    }

    private String token(int cell) {
        if (capitals.containsKey(cell)) {
            return CITY + ":" + capitals.get(cell).getKey();
        }
        if (villages.contains(cell)) {
            return VILLAGE + ":";
        }
        Tile tile = tile(cell);
        return tile.terrain + ":" + (tile.resource == NONE ? "" : String.valueOf(tile.resource));
    }

    private static boolean isLandTerrain(char terrain) {
        return terrain == PLAIN || terrain == FOREST || terrain == MOUNTAIN || terrain == VILLAGE || terrain == CITY;
    }

    private static boolean isWaterTerrain(char terrain) {
        return terrain == SHALLOW_WATER || terrain == DEEP_WATER;
    }

    private static int domainsPerSide(int players) {
        if (players <= 4) {
            return 2;
        }
        if (players <= 9) {
            return 3;
        }
        return 4;
    }

    private static int tinyIslandVillageCount(int size) {
        if (size <= 11) {
            return 0;
        }
        if (size <= 14) {
            return 1;
        }
        if (size <= 16) {
            return 2;
        }
        if (size <= 18) {
            return 3;
        }
        if (size <= 20) {
            return 4;
        }
        return 9;
    }

    private static int ruinCount(int size) {
        if (size <= 11) {
            return 4;
        }
        if (size <= 14) {
            return 5;
        }
        if (size <= 16) {
            return 7;
        }
        if (size <= 18) {
            return 9;
        }
        if (size <= 20) {
            return 11;
        }
        return 23;
    }

    private static TribeRates rates(Types.TRIBE tribe) {
        if (tribe == null) {
            return new TribeRates();
        }
        return TRIBE_RATES.get(tribe);
    }

    private static <T> void shuffle(List<T> list) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = (int) Math.floor(Math.random() * (i + 1));
            T tmp = list.get(i);
            list.set(i, list.get(j));
            list.set(j, tmp);
        }
    }

    private static class Tile {
        private char terrain = DEEP_WATER;
        private char resource = NONE;
        private Types.TRIBE owner;
    }

    private static class TribeRates {
        private double mountain = 1.0;
        private double forest = 1.0;
        private double fruit = 1.0;
        private double crop = 1.0;
        private double animal = 1.0;
        private double fish = 1.0;
        private double ore = 1.0;
    }
}
