import core.levelgen.PolytopiaStyleLevelGenerator;
import core.levelgen.PolytopiaStyleLevelGenerator.GeneratedLevel;
import core.levelgen.PolytopiaStyleLevelGenerator.MapType;

import java.io.File;
import java.io.FileWriter;

public class GeneratePolytopiaStyleLevels {

    private static final int[] SIZES = new int[]{11, 14, 16, 18, 20, 30};
    private static final int[] PLAYER_COUNTS = new int[]{2, 4, 8};

    public static void main(String[] args) throws Exception {
        File outputDir = args.length >= 1 ? new File(args[0]) : new File("levels/polytopia_style");
        int variants = args.length >= 2 ? Integer.parseInt(args[1]) : 2;

        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IllegalStateException("Could not create " + outputDir.getPath());
        }

        File manifest = new File(outputDir, "manifest.tsv");
        try (FileWriter manifestWriter = new FileWriter(manifest)) {
            manifestWriter.write("file\tmap_type\tsize\tplayers\ttribes\n");

            int count = 0;
            for (MapType mapType : MapType.values()) {
                for (int size : SIZES) {
                    for (int players : PLAYER_COUNTS) {
                        for (int variant = 1; variant <= variants; variant++) {
                            GeneratedLevel level = PolytopiaStyleLevelGenerator.generate(
                                    new PolytopiaStyleLevelGenerator.Config(size, players, mapType, null));
                            String filename = mapType.slug() + "_" + size + "x" + size + "_"
                                    + players + "p_" + pad(variant) + ".csv";
                            level.writeCsv(new File(outputDir, filename));
                            manifestWriter.write(filename + "\t" + level.summary() + "\n");
                            count++;
                        }
                    }
                }
            }

            manifestWriter.write("# generated\t" + count + "\n");
            System.out.println("Generated " + count + " Polytopia-style Tribes levels in "
                    + outputDir.getPath());
        }
    }

    private static String pad(int n) {
        return n < 10 ? "00" + n : n < 100 ? "0" + n : String.valueOf(n);
    }
}
