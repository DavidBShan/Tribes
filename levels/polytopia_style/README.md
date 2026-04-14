Polytopia-style generated levels
================================

This folder contains original Tribes CSV levels generated from a research-derived
approximation of Polytopia map-generation rules. The files are not copied from
official Polytopia maps.

The generator lives at:

    src/core/levelgen/PolytopiaStyleLevelGenerator.java

The batch runner lives at:

    src/GeneratePolytopiaStyleLevels.java

Generation model
----------------

The generator uses Math.random() throughout and creates new random layouts for:

- Drylands
- Lakes
- Continents
- Pangea
- Archipelago
- Water World

It follows these broad Polytopia-style rules:

- official square size families: 11, 14, 16, 18, 20, and 30
- different wetness and landmass patterns per map type
- quadrant/domain starts for Drylands, Lakes, Archipelago, and Water World
- village-first capital conversion for Continents and Pangea
- suburb villages on Lakes and Archipelago
- settlement-spaced post-terrain villages
- tribe-biased forests, mountains, and resources
- resources concentrated within two tiles of settlements
- ruin counts by map size

Regenerate the pack
-------------------

From the Tribes repo root:

    javac -cp src:lib/json.jar -d build src/GeneratePolytopiaStyleLevels.java src/core/levelgen/PolytopiaStyleLevelGenerator.java
    java -cp build:src:lib/json.jar GeneratePolytopiaStyleLevels levels/polytopia_style 2

The final number is variants per map-type/size/player-count combination.
The default pack uses 2 variants, producing 216 CSV levels.
