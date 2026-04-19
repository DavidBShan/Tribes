package players.alphazero;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

public class ActionPolicyTrainingExample {
    public static final long NO_GROUP = Long.MIN_VALUE;

    public final double target;
    public final double[] features;
    public final long groupId;

    public ActionPolicyTrainingExample(double target, double[] features) {
        this(target, features, NO_GROUP);
    }

    public ActionPolicyTrainingExample(double target, double[] features, long groupId) {
        this.target = Math.max(0.0, Math.min(1.0, target));
        this.features = features;
        this.groupId = groupId;
    }

    public ActionPolicyTrainingExample withGroup(long groupId) {
        return new ActionPolicyTrainingExample(target, features, groupId);
    }

    static ArrayList<ArrayList<ActionPolicyTrainingExample>> groupedBatches(
            ArrayList<ActionPolicyTrainingExample> examples, int expectedFeatureCount) {
        LinkedHashMap<Long, ArrayList<ActionPolicyTrainingExample>> byGroup = new LinkedHashMap<>();
        for (ActionPolicyTrainingExample example : examples) {
            if (example == null || example.groupId == NO_GROUP
                    || example.features == null || example.features.length != expectedFeatureCount) {
                continue;
            }
            ArrayList<ActionPolicyTrainingExample> group = byGroup.get(example.groupId);
            if (group == null) {
                group = new ArrayList<>();
                byGroup.put(example.groupId, group);
            }
            group.add(example);
        }

        ArrayList<ArrayList<ActionPolicyTrainingExample>> groups = new ArrayList<>();
        for (Map.Entry<Long, ArrayList<ActionPolicyTrainingExample>> entry : byGroup.entrySet()) {
            if (entry.getValue().size() > 1) {
                groups.add(entry.getValue());
            }
        }
        return groups;
    }

    static double targetSum(ArrayList<ActionPolicyTrainingExample> group) {
        double sum = 0.0;
        for (ActionPolicyTrainingExample example : group) {
            sum += Math.max(0.0, example.target);
        }
        return sum;
    }
}
