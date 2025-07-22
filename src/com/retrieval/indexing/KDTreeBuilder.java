package com.retrieval.indexing;

import com.retrieval.models.ImageFeature;
import java.util.*;

public class KDTreeBuilder {
    private KDNode root;

    public KDNode getKDTreeRoot(List<ImageFeature> features) {
        if (features == null || features.isEmpty()) {
            return null;
        }
        this.root = buildRecursive(new ArrayList<>(features), 0);
        return this.root;
    }

    private KDNode buildRecursive(List<ImageFeature> points, int depth) {
        if (points.isEmpty()) return null;
        int dimensions = points.get(0).getDimensions();
        int axis = depth % dimensions;

        points.sort(Comparator.comparingDouble(p -> p.getFeatureVector()[axis]));
        int medianIndex = points.size() / 2;

        KDNode node = new KDNode(points.get(medianIndex), axis);
        node.setLeft(buildRecursive(points.subList(0, medianIndex), depth + 1));
        node.setRight(buildRecursive(points.subList(medianIndex + 1, points.size()), depth + 1));

        return node;
    }
}
