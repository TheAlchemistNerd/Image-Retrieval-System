package com.retrieval.indexing.BallTree;

import com.retrieval.models.ImageFeature;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Represents leaf node in a Ball Tree.
 * Leaf nodes directly contain the actual ImageFeatures.
 */
public class LeafNode extends BallTreeNode {
    private final List<ImageFeature> features;

    /**
     * Constructs a leaf BallTreeNode.
     *
     * @param centroid The centroid of the ball.
     * @param radius   The radius of the ball.
     * @param features The list of ImageFeatures contained within this leaf node.
     * @throws IllegalArgumentException if features is null or empty
     */
    public LeafNode(double[] centroid, double radius, List<ImageFeature> features) {
        super(centroid, radius);

        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("Leaf node must contain at least one feature");
        }

        // Create defensive copy to prevent external modification
        this.features = Collections.unmodifiableList(new ArrayList<>(features));
    }

    /**
     * Gets an immutable view of the ImageFeatures stored in this node.
     * The returned list cannot be modified, ensuring data integrity.
     *
     * @return An immutable list of ImageFeatures.
     */
    public List<ImageFeature> getFeatures() {
        return features; // Already immutable from constructor
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public int getFeatureCount() {
        return features.size();
    }

    /**
     * Checks if this leaf contains a specific ImageFeature.
     *
     * @param feature The feature to search for
     * @return true if the feature is contained in this leaf
     */
    public boolean containsFeature(ImageFeature feature) {
        return features.contains(feature);
    }

    /**
     * Gets the feature at the specified index.
     *
     * @param index The index of the feature to retrieve
     * @return The ImageFeature at the specified index
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public ImageFeature getFeature(int index) {
        return features.get(index);
    }

    /**
     * Gets basic statistics about the features in this leaf.
     *
     * @return A string with statistics about this leaf
     */
    public String getStatistics() {
        if (features.isEmpty()) {
            return "Empty leaf";
        }

        int dimensions = features.get(0).getDimensions();
        double minRadius = Double.MAX_VALUE;
        double maxRadius = Double.MIN_VALUE;
        double avgRadius = 0.0;

        for (ImageFeature feature : features) {
            double distance = 0.0;
            double[] featureVector = feature.getFeatureVector();
            for (int i = 0; i < dimensions; i++) {
                double diff = featureVector[i] - centroid[i];
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);

            minRadius = Math.min(minRadius, distance);
            maxRadius = Math.max(maxRadius, distance);
            avgRadius += distance;
        }
        avgRadius /= features.size();

        return String.format("LeafNode{features=%d, radius=%.3f, distances=[%.3f-%.3f], avg=%.3f}",
                features.size(), radius, minRadius, maxRadius, avgRadius);
    }
}