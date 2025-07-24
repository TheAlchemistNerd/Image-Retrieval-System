package com.retrieval.indexing.BallTree;

import java.util.Arrays;

/**
 * Abstract base class for a node in a Ball Tree.
 * Each node defines a hypersphere (a "ball") that encloses a subset of the data points.
 * It stores the centroid of the ball and its radius.
 */
public abstract class BallTreeNode {
    protected final double[] centroid;
    protected final double radius;

    /**
     * Constructs a BallTreeNode with a centroid and radius.
     *
     * @param centroid The centroid of the ball (will be copied to prevent external modification).
     * @param radius   The radius of the ball.
     * @throws IllegalArgumentException if centroid is null or radius is negative
     */
    public BallTreeNode(double[] centroid, double radius) {
        if (centroid == null) {
            throw new IllegalArgumentException("Centroid cannot be null");
        }
        if (radius < 0) {
            throw new IllegalArgumentException("Radius cannot be negative");
        }

        // Defensive copy to prevent external modification
        this.centroid = Arrays.copyOf(centroid, centroid.length);
        this.radius = radius;
    }

    /**
     * Gets a copy of the centroid of the ball.
     * Returns a copy to prevent external modification of the internal state.
     *
     * @return A copy of the centroid as a double array.
     */
    public double[] getCentroid() {
        return Arrays.copyOf(centroid, centroid.length);
    }

    /**
     * Gets the radius of the ball.
     *
     * @return The radius as a double.
     */
    public double getRadius() {
        return radius;
    }

    /**
     * Gets the dimensionality of the feature space.
     *
     * @return The number of dimensions
     */
    public int getDimensions() {
        return centroid.length;
    }

    /**
     * Checks if this node represents a leaf node.
     *
     * @return true if this is a leaf node, false otherwise
     */
    public abstract boolean isLeaf();

    /**
     * Gets the number of features contained in this subtree.
     *
     * @return The number of features
     */
    public abstract int getFeatureCount();

    @Override
    public String toString() {
        return String.format("%s{centroid=[%.3f, %.3f, ...], radius=%.3f, features=%d}",
                getClass().getSimpleName(),
                centroid.length > 0 ? centroid[0] : 0.0,
                centroid.length > 1 ? centroid[1] : 0.0,
                radius,
                getFeatureCount());
    }
}
