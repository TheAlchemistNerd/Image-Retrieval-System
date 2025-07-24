package com.retrieval.indexing.BallTree;

import com.retrieval.models.ImageFeature;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Improved Ball Tree builder with better splitting strategies and error handling.
 */
public class BallTreeBuilder {
    private static final Logger log = LoggerFactory.getLogger(BallTreeBuilder.class);
    private static final int DEFAULT_LEAF_SIZE = 50;
    private static final int MIN_SPLIT_SIZE = 2;
    private static final Random random = new Random();

    private final int leafSize;

    public BallTreeBuilder() {
        this(DEFAULT_LEAF_SIZE);
    }

    public BallTreeBuilder(int leafSize) {
        if (leafSize <= 0) {
            throw new IllegalArgumentException("Leaf size must be positive");
        }
        this.leafSize = leafSize;
    }

    public BallTreeNode buildBallTree(List<ImageFeature> features) {
        if (features == null || features.isEmpty()) {
            log.warn("Attempted to build Ball Tree with null or empty feature list.");
            return null;
        }

        log.info("Starting Ball Tree construction with {} features, leaf size = {}.",
                features.size(), leafSize);

        BallTreeNode root = buildRecursive(new ArrayList<>(features), 0);
        log.info("Ball Tree construction complete.");
        return root;
    }

    /**
     * Recursively builds the Ball Tree with improved splitting strategy.
     */
    private BallTreeNode buildRecursive(List<ImageFeature> currentFeatures, int depth) {
        if (currentFeatures.isEmpty()) {
            return null;
        }

        // Calculate centroid and radius for the current ball
        double[] centroid = calculateCentroid(currentFeatures);
        double radius = calculateRadius(currentFeatures, centroid);

        // Create leaf node if we're at the leaf size threshold
        if (currentFeatures.size() <= leafSize) {
            return new LeafNode(centroid, radius, currentFeatures);
        }

        // Find the best split using farthest point heuristic
        SplitResult splitResult = findBestSplit(currentFeatures);

        if (splitResult == null || splitResult.leftSubset.isEmpty() || splitResult.rightSubset.isEmpty()) {
            // Fallback: if we can't split effectively, create a leaf
            log.debug("Cannot split {} features at depth {}, creating leaf",
                    currentFeatures.size(), depth);
            return new LeafNode(centroid, radius, currentFeatures);
        }

        // Create internal node with children
        InternalNode node = new InternalNode(centroid, radius);
        node.setLeftChild(buildRecursive(splitResult.leftSubset, depth + 1));
        node.setRightChild(buildRecursive(splitResult.rightSubset, depth + 1));

        return node;
    }

    /**
     * Improved splitting strategy using farthest point heuristic.
     */
    private SplitResult findBestSplit(List<ImageFeature> features) {
        if (features.size() < MIN_SPLIT_SIZE) {
            return null;
        }

        // Step 1: Pick a random starting point
        ImageFeature p1Feature = features.get(random.nextInt(features.size()));
        double[] p1 = p1Feature.getFeatureVector();

        // Step 2: Find the point farthest from p1
        double maxDist1 = -1.0;
        double[] p2 = null;
        for (ImageFeature feature : features) {
            double dist = FeatureUtils.euclideanDistance(p1, feature.getFeatureVector());
            if (dist > maxDist1) {
                maxDist1 = dist;
                p2 = feature.getFeatureVector();
            }
        }

        // Step 3: Find the point farthest from p2 (this becomes our new p1)
        if (p2 != null) {
            double maxDist2 = -1.0;
            double[] newP1 = null;
            for (ImageFeature feature : features) {
                double dist = FeatureUtils.euclideanDistance(p2, feature.getFeatureVector());
                if (dist > maxDist2) {
                    maxDist2 = dist;
                    newP1 = feature.getFeatureVector();
                }
            }
            if (newP1 != null) {
                p1 = newP1;
            }
        }

        // Step 4: Split based on distance to p1 vs p2
        List<ImageFeature> leftSubset = new ArrayList<>();
        List<ImageFeature> rightSubset = new ArrayList<>();

        for (ImageFeature feature : features) {
            double[] vector = feature.getFeatureVector();
            double distToP1 = FeatureUtils.euclideanDistance(vector, p1);
            double distToP2 = p2 != null ? FeatureUtils.euclideanDistance(vector, p2) : Double.MAX_VALUE;

            if (distToP1 <= distToP2) {
                leftSubset.add(feature);
            } else {
                rightSubset.add(feature);
            }
        }

        // Handle degenerate cases where all points are equidistant
        if (leftSubset.isEmpty() || rightSubset.isEmpty()) {
            return balancedSplit(features);
        }

        return new SplitResult(leftSubset, rightSubset);
    }

    /**
     * Fallback balanced split when farthest point heuristic fails.
     */
    private SplitResult balancedSplit(List<ImageFeature> features) {
        if (features.size() < MIN_SPLIT_SIZE) {
            return null;
        }

        // Shuffle to avoid bias, then split in half
        List<ImageFeature> shuffled = new ArrayList<>(features);
        Collections.shuffle(shuffled, random);

        int midPoint = shuffled.size() / 2;
        List<ImageFeature> leftSubset = shuffled.subList(0, midPoint);
        List<ImageFeature> rightSubset = shuffled.subList(midPoint, shuffled.size());

        return new SplitResult(new ArrayList<>(leftSubset), new ArrayList<>(rightSubset));
    }

    /**
     * Calculates the centroid with improved error handling.
     */
    private double[] calculateCentroid(List<ImageFeature> features) {
        if (features.isEmpty()) {
            throw new IllegalArgumentException("Cannot calculate centroid for an empty list of features.");
        }

        int dimensions = features.get(0).getDimensions();
        double[] centroid = new double[dimensions];

        for (ImageFeature feature : features) {
            double[] vector = feature.getFeatureVector();
            if (vector.length != dimensions) {
                throw new IllegalArgumentException(
                        String.format("Inconsistent feature vector dimensions. Expected %d, got %d.",
                                dimensions, vector.length));
            }
            for (int i = 0; i < dimensions; i++) {
                centroid[i] += vector[i];
            }
        }

        for (int i = 0; i < dimensions; i++) {
            centroid[i] /= features.size();
        }

        return centroid;
    }

    /**
     * Calculates the radius with validation.
     */
    private double calculateRadius(List<ImageFeature> features, double[] centroid) {
        double maxRadius = 0.0;
        for (ImageFeature feature : features) {
            double distance = FeatureUtils.euclideanDistance(feature.getFeatureVector(), centroid);
            maxRadius = Math.max(maxRadius, distance);
        }
        return maxRadius;
    }

    /**
     * Helper class to hold split results.
     */
    private static class SplitResult {
        final List<ImageFeature> leftSubset;
        final List<ImageFeature> rightSubset;

        SplitResult(List<ImageFeature> leftSubset, List<ImageFeature> rightSubset) {
            this.leftSubset = leftSubset;
            this.rightSubset = rightSubset;
        }
    }
}