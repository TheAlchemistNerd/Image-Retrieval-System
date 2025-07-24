package com.retrieval.search.implementations;

import com.retrieval.indexing.BallTree.BallTreeBuilder;
import com.retrieval.indexing.BallTree.BallTreeNode;
import com.retrieval.indexing.BallTree.InternalNode;
import com.retrieval.indexing.BallTree.LeafNode;
import com.retrieval.models.ImageFeature;
import com.retrieval.search.annotations.SearchCapabilities;
import com.retrieval.search.interfaces.Buildable;
import com.retrieval.search.interfaces.Searchable;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.AbstractMap.SimpleEntry;

/**
 * Implements a search strategy using a Ball Tree for efficient approximate nearest neighbor (ANN) search.
 * Ball Trees are hierarchical data structures that partition data into nested hyperspheres,
 * allowing for faster distance-based queries by pruning branches that cannot contain
 * the nearest neighbors.
 */
@SearchCapabilities(insertable = false, buildable = true, searchable = true)
public class BallTreeSearch implements Buildable, Searchable {

    private static final Logger log = LoggerFactory.getLogger(BallTreeSearch.class);
    private BallTreeNode root;
    private int indexSize = 0;

    /**
     * Builds the Ball Tree index from a list of image features.
     * This method constructs the hierarchical Ball Tree structure by recursively
     * partitioning the feature space.
     *
     * @param features The dataset of ImageFeature objects to be indexed.
     */
    @Override
    public void buildIndex(List<ImageFeature> features) {
        if (features == null || features.isEmpty()) {
            log.warn("Building Ball Tree index with null or empty feature list. Index will be empty.");
            this.root = null;
            this.indexSize = 0;
            return;
        }

        log.info("Building Ball Tree index with {} features.", features.size());
        BallTreeBuilder builder = new BallTreeBuilder();
        this.root = builder.buildBallTree(features);
        this.indexSize = features.size();

        if (this.root != null) {
            log.info("Ball Tree index built successfully with {} features.", this.indexSize);
        } else {
            log.error("Failed to build Ball Tree index.");
            this.indexSize = 0;
        }
    }

    /**
     * Performs a K-nearest neighbors (KNN) query on the Ball Tree.
     * It efficiently searches for the 'k' closest ImageFeatures to the given
     * query vector by traversing the tree and pruning branches that are
     * unlikely to contain nearest neighbors. Uses Euclidean distance.
     *
     * @param queryVector The feature vector of the query image.
     * @param k           The number of similar images to retrieve.
     * @return A list of the top K matching ImageFeature objects, sorted by increasing Euclidean distance.
     * @throws IllegalArgumentException if queryVector is null or empty, or k is not positive.
     * @throws IllegalStateException    if the index has not been built.
     */
    @Override
    public List<ImageFeature> query(double[] queryVector, int k) {
        if (queryVector == null || queryVector.length == 0) {
            throw new IllegalArgumentException("Query vector cannot be null or empty.");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive.");
        }
        if (root == null) {
            throw new IllegalStateException("Ball Tree index has not been built or is empty.");
        }

        // Limit k to the actual number of indexed features
        k = Math.min(k, indexSize);

        // PriorityQueue to store the k nearest neighbors found so far.
        // Uses max-heap (reverse order) to keep the largest distance at the top for easy removal.
        PriorityQueue<SimpleEntry<ImageFeature, Double>> topKResults = new PriorityQueue<>(
                k,
                (entry1, entry2) -> Double.compare(entry2.getValue(), entry1.getValue()) // Fixed: proper comparator syntax
        );

        // PriorityQueue for nodes to visit during traversal, ordered by minimum possible distance.
        PriorityQueue<SimpleEntry<BallTreeNode, Double>> nodeQueue = new PriorityQueue<>(
                Comparator.comparingDouble(SimpleEntry::getValue)
        );

        nodeQueue.offer(new SimpleEntry<>(root, 0.0)); // Start with the root node

        int nodesVisited = 0;
        int leavesProcessed = 0;

        while (!nodeQueue.isEmpty()) {
            SimpleEntry<BallTreeNode, Double> currentEntry = nodeQueue.poll();
            BallTreeNode currentNode = currentEntry.getKey();
            double minDistanceToNode = currentEntry.getValue();
            nodesVisited++;

            // Pruning: if we have k results and the minimum possible distance to this node
            // is greater than or equal to the worst result so far, skip this branch
            if (topKResults.size() >= k && minDistanceToNode >= topKResults.peek().getValue()) {
                continue;
            }

            if (currentNode instanceof LeafNode leafNode) {
                leavesProcessed++;
                // Process all features in this leaf node
                for (ImageFeature feature : leafNode.getFeatures()) {
                    double distance = FeatureUtils.euclideanDistance(queryVector, feature.getFeatureVector());

                    if (topKResults.size() < k) {
                        // We don't have k results yet, so add this one
                        topKResults.offer(new SimpleEntry<>(feature, distance));
                    } else if (distance < topKResults.peek().getValue()) {
                        // This result is better than our worst current result
                        topKResults.poll(); // Remove the worst
                        topKResults.offer(new SimpleEntry<>(feature, distance)); // Add the new one
                    }
                }
            } else if (currentNode instanceof InternalNode internalNode) {
                // For internal nodes, add children to the queue with their minimum possible distances
                BallTreeNode leftChild = internalNode.getLeftChild();
                BallTreeNode rightChild = internalNode.getRightChild();

                if (leftChild != null) {
                    double distToLeftCentroid = FeatureUtils.euclideanDistance(queryVector, leftChild.getCentroid());
                    double minPossibleDistToLeft = Math.max(0.0, distToLeftCentroid - leftChild.getRadius());
                    nodeQueue.offer(new SimpleEntry<>(leftChild, minPossibleDistToLeft));
                }

                if (rightChild != null) {
                    double distToRightCentroid = FeatureUtils.euclideanDistance(queryVector, rightChild.getCentroid());
                    double minPossibleDistToRight = Math.max(0.0, distToRightCentroid - rightChild.getRadius());
                    nodeQueue.offer(new SimpleEntry<>(rightChild, minPossibleDistToRight));
                }
            }
        }

        // Convert the priority queue to a sorted list (ascending order by distance)
        List<SimpleEntry<ImageFeature, Double>> resultPairs = new ArrayList<>();
        while (!topKResults.isEmpty()) {
            resultPairs.add(topKResults.poll());
        }

        // Reverse to get ascending order (since we used a max-heap)
        Collections.reverse(resultPairs);

        // Extract just the ImageFeatures
        List<ImageFeature> results = new ArrayList<>();
        for (SimpleEntry<ImageFeature, Double> pair : resultPairs) {
            results.add(pair.getKey());
        }

        log.debug("Query completed: visited {} nodes, processed {} leaves, returned {} results",
                nodesVisited, leavesProcessed, results.size());

        return results;
    }

    /**
     * Gets the number of features in the index.
     * @return The number of indexed features
     */
    public int getIndexSize() {
        return indexSize;
    }

    /**
     * Checks if the index is built and ready for queries.
     * @return true if the index is built, false otherwise
     */
    public boolean isIndexBuilt() {
        return root != null;
    }
}