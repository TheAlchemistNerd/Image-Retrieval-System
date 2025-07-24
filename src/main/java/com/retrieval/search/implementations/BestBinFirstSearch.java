package com.retrieval.search.implementations;

import com.retrieval.indexing.KDTree.KDNode;
import com.retrieval.indexing.KDTree.KDTreeBuilder;
import com.retrieval.models.ImageFeature;
import com.retrieval.utils.FeatureUtils;
import com.retrieval.search.annotations.SearchCapabilities;
import com.retrieval.search.interfaces.Buildable;
import com.retrieval.search.interfaces.Searchable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * This strategy builds a K-D tree index and searches using approximate nearest neighbor techniques.
 * Uses either cosine or euclidean distance similarity
 */

@SearchCapabilities(insertable = false, buildable = true, searchable = true)
public class BestBinFirstSearch implements Searchable, Buildable {
    private static final Logger log = LoggerFactory.getLogger(BestBinFirstSearch.class);

    private KDNode root;
    private final int maxChecks;
    private final boolean useCosineSimilarity;

    /**
     * Internal node class for the priority queue during search.
     */
    private static class HeapNode implements Comparable<HeapNode> {
        final KDNode node;
        final double priority;

        HeapNode(KDNode node, double priority) {
            this.node = node;
            this.priority = priority;
        }

        @Override
        public int compareTo(HeapNode other) {
            return Double.compare(this.priority, other.priority);
        }
    }

    public BestBinFirstSearch() {
        this(1000, true);
    }

    public BestBinFirstSearch(int maxChecks) {
        this(maxChecks, true);
    }

    public BestBinFirstSearch(int maxChecks, boolean useCosineSimilarity) {
        if (maxChecks <= 0) {
            throw new IllegalArgumentException("maxChecks must be positive");
        }
        this.maxChecks = maxChecks;
        this.useCosineSimilarity = useCosineSimilarity;
    }

    @Override
    public void buildIndex(List<ImageFeature> features) {
        if (features == null || features.isEmpty()) {
            log.warn("Building index with null or empty feature list");
            this.root = null;
            return;
        }

        log.info("Building K-D tree index with {} features", features.size());
        KDTreeBuilder builder = new KDTreeBuilder();
        this.root = builder.getKDTreeRoot(features);
        log.info("K-D tree index built successfully");
    }

    @Override
    public List<ImageFeature> query(double[] queryVector, int k) {
        if (queryVector == null || queryVector.length == 0) {
            throw new IllegalArgumentException("Query vector cannot be null or empty");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (root == null) {
            throw new IllegalStateException("Index has not been built or is empty");
        }

        PriorityQueue<HeapNode> searchQueue = new PriorityQueue<>();
        PriorityQueue<Map.Entry<ImageFeature, Double>> resultHeap =
                new PriorityQueue<>(Map.Entry.comparingByValue(Collections.reverseOrder()));
        Set<KDNode> visited = new HashSet<>();

        searchQueue.add(new HeapNode(root, 0.0));
        int checks = 0;

        while (!searchQueue.isEmpty() && checks < maxChecks) {
            HeapNode current = searchQueue.poll();
            KDNode node = current.node;

            if (visited.contains(node)) continue;
            visited.add(node);
            checks++;

            ImageFeature feature = node.getFeature();
            double distance = useCosineSimilarity
                    ? FeatureUtils.cosineDistance(queryVector, feature.getFeatureVector())
                    : FeatureUtils.euclideanDistance(queryVector, feature.getFeatureVector());

            resultHeap.offer(Map.entry(feature, distance));
            if (resultHeap.size() > k) {
                resultHeap.poll();
            }

            int axis = node.getAxis();
            double splitValue = feature.getFeatureVector()[axis];
            double queryValue = queryVector[axis];

            KDNode nearChild = queryValue < splitValue ? node.getLeft() : node.getRight();
            KDNode farChild = queryValue < splitValue ? node.getRight() : node.getLeft();

            if (nearChild != null) {
                searchQueue.offer(new HeapNode(nearChild, 0.0));
            }

            // Always check far child if within hyperplane distance
            if (farChild != null) {
                double diff = queryValue - splitValue;
                double penalty = useCosineSimilarity ? 0.0 : diff * diff; // cosine doesn't need penalty
                searchQueue.offer(new HeapNode(farChild, penalty));
            }
        }

        List<ImageFeature> results = new ArrayList<>();
        resultHeap.stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(k)
                .forEachOrdered(entry -> results.add(entry.getKey()));

        return results;
    }
}
