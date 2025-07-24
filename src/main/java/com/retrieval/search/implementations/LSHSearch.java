package com.retrieval.search.implementations;

import com.retrieval.models.ImageFeature;
import com.retrieval.search.annotations.SearchCapabilities;
import com.retrieval.search.interfaces.Buildable;
import com.retrieval.search.interfaces.Searchable;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * Implements Locality Sensitive Hashing (LSH) for approximate nearest neighbor search.
 * LSH is suitable for high-dimensional data where exact nearest neighbor search is
 * computationally expensive. It uses multiple hash tables to group similar items
 * into the same buckets with high probability.
 * This implementation uses a simple LSH scheme based on random projection for
 * angular (cosine) similarity.
 * Capabilities: Buildable, Searchable. Not Insertable due to the nature of LSH
 * where insertions are typically batched or require re-hashing.
 */
@SearchCapabilities(insertable = false, buildable = true, searchable = true)
public class LSHSearch implements Buildable, Searchable {

    private static final Logger log = LoggerFactory.getLogger(LSHSearch.class);

    private int numberOfHashTables; // L parameter in LSH, number of independent hash tables
    private int numberOfHashesPerTable; // K parameter in LSH, number of hash functions per table
    private List<Map<String, List<ImageFeature>>> hashTables; // Each map is a hash table: hash_code -> list of features
    private List<double[][]> randomProjections; // Random vectors for each hash table

    /**
     * Constructs an LSHSearch instance with default parameters.
     * Default: 10 hash tables, 8 hashes per table.
     */
    public LSHSearch() {
        this(10, 8);
    }

    /**
     * Constructs an LSHSearch instance with specified parameters.
     *
     * @param numberOfHashTables     The number of independent hash tables to use (L).
     * @param numberOfHashesPerTable The number of hash functions (random projections)
     * to use for each hash table (K).
     * @throws IllegalArgumentException if numberOfHashTables or numberOfHashesPerTable is not positive.
     */
    public LSHSearch(int numberOfHashTables, int numberOfHashesPerTable) {
        if (numberOfHashTables <= 0 || numberOfHashesPerTable <= 0) {
            throw new IllegalArgumentException("Number of hash tables and hashes per table must be positive.");
        }
        this.numberOfHashTables = numberOfHashTables;
        this.numberOfHashesPerTable = numberOfHashesPerTable;
        this.hashTables = new ArrayList<>(numberOfHashTables);
        this.randomProjections = new ArrayList<>(numberOfHashTables);
    }

    /**
     * Builds the LSH index from a list of image features.
     * For each feature, it generates a hash code for each hash table based on
     * random projections and stores the feature in the corresponding bucket.
     *
     * @param features The dataset of ImageFeature objects. Each feature vector
     * is expected to be normalized for cosine similarity.
     */
    @Override
    public void buildIndex(List<ImageFeature> features) {
        if (features == null || features.isEmpty()) {
            log.warn("Building LSH index with null or empty feature list. Index will be empty.");
            clearIndex();
            return;
        }

        log.info("Building LSH index with {} features using {} tables and {} hashes per table.",
                features.size(), numberOfHashTables, numberOfHashesPerTable);

        // Clear previous index
        clearIndex();

        int dimensions = features.get(0).getDimensions();
        if (dimensions == 0) {
            log.error("Feature vectors have zero dimensions. Cannot build LSH index.");
            return;
        }

        // Generate random projections for all hash tables
        for (int i = 0; i < numberOfHashTables; i++) {
            double[][] projections = new double[numberOfHashesPerTable][dimensions];
            for (int j = 0; j < numberOfHashesPerTable; j++) {
                // Generate random vector components from a standard normal distribution
                for (int d = 0; d < dimensions; d++) {
                    projections[j][d] = ThreadLocalRandom.current().nextGaussian();
                }
                FeatureUtils.normalize(projections[j]); // Normalize projection vector
            }
            randomProjections.add(projections);
            hashTables.add(new HashMap<>());
        }

        // Populate hash tables
        for (ImageFeature feature : features) {
            double[] featureVector = feature.getFeatureVector();
            if (!FeatureUtils.isNormalized(featureVector)) {
                log.warn("Feature vector for image {} is not normalized. Normalizing copy for LSH.", feature.getImageId());
                featureVector = FeatureUtils.normalizedCopy(featureVector);
            }

            for (int i = 0; i < numberOfHashTables; i++) {
                String hashCode = generateHashCode(featureVector, randomProjections.get(i));
                hashTables.get(i)
                        .computeIfAbsent(hashCode, k -> new ArrayList<>())
                        .add(feature);
            }
        }
        log.info("LSH index built successfully.");
    }

    /**
     * Performs a query to find the top K most similar images to a given query vector.
     * The query vector is hashed into each of the LSH tables, and candidate features
     * from the corresponding buckets are collected. The exact cosine distance is
     * then computed for these candidates to find the top K nearest neighbors.
     *
     * @param queryVector The feature vector of the query image. Expected to be normalized.
     * @param k           The number of similar images to retrieve.
     * @return A list of the top K matching ImageFeature objects, sorted by similarity (smallest cosine distance first).
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
        if (hashTables == null || hashTables.isEmpty() || randomProjections.isEmpty()) {
            throw new IllegalStateException("LSH index has not been built or is empty.");
        }

        // Ensure query vector is normalized for cosine distance
        if (!FeatureUtils.isNormalized(queryVector)) {
            log.warn("Query vector is not normalized. Normalizing copy for LSH query.");
            queryVector = FeatureUtils.normalizedCopy(queryVector);
        }

        Set<ImageFeature> candidateFeatures = new HashSet<>();
        for (int i = 0; i < numberOfHashTables; i++) {
            String hashCode = generateHashCode(queryVector, randomProjections.get(i));
            List<ImageFeature> bucket = hashTables.get(i).get(hashCode);
            if (bucket != null) {
                candidateFeatures.addAll(bucket);
            }
        }

        if (candidateFeatures.isEmpty()) {
            log.info("No candidates found for the query vector in LSH buckets.");
            return new ArrayList<>();
        }

        // Perform exact k-NN search on candidates
        double[] finalQueryVector = queryVector;
        return candidateFeatures.parallelStream()
                .map(feature -> new AbstractMap.SimpleEntry<>(
                        feature,
                        FeatureUtils.cosineDistance(finalQueryVector, feature.getFeatureVector())
                ))
                .sorted(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                .limit(k)
                .map(AbstractMap.SimpleEntry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * Generates a hash code for a given feature vector based on a set of random projections.
     * The hash code is a binary string where each bit is determined by the sign of the
     * dot product between the feature vector and a random projection.
     *
     * @param featureVector   The vector for which to generate the hash code.
     * @param projections The set of random projection vectors for one hash table.
     * @return A string representing the binary hash code.
     */
    private String generateHashCode(double[] featureVector, double[][] projections) {
        StringBuilder hashCodeBuilder = new StringBuilder();
        for (double[] projection : projections) {
            double dotProduct = 0.0;
            for (int i = 0; i < featureVector.length; i++) {
                dotProduct += featureVector[i] * projection[i];
            }
            hashCodeBuilder.append(dotProduct >= 0 ? '1' : '0');
        }
        return hashCodeBuilder.toString();
    }

    /**
     * Clears the LSH index, removing all hash tables and random projections.
     */
    private void clearIndex() {
        this.hashTables.clear();
        this.randomProjections.clear();
    }
}