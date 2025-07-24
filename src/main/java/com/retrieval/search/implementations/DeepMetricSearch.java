package com.retrieval.search.implementations;

import com.retrieval.models.ImageFeature;
import com.retrieval.search.interfaces.Insertable;
import com.retrieval.utils.FeatureUtils;
import com.retrieval.search.annotations.SearchCapabilities;
import com.retrieval.search.interfaces.Buildable;
import com.retrieval.search.interfaces.Searchable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * Search class for deep learning visual embeddings
 * Performs brute force linear scan k-NN (k-nearest neighbors) search using cosine distance:
 */
@SearchCapabilities(insertable = true, buildable = true, searchable = true)
public class DeepMetricSearch implements Searchable, Buildable, Insertable {
    private static final Logger log = LoggerFactory.getLogger(DeepMetricSearch.class);

    private final List<ImageFeature> features = new ArrayList<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    @Override
    public void buildIndex(List<ImageFeature> featureList) {
        lock.writeLock().lock();
        try {
            features.clear();
            if (featureList != null) {
                features.addAll(featureList);
            }
            log.info("Built index with {} items", features.size());
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public void insert(ImageFeature feature) {
        if (feature == null) {
            throw new IllegalArgumentException("Feature cannot be null");
        }

        lock.writeLock().lock();
        try {
            features.add(feature);
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public List<ImageFeature> query(double[] queryVector, int k) {
        if (queryVector == null || queryVector.length == 0) {
            throw new IllegalArgumentException("Query vector cannot be null or empty");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }

        lock.readLock().lock();
        try {
            if (features.isEmpty()) {
                return new ArrayList<>();
            }

            // Performs k-NN (k-nearest neighbors) search using cosine distance:
            return features.parallelStream()
                    .map(feature -> new SimpleEntry<>(
                            feature,
                            FeatureUtils.cosineDistance(queryVector, feature.getFeatureVector())
                    ))
                    .sorted(Comparator.comparingDouble(SimpleEntry::getValue))
                    .limit(k)
                    .map(SimpleEntry::getKey)
                    .collect(Collectors.toList());
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Get the current size of the index.
     * @return Number of features in the index
     */
    public int size() {
        lock.readLock().lock();
        try {
            return features.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Clear all features from the index.
     */
    public void clear() {
        lock.writeLock().lock();
        try {
            features.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }
}