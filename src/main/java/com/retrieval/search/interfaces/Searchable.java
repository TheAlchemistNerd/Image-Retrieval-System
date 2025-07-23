package com.retrieval.search.interfaces;

import com.retrieval.models.ImageFeature;

import java.util.List;

/**
 * The most fundamental search interface.
 * Any class that can return results for a query should implement this.
 * This is the base contract for all retrieval strategies.
 */
public interface Searchable {
    /**
     * Performs a query to find the top K most similar images to a given vector.
     *
     * @param queryVector The feature vector of the query image.
     * @param k           The number of similar images to retrieve.
     * @return A list of the top K matching ImageFeature objects, sorted by similarity.
     */
    List<ImageFeature> query(double[] queryVector, int k);
}
