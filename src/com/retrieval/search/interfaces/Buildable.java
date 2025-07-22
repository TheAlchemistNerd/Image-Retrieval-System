package com.retrieval.search.interfaces;

import com.retrieval.models.ImageFeature;

import java.util.List;

/**
 * An interface for search strategies that can be built or trained from a
 * bulk collection of features. This is typical for static indexes.
 */
public interface Buildable {
    /**
     * Builds or initializes the search index from a list of image features.
     *
     * @param features The dataset of ImageFeature objects.
     */
    void buildIndex(List<ImageFeature> features);
}
