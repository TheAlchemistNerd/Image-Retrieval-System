package com.retrieval.search.interfaces;

import com.retrieval.models.ImageFeature;

/**
 * An interface for search strategies that support efficient insertion
 * of a single item into an existing index. This is ideal for dynamic datasets.
 */
public interface Insertable {
    /**
     * Inserts a single image feature into an existing index.
     *
     * @param feature The ImageFeature to add.
     */
    void insert(ImageFeature feature);
}
