package com.retrieval.features.extractor;

/**
 * The Strategy interface for all feature extraction algorithms.
 * Any class that can extract a feature vector from an image should implement this interface.
 * This allows for a plug-and-play architecture where different extraction methods
 * can be used interchangeably.
 */

@FunctionalInterface
public interface Extractable {

    /**
     * Extracts a numerical feature vector from the given image file.
     *
     * @param imagePath The path to the image file.
     * @return A double array representing the feature vector. Returns an empty array if extraction fails.
     */

    double[] extract(String imagePath);
}
