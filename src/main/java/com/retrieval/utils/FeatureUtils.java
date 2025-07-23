package com.retrieval.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Consolidated utility class for feature vector operations and distance calculations.
 * This class provides common mathematical operations used across the image retrieval system.
 */
public class FeatureUtils {
    private static final Logger log = LoggerFactory.getLogger(FeatureUtils.class);

    private static final double EPSILON = 1e-12; // Small value to prevent division by zero

    /**
     * Normalizes a feature vector to unit length (L2 normalization).
     * This operation is performed in-place, modifying the original vector.
     *
     * @param vector The feature vector to normalize
     * @throws IllegalArgumentException if vector is null or empty
     */
    public static void normalize(double[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }

        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);

        if (norm > EPSILON) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        } else {
            log.warn("Vector has near-zero norm ({}), skipping normalization", norm);
        }
    }

    /**
     * Creates a normalized copy of the input vector without modifying the original.
     *
     * @param vector The feature vector to normalize
     * @return A new normalized vector
     * @throws IllegalArgumentException if vector is null or empty
     */
    public static double[] normalizedCopy(double[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }

        double[] copy = new double[vector.length];
        System.arraycopy(vector, 0, copy, 0, vector.length);
        normalize(copy);
        return copy;
    }

    /**
     * Calculates the cosine distance between two feature vectors.
     * Cosine distance = 1 - cosine similarity, where cosine similarity ranges from -1 to 1.
     * The returned distance ranges from 0 (identical direction) to 2 (opposite direction).
     *
     * @param vectorA First feature vector
     * @param vectorB Second feature vector
     * @return The cosine distance between the vectors
     * @throws IllegalArgumentException if vectors are null, empty, or have different lengths
     */
    public static double cosineDistance(double[] vectorA, double[] vectorB) {
        if (vectorA == null || vectorB == null) {
            throw new IllegalArgumentException("Vectors cannot be null");
        }
        if (vectorA.length == 0 || vectorB.length == 0) {
            throw new IllegalArgumentException("Vectors cannot be empty");
        }
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException(
                    String.format("Vector dimensions must match: %d vs %d",
                            vectorA.length, vectorB.length));
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        // Handle edge cases where one or both vectors are zero vectors
        if (normA < EPSILON || normB < EPSILON) {
            log.warn("One or both vectors have near-zero norm (normA={}, normB={})",
                    Math.sqrt(normA), Math.sqrt(normB));
            return 1.0; // Maximum dissimilarity for zero vectors
        }

        double cosineSimilarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

        // Clamp to [-1, 1] to handle numerical precision issues
        cosineSimilarity = Math.max(-1.0, Math.min(1.0, cosineSimilarity));

        return 1.0 - cosineSimilarity;
    }

    /**
     * Calculates the Euclidean distance between two feature vectors.
     *
     * @param vectorA First feature vector
     * @param vectorB Second feature vector
     * @return The Euclidean distance between the vectors
     * @throws IllegalArgumentException if vectors are null, empty, or have different lengths
     */
    public static double euclideanDistance(double[] vectorA, double[] vectorB) {
        if (vectorA == null || vectorB == null) {
            throw new IllegalArgumentException("Vectors cannot be null");
        }
        if (vectorA.length == 0 || vectorB.length == 0) {
            throw new IllegalArgumentException("Vectors cannot be empty");
        }
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException(
                    String.format("Vector dimensions must match: %d vs %d",
                            vectorA.length, vectorB.length));
        }

        double sum = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            double diff = vectorA[i] - vectorB[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /**
     * Calculates the Manhattan (L1) distance between two feature vectors.
     *
     * @param vectorA First feature vector
     * @param vectorB Second feature vector
     * @return The Manhattan distance between the vectors
     * @throws IllegalArgumentException if vectors are null, empty, or have different lengths
     */
    public static double manhattanDistance(double[] vectorA, double[] vectorB) {
        if (vectorA == null || vectorB == null) {
            throw new IllegalArgumentException("Vectors cannot be null");
        }
        if (vectorA.length == 0 || vectorB.length == 0) {
            throw new IllegalArgumentException("Vectors cannot be empty");
        }
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException(
                    String.format("Vector dimensions must match: %d vs %d",
                            vectorA.length, vectorB.length));
        }

        double sum = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            sum += Math.abs(vectorA[i] - vectorB[i]);
        }

        return sum;
    }

    /**
     * Computes the L2 norm (magnitude) of a feature vector.
     *
     * @param vector The feature vector
     * @return The L2 norm of the vector
     * @throws IllegalArgumentException if vector is null or empty
     */
    public static double l2Norm(double[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }

        double sum = 0.0;
        for (double v : vector) {
            sum += v * v;
        }

        return Math.sqrt(sum);
    }

    /**
     * Checks if a vector is normalized (has unit length within tolerance).
     *
     * @param vector The vector to check
     * @param tolerance The tolerance for considering the vector normalized
     * @return true if the vector is normalized within the given tolerance
     */
    public static boolean isNormalized(double[] vector, double tolerance) {
        double norm = l2Norm(vector);
        return Math.abs(norm - 1.0) <= tolerance;
    }

    /**
     * Checks if a vector is normalized with default tolerance (1e-6).
     *
     * @param vector The vector to check
     * @return true if the vector is normalized within default tolerance
     */
    public static boolean isNormalized(double[] vector) {
        return isNormalized(vector, 1e-6);
    }

    /**
     * Calculates basic statistics for a feature vector.
     *
     * @param vector The feature vector
     * @return A Statistics object containing mean, std, min, max values
     */
    public static Statistics calculateStatistics(double[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }

        double sum = 0.0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (double v : vector) {
            sum += v;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }

        double mean = sum / vector.length;

        double varianceSum = 0.0;
        for (double v : vector) {
            double diff = v - mean;
            varianceSum += diff * diff;
        }
        double stdDev = Math.sqrt(varianceSum / vector.length);

        return new Statistics(mean, stdDev, min, max);
    }

    /**
     * Simple statistics container class.
     */
    public static class Statistics {
        public final double mean;
        public final double stdDev;
        public final double min;
        public final double max;

        public Statistics(double mean, double stdDev, double min, double max) {
            this.mean = mean;
            this.stdDev = stdDev;
            this.min = min;
            this.max = max;
        }

        @Override
        public String toString() {
            return String.format("Statistics{mean=%.4f, stdDev=%.4f, min=%.4f, max=%.4f}",
                    mean, stdDev, min, max);
        }
    }
}