package com.retrieval.indexing;

import com.retrieval.models.ImageFeature;

/**
 * Represents a single node in the K-D Tree.
 * Each node stores an ImageFeature and splits the dataset along a specific axis (dimension).
 */
public class KDNode {
    final ImageFeature feature;
    final int axis;
    KDNode left;
    KDNode right;

    public KDNode(ImageFeature feature, int axis) {
        this.feature = feature;
        this.axis = axis;
        this.left = null;
        this.right = null;
    }

    public ImageFeature getFeature() {
        return feature;
    }

    public int getAxis() {
        return axis;
    }

    public KDNode getLeft() {
        return left;
    }

    public KDNode getRight() {
        return right;
    }

    public void setLeft(KDNode left) {
        this.left = left;
    }

    public void setRight(KDNode right) {
        this.right = right;
    }
}
