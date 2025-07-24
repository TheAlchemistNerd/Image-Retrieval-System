package com.retrieval.indexing.BallTree;

/**
 * Internal nodes have left and right children, which are themselves BallTreeNodes.
 * They do not directly store ImageFeatures.
 */
public class InternalNode extends BallTreeNode {
    private BallTreeNode leftChild;
    private BallTreeNode rightChild;

    /**
     * Constructs an internal BallTreeNode.
     *
     * @param centroid The centroid of the ball.
     * @param radius   The radius of the ball.
     */
    public InternalNode(double[] centroid, double radius) {
        super(centroid, radius);
    }

    /**
     * Full constructor with children.
     *
     * @param centroid The centroid of the ball.
     * @param radius   The radius of the ball.
     * @param leftChild The left child node.
     * @param rightChild The right child node.
     */
    public InternalNode(double[] centroid, double radius, BallTreeNode leftChild, BallTreeNode rightChild) {
        super(centroid, radius);
        this.leftChild = leftChild;
        this.rightChild = rightChild;
    }

    /**
     * Gets the left child of this node.
     *
     * @return The left child BallTreeNode, or null if not set.
     */
    public BallTreeNode getLeftChild() {
        return leftChild;
    }

    /**
     * Sets the left child of this node.
     *
     * @param leftChild The BallTreeNode to set as the left child.
     */
    public void setLeftChild(BallTreeNode leftChild) {
        this.leftChild = leftChild;
    }

    /**
     * Gets the right child of this node.
     *
     * @return The right child BallTreeNode, or null if not set.
     */
    public BallTreeNode getRightChild() {
        return rightChild;
    }

    /**
     * Sets the right child of this node.
     *
     * @param rightChild The BallTreeNode to set as the right child.
     */
    public void setRightChild(BallTreeNode rightChild) {
        this.rightChild = rightChild;
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public int getFeatureCount() {
        int count = 0;
        if (leftChild != null) {
            count += leftChild.getFeatureCount();
        }
        if (rightChild != null) {
            count += rightChild.getFeatureCount();
        }
        return count;
    }

    /**
     * Checks if this internal node has both children.
     *
     * @return true if both left and right children are present
     */
    public boolean hasCompleteChildren() {
        return leftChild != null && rightChild != null;
    }

    /**
     * Checks if this internal node has at least one child.
     *
     * @return true if at least one child is present
     */
    public boolean hasAnyChild() {
        return leftChild != null || rightChild != null;
    }

    /**
     * Gets the maximum depth of this subtree.
     *
     * @return The maximum depth from this node to any leaf
     */
    public int getMaxDepth() {
        int leftDepth = leftChild != null ?
                (leftChild.isLeaf() ? 1 : ((InternalNode) leftChild).getMaxDepth() + 1) : 0;
        int rightDepth = rightChild != null ?
                (rightChild.isLeaf() ? 1 : ((InternalNode) rightChild).getMaxDepth() + 1) : 0;
        return Math.max(leftDepth, rightDepth);
    }
}