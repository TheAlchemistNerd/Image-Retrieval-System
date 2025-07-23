package com.retrieval.features.extractor;

import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Updated ORB implementation with fixed averaging loop and proper normalization.
 */
public class ORBExtractor implements Extractable {
    private static final Logger log = LoggerFactory.getLogger(ORBExtractor.class);

    @Override
    public double[] extract(String imagePath) {
        if (imagePath == null || imagePath.trim().isEmpty()) {
            throw new IllegalArgumentException("Image path cannot be null or empty");
        }

        try (Mat image = opencv_imgcodecs.imread(imagePath, opencv_imgcodecs.IMREAD_GRAYSCALE);
             org.bytedeco.opencv.opencv_features2d.ORB orb = org.bytedeco.opencv.opencv_features2d.ORB.create();
             KeyPointVector keypoints = new KeyPointVector();
             Mat descriptors = new Mat()) {

            if (image.empty()) {
                log.error("ORBExtractor: Could not read image {}", imagePath);
                return new double[0];
            }

            // Detect features and compute descriptors
            orb.detectAndCompute(image, new Mat(), keypoints, descriptors);

            if (descriptors.rows() == 0) {
                log.warn("ORBExtractor: No descriptors found in {}", imagePath);
                return new double[0];
            }

            int numDescriptors = (int) descriptors.rows();
            int descriptorLength = (int) descriptors.cols();
            double[] averageDescriptor = new double[descriptorLength];

            // Create indexer for accessing descriptor values
            try (UByteRawIndexer indexer = descriptors.createIndexer()) {
                // Process each descriptor
                for (int row = 0; row < numDescriptors; row++) {
                    double[] currentDescriptor = new double[descriptorLength];

                    // Extract descriptor values
                    for (int col = 0; col < descriptorLength; col++) {
                        currentDescriptor[col] = indexer.get(row, col) & 0xFF; // Convert unsigned byte
                    }

                    // Normalize individual descriptor
                    FeatureUtils.normalize(currentDescriptor);

                    // Add to running average
                    for (int col = 0; col < descriptorLength; col++) {
                        averageDescriptor[col] += currentDescriptor[col];
                    }
                }
            }

            // Complete the averaging
            for (int col = 0; col < descriptorLength; col++) {
                averageDescriptor[col] /= numDescriptors;
            }

            // Final normalization
            FeatureUtils.normalize(averageDescriptor);

            log.debug("ORBExtractor: Processed {} descriptors from {}, final descriptor length: {}",
                    numDescriptors, imagePath, descriptorLength);

            return averageDescriptor;

        } catch (Exception e) {
            log.error("Error extracting ORB features for {}", imagePath, e);
            return new double[0];
        }
    }
}